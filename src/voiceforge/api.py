from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    CreativeBrief,
    InputMode,
    JobState,
    LocalLLMConfig,
    RatingRequest,
    RenderRequest,
    TuningControls,
    now_iso,
)
from .preference import PreferenceRanker
from .storage import StudioStore
from .studio import SUPPORTED_EXTENSIONS, StudioEngine

MAX_UPLOAD_BYTES = 500 * 1024 * 1024


class Cancelled(RuntimeError):
    pass


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, JobState] = {}
        self.cancellations: dict[str, threading.Event] = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cutetuner")
        self.lock = threading.RLock()

    def start(self, project_id: str, kind: str, work) -> JobState:
        job_id = str(uuid.uuid4())
        state = JobState(
            id=job_id,
            project_id=project_id,
            kind=kind,
            stage="queued",
            progress=0,
            status="queued",
            message="Waiting for the local audio worker",
        )
        cancellation = threading.Event()
        with self.lock:
            self.jobs[job_id] = state
            self.cancellations[job_id] = cancellation

        def update(stage: str, progress: float, message: str) -> None:
            if cancellation.is_set():
                raise Cancelled("Job cancelled")
            with self.lock:
                current = self.jobs[job_id]
                self.jobs[job_id] = current.model_copy(
                    update={
                        "stage": stage,
                        "progress": progress,
                        "message": message,
                        "status": "running" if progress < 1 else "complete",
                        "updated_at": now_iso(),
                    }
                )

        def runner() -> None:
            update("starting", 0.01, "Starting local processing")
            try:
                work(update)
                update("complete", 1, "Complete")
            except Cancelled:
                with self.lock:
                    current = self.jobs[job_id]
                    self.jobs[job_id] = current.model_copy(
                        update={
                            "status": "cancelled",
                            "message": "Cancelled",
                            "updated_at": now_iso(),
                        }
                    )
            except Exception as error:  # noqa: BLE001 -- background jobs must report all failures to the UI
                with self.lock:
                    current = self.jobs[job_id]
                    self.jobs[job_id] = current.model_copy(
                        update={
                            "status": "failed",
                            "message": "Processing failed",
                            "error": str(error),
                            "updated_at": now_iso(),
                        }
                    )

        self.executor.submit(runner)
        return state

    def get(self, job_id: str) -> JobState:
        with self.lock:
            state = self.jobs.get(job_id)
        if not state:
            raise KeyError(job_id)
        return state

    def cancel(self, job_id: str) -> JobState:
        self.get(job_id)
        self.cancellations[job_id].set()
        return self.get(job_id)


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type: {extension or 'unknown'}")
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                target.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(413, "Audio files are limited to 500 MB.")
            target.write(chunk)


def create_app(
    data_dir: str | Path | None = None, frontend_dir: str | Path | None = None
) -> FastAPI:
    store = StudioStore(data_dir)
    engine = StudioEngine(store)
    ranker = PreferenceRanker(store.root / "preference.json")
    jobs = JobManager()
    app = FastAPI(title="CUTE Tuner Local Studio", version="0.2.0")
    app.state.store = store
    app.state.engine = engine
    app.state.jobs = jobs
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "local_only": True}

    @app.get("/api/system")
    def system():
        settings = store.settings()
        return {
            "models": engine.model_status(),
            "rating_count": store.rating_count(),
            "model_budget_gb": 5,
            "llm_config": {
                "endpoint": settings.get("llm_endpoint"),
                "model": settings.get("llm_model"),
            },
            "keep_masters": settings.get("keep_masters", True),
        }

    @app.put("/api/settings/llm")
    def configure_llm(config: LocalLLMConfig):
        endpoint = config.endpoint.strip().rstrip("/") if config.endpoint else None
        model = config.model.strip() if config.model else None
        if bool(endpoint) != bool(model):
            raise HTTPException(422, "Provide both a local endpoint and model name, or clear both.")
        if endpoint:
            parsed = urlparse(endpoint)
            if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise HTTPException(422, "The producer model endpoint must be local and use HTTP.")
        store.save_settings(llm_endpoint=endpoint, llm_model=model)
        return {"endpoint": endpoint, "model": model}

    @app.put("/api/settings/masters")
    def configure_masters(keep: bool):
        store.save_settings(keep_masters=keep)
        return {"keep_masters": keep}

    @app.get("/api/projects")
    def list_projects():
        return store.list_projects()

    @app.post("/api/projects", status_code=201)
    async def create_project(
        name: Annotated[str, Form()],
        mode: Annotated[InputMode, Form()],
        primary: Annotated[UploadFile, File()],
        backing: Annotated[UploadFile | None, File()] = None,
    ):
        if mode == InputMode.VOCAL_BACKING and backing is None:
            raise HTTPException(422, "Dry vocal + backing mode requires both files.")
        project = store.create_project(
            name.strip()[:120] or "Untitled vocal",
            mode,
            primary.filename or "audio.wav",
            backing.filename if backing else None,
        )
        primary_path, backing_path = store.upload_paths(project.id)
        try:
            await _save_upload(primary, primary_path)
            if backing and backing_path:
                await _save_upload(backing, backing_path)
        except Exception:
            store.delete_project(project.id)
            raise
        return store.get_project(project.id)

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        try:
            return store.get_project(project_id)
        except (KeyError, ValueError) as error:
            raise HTTPException(404, "Project not found") from error

    @app.delete("/api/projects/{project_id}", status_code=204)
    def delete_project(project_id: str):
        try:
            store.delete_project(project_id)
        except (KeyError, ValueError) as error:
            raise HTTPException(404, "Project not found") from error

    @app.delete("/api/projects", status_code=204)
    def delete_all_projects():
        store.delete_all(include_models=True)

    @app.post("/api/projects/{project_id}/analyze", status_code=202)
    def analyze(project_id: str):
        try:
            store.get_project(project_id)
        except (KeyError, ValueError) as error:
            raise HTTPException(404, "Project not found") from error
        return jobs.start(
            project_id, "analysis", lambda progress: engine.analyze(project_id, progress)
        )

    @app.put("/api/projects/{project_id}/brief")
    def submit_brief(project_id: str, brief: CreativeBrief):
        try:
            return engine.create_plan(project_id, brief)
        except KeyError as error:
            raise HTTPException(404, "Project not found") from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(409, str(error)) from error

    @app.put("/api/projects/{project_id}/plan")
    def update_plan(project_id: str, controls: TuningControls):
        try:
            project = store.get_project(project_id)
            if not project.brief:
                raise RuntimeError("Submit the producer brief first.")
            return engine.create_plan(project_id, project.brief, controls)
        except KeyError as error:
            raise HTTPException(404, "Project not found") from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/projects/{project_id}/previews", status_code=202)
    def previews(project_id: str):
        try:
            store.get_project(project_id)
        except KeyError as error:
            raise HTTPException(404, "Project not found") from error
        return jobs.start(
            project_id, "preview", lambda progress: engine.render_previews(project_id, progress)
        )

    @app.post("/api/projects/{project_id}/render", status_code=202)
    def render(project_id: str, request: RenderRequest):
        try:
            store.get_project(project_id)
        except KeyError as error:
            raise HTTPException(404, "Project not found") from error
        return jobs.start(
            project_id,
            "render",
            lambda progress: engine.render(project_id, progress, request.controls),
        )

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return jobs.get(job_id)
        except KeyError as error:
            raise HTTPException(404, "Job not found") from error

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            return jobs.cancel(job_id)
        except KeyError as error:
            raise HTTPException(404, "Job not found") from error

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str):
        try:
            jobs.get(job_id)
        except KeyError as error:
            raise HTTPException(404, "Job not found") from error

        def stream():
            last = ""
            while True:
                state = jobs.get(job_id)
                payload = state.model_dump_json()
                if payload != last:
                    yield f"data: {payload}\n\n"
                    last = payload
                if state.status in {"complete", "failed", "cancelled"}:
                    break
                time.sleep(0.2)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/projects/{project_id}/outputs/{kind}")
    def download_output(project_id: str, kind: str):
        try:
            project = store.get_project(project_id)
        except KeyError as error:
            raise HTTPException(404, "Project not found") from error
        output = next((item for item in project.outputs if item.kind == kind), None)
        if output is None:
            raise HTTPException(404, "Output not found")
        folder = "previews" if kind.startswith("preview_") else "outputs"
        path = (store.project_dir(project_id) / folder / output.filename).resolve()
        if store.project_dir(project_id) not in path.parents or not path.is_file():
            raise HTTPException(404, "Output file is missing")
        return FileResponse(path, media_type=output.media_type, filename=output.filename)

    @app.get("/api/projects/{project_id}/audio/{kind}")
    def project_audio(project_id: str, kind: str):
        try:
            paths = store.paths(project_id)
        except KeyError as error:
            raise HTTPException(404, "Project not found") from error
        mapping = {
            "original": paths["primary_path"],
            "vocal": paths["vocal_path"] or paths["primary_path"],
            "backing": paths["accompaniment_path"] or paths["backing_path"],
        }
        path = mapping.get(kind)
        if path is None or not path.is_file():
            raise HTTPException(404, "Audio source not found")
        if store.project_dir(project_id) not in path.resolve().parents:
            raise HTTPException(404, "Audio source not found")
        return FileResponse(path)

    @app.post("/api/projects/{project_id}/ratings", status_code=201)
    def rate(project_id: str, rating: RatingRequest):
        try:
            project = store.get_project(project_id)
            context = dict(rating.context)
            if project.plan:
                context.update(project.plan.controls.model_dump())
            if project.report:
                context["pitch_span"] = project.report.vocal["pitch_span"].value
                context["note_error"] = project.report.vocal["median_note_error"].value
            rating_id = store.add_rating(project_id, rating.winner, rating.comment, context)
        except KeyError as error:
            raise HTTPException(404, "Project not found") from error
        trained = ranker.train(store.ratings())
        return {
            "id": rating_id,
            "rating_count": store.rating_count(),
            "personalization_ready": store.rating_count() >= 20,
            "model_updated": trained,
            "recommended_variant": ranker.recommend(context),
        }

    selected_frontend = (
        Path(frontend_dir)
        if frontend_dir
        else Path(__file__).resolve().parents[2] / "frontend" / "dist"
    )
    if selected_frontend.is_dir():
        app.mount("/", StaticFiles(directory=selected_frontend, html=True), name="studio")
    return app


app = create_app()
