from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

from .models import (
    AnalysisReport,
    CreativeBrief,
    InputMode,
    OutputFile,
    ProjectDetail,
    ProjectSummary,
    TuningPlan,
    now_iso,
)


class StudioStore:
    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.getenv("CUTETUNER_DATA_DIR")
        self.root = Path(configured or user_data_dir("CUTE Tuner", "Kanishk Paul")).resolve()
        self.projects_root = self.root / "projects"
        self.models_root = self.root / "models"
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "studio.sqlite3"
        self.settings_file = self.root / "settings.json"
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=20)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as database:
            database.execute("PRAGMA journal_mode=WAL")
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    primary_path TEXT NOT NULL,
                    backing_path TEXT,
                    vocal_path TEXT,
                    accompaniment_path TEXT,
                    report_json TEXT,
                    brief_json TEXT,
                    plan_json TEXT,
                    outputs_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    winner TEXT NOT NULL,
                    comment TEXT,
                    context_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
                """
            )
            database.execute(
                "CREATE INDEX IF NOT EXISTS idx_projects_updated_at ON projects(updated_at DESC)"
            )
            database.execute(
                "CREATE INDEX IF NOT EXISTS idx_ratings_project_id ON ratings(project_id)"
            )
            database.execute("PRAGMA optimize")

    def project_dir(self, project_id: str) -> Path:
        if not project_id or any(
            character not in "0123456789abcdef-" for character in project_id.lower()
        ):
            raise ValueError("invalid project id")
        path = (self.projects_root / project_id).resolve()
        if self.projects_root not in path.parents:
            raise ValueError("invalid project path")
        return path

    def create_project(
        self, name: str, mode: InputMode, primary_name: str, backing_name: str | None
    ) -> ProjectDetail:
        project_id = str(uuid.uuid4())
        created = now_iso()
        directory = self.project_dir(project_id)
        directory.mkdir(parents=True)
        primary_path = directory / f"primary{Path(primary_name).suffix.lower()}"
        backing_path = (
            directory / f"backing{Path(backing_name).suffix.lower()}" if backing_name else None
        )
        with self._connect() as database:
            database.execute(
                """INSERT INTO projects
                (id, name, mode, status, primary_path, backing_path, created_at, updated_at)
                VALUES (?, ?, ?, 'uploaded', ?, ?, ?, ?)""",
                (
                    project_id,
                    name,
                    mode.value,
                    str(primary_path),
                    str(backing_path) if backing_path else None,
                    created,
                    created,
                ),
            )
        return self.get_project(project_id)

    def upload_paths(self, project_id: str) -> tuple[Path, Path | None]:
        row = self._row(project_id)
        return Path(row["primary_path"]), Path(row["backing_path"]) if row["backing_path"] else None

    def _row(self, project_id: str) -> sqlite3.Row:
        with self._connect() as database:
            row = database.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(project_id)
        return row

    def list_projects(self) -> list[ProjectSummary]:
        with self._connect() as database:
            rows = database.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [self._summary(row) for row in rows]

    def _summary(self, row: sqlite3.Row) -> ProjectSummary:
        return ProjectSummary(
            id=row["id"],
            name=row["name"],
            mode=row["mode"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            has_report=bool(row["report_json"]),
            has_outputs=bool(json.loads(row["outputs_json"])),
        )

    def get_project(self, project_id: str) -> ProjectDetail:
        row = self._row(project_id)
        summary = self._summary(row)
        return ProjectDetail(
            **summary.model_dump(),
            report=AnalysisReport.model_validate_json(row["report_json"])
            if row["report_json"]
            else None,
            brief=CreativeBrief.model_validate_json(row["brief_json"])
            if row["brief_json"]
            else None,
            plan=TuningPlan.model_validate_json(row["plan_json"]) if row["plan_json"] else None,
            outputs=[OutputFile.model_validate(item) for item in json.loads(row["outputs_json"])],
        )

    def paths(self, project_id: str) -> dict[str, Path | None]:
        row = self._row(project_id)
        return {
            key: Path(row[key]) if row[key] else None
            for key in ("primary_path", "backing_path", "vocal_path", "accompaniment_path")
        }

    def update(self, project_id: str, **values: Any) -> ProjectDetail:
        allowed = {
            "status",
            "vocal_path",
            "accompaniment_path",
            "report_json",
            "brief_json",
            "plan_json",
            "outputs_json",
        }
        if not values or set(values) - allowed:
            raise ValueError("unsupported project update")
        values["updated_at"] = now_iso()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self._connect() as database:
            database.execute(
                f"UPDATE projects SET {assignments} WHERE id = ?",
                (*values.values(), project_id),
            )
        return self.get_project(project_id)

    def add_rating(
        self, project_id: str, winner: str, comment: str | None, context: dict[str, object]
    ) -> int:
        self._row(project_id)
        with self._connect() as database:
            cursor = database.execute(
                "INSERT INTO ratings(project_id, winner, comment, context_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (project_id, winner, comment, json.dumps(context), now_iso()),
            )
            return int(cursor.lastrowid)

    def rating_count(self) -> int:
        with self._connect() as database:
            return int(database.execute("SELECT COUNT(*) FROM ratings").fetchone()[0])

    def ratings(self) -> list[dict[str, object]]:
        with self._connect() as database:
            rows = database.execute(
                "SELECT winner, comment, context_json, created_at FROM ratings ORDER BY id"
            ).fetchall()
        return [
            {
                "winner": row["winner"],
                "comment": row["comment"],
                "context": json.loads(row["context_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def settings(self) -> dict[str, object]:
        defaults: dict[str, object] = {
            "llm_endpoint": None,
            "llm_model": None,
            "keep_masters": True,
        }
        if not self.settings_file.is_file():
            return defaults
        try:
            values = json.loads(self.settings_file.read_text())
        except (OSError, json.JSONDecodeError):
            return defaults
        return {
            "llm_endpoint": values.get("llm_endpoint"),
            "llm_model": values.get("llm_model"),
            "keep_masters": bool(values.get("keep_masters", True)),
        }

    def save_settings(self, **changes: object) -> dict[str, object]:
        if set(changes) - {"llm_endpoint", "llm_model", "keep_masters"}:
            raise ValueError("unsupported setting")
        values = self.settings()
        values.update(changes)
        temporary = self.settings_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2))
        temporary.replace(self.settings_file)
        return values

    def delete_project(self, project_id: str) -> None:
        directory = self.project_dir(project_id)
        self._row(project_id)
        with self._connect() as database:
            database.execute("DELETE FROM ratings WHERE project_id = ?", (project_id,))
            database.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        if directory.exists():
            shutil.rmtree(directory)

    def delete_all(self, include_models: bool = False) -> None:
        for project in self.list_projects():
            self.delete_project(project.id)
        (self.root / "preference.json").unlink(missing_ok=True)
        self.settings_file.unlink(missing_ok=True)
        if include_models:
            for directory in (self.models_root, self.root / "runtimes"):
                if directory.exists():
                    shutil.rmtree(directory)
            self.models_root.mkdir(parents=True, exist_ok=True)
