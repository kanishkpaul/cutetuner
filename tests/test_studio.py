from __future__ import annotations

import io
import json
import shutil
import subprocess
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voiceforge.api import create_app
from voiceforge.config import CorrectionConfig
from voiceforge.models import ChordEvent, NoteEvent
from voiceforge.pitch import _viterbi_path
from voiceforge.preference import LABELS, PreferenceRanker
from voiceforge.studio import chord_aware_targets, select_separator_stems


@pytest.fixture(autouse=True)
def use_lightweight_analysis_for_generated_fixtures(monkeypatch) -> None:
    monkeypatch.setattr("voiceforge.studio.memory_free_percent", lambda: 0)


def vocal_wav(duration: float = 1.2, frequency: float = 222.0) -> bytes:
    sample_rate = 44_100
    clock = np.arange(round(sample_rate * duration)) / sample_rate
    envelope = np.minimum(1, clock * 12) * np.minimum(1, (duration - clock) * 12)
    vibrato = 1.4 * np.sin(2 * np.pi * 5.2 * clock)
    phase = 2 * np.pi * np.cumsum(frequency * np.exp2(vibrato / 1200)) / sample_rate
    audio = 0.28 * envelope * np.sin(phase)
    stream = io.BytesIO()
    sf.write(stream, audio, sample_rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def wait_for_job(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(1_200):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in {"complete", "failed", "cancelled"}:
            return state
        time.sleep(0.02)
    raise AssertionError("background audio job did not complete")


def test_separator_routes_other_stem_without_confusing_model_name(tmp_path) -> None:
    other = tmp_path / "song_(other)_vocals_mel_band_roformer.wav"
    vocals = tmp_path / "song_(vocals)_vocals_mel_band_roformer.wav"
    selected_vocal, selected_backing = select_separator_stems([other, vocals])
    assert selected_vocal == vocals
    assert selected_backing == other


def test_banded_viterbi_never_makes_an_impossible_crepe_jump() -> None:
    probabilities = np.full((4, 360), 1e-6)
    probabilities[0, 100] = 1
    probabilities[1, 104] = 1
    probabilities[2, 250] = 1
    probabilities[3, 110] = 1
    path = _viterbi_path(probabilities)
    assert len(path) == 4
    assert np.max(np.abs(np.diff(path))) <= 11


def create_vocal_project(client: TestClient) -> str:
    response = client.post(
        "/api/projects",
        data={"name": "Private take", "mode": "vocal_only"},
        files={"primary": ("../../unsafe name.wav", vocal_wav(), "audio/wav")},
    )
    assert response.status_code == 201
    project = response.json()
    assert project["name"] == "Private take"
    job = client.post(f"/api/projects/{project['id']}/analyze").json()
    state = wait_for_job(client, job["id"])
    assert state["status"] == "complete", state
    return str(project["id"])


def test_vocal_only_journey_and_mp3_export(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", frontend_dir=tmp_path / "missing")
    with TestClient(app) as client:
        project_id = create_vocal_project(client)
        project = client.get(f"/api/projects/{project_id}").json()
        assert project["report"]["notes"]
        assert project["report"]["scale"] is None
        assert any("No harmonic context" in warning for warning in project["report"]["warnings"])

        brief = {
            "vibe": "intimate and aching",
            "delivery": "close and honest",
            "lyrics": "stay with me",
            "reference_description": "warm piano ballad",
            "tuning_intent": "transparent",
            "protected_ranges": [{"start": 0.2, "end": 0.32}],
        }
        response = client.put(f"/api/projects/{project_id}/brief", json=brief)
        assert response.status_code == 200
        assert response.json()["controls"]["scale"] == "chromatic"

        preview_job = client.post(f"/api/projects/{project_id}/previews").json()
        assert wait_for_job(client, preview_job["id"])["status"] == "complete"
        preview = client.get(f"/api/projects/{project_id}/outputs/preview_recommended")
        assert preview.status_code == 200
        assert preview.content[:3] in {b"ID3", b"\xff\xfb", b"\xff\xf3"}

        render_job = client.post(f"/api/projects/{project_id}/render", json={}).json()
        state = wait_for_job(client, render_job["id"])
        assert state["status"] == "complete", state
        exported = client.get(f"/api/projects/{project_id}/outputs/corrected_vocal")
        assert exported.status_code == 200
        assert "corrected-vocal.mp3" in exported.headers["content-disposition"]

        output = tmp_path / "corrected.mp3"
        output.write_bytes(exported.content)
        probe = json.loads(
            subprocess.run(
                [
                    shutil.which("ffprobe") or "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_name,bit_rate,channels:format=duration",
                    "-of",
                    "json",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        stream = probe["streams"][0]
        assert stream["codec_name"] == "mp3"
        assert stream["bit_rate"] == "320000"
        assert stream["channels"] == 1
        assert abs(float(probe["format"]["duration"]) - 1.2) <= 0.02
        decoded = subprocess.run(
            [
                shutil.which("ffmpeg") or "ffmpeg",
                "-v",
                "error",
                "-i",
                str(output),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        assert float(np.max(np.abs(np.frombuffer(decoded, dtype="<f4")))) < 1.0

        assert client.delete(f"/api/projects/{project_id}").status_code == 204
        assert not client.get("/api/projects").json()


def test_upload_validation_and_loopback_health(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", frontend_dir=tmp_path / "missing")
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health == {"status": "ok", "local_only": True}
        missing_backing = client.post(
            "/api/projects",
            data={"name": "Pair", "mode": "vocal_backing"},
            files={"primary": ("voice.wav", vocal_wav(), "audio/wav")},
        )
        assert missing_backing.status_code == 422
        unsupported = client.post(
            "/api/projects",
            data={"name": "Bad", "mode": "vocal_only"},
            files={"primary": ("voice.txt", b"not audio", "text/plain")},
        )
        assert unsupported.status_code == 415
        assert client.get("/api/projects/not-a-project").status_code == 404
        assert (
            client.put(
                "/api/settings/llm",
                json={"endpoint": "https://example.com/v1", "model": "remote"},
            ).status_code
            == 422
        )
        assert (
            client.put(
                "/api/settings/llm",
                json={"endpoint": "http://127.0.0.1:8080/v1", "model": "local-gguf"},
            ).status_code
            == 200
        )


def test_vocal_and_backing_exports_both_mp3_files(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", frontend_dir=tmp_path / "missing")
    with TestClient(app) as client:
        audio = vocal_wav(duration=1.0)
        response = client.post(
            "/api/projects",
            data={"name": "Song pair", "mode": "vocal_backing"},
            files={
                "primary": ("vocal.wav", audio, "audio/wav"),
                "backing": ("backing.wav", vocal_wav(duration=1.0, frequency=261.63), "audio/wav"),
            },
        )
        assert response.status_code == 201
        project_id = response.json()["id"]
        job = client.post(f"/api/projects/{project_id}/analyze").json()
        assert wait_for_job(client, job["id"])["status"] == "complete"
        brief = {
            "vibe": "warm and clean",
            "delivery": "controlled",
            "tuning_intent": "balanced",
            "protected_ranges": [],
        }
        assert client.put(f"/api/projects/{project_id}/brief", json=brief).status_code == 200
        job = client.post(f"/api/projects/{project_id}/render", json={}).json()
        assert wait_for_job(client, job["id"])["status"] == "complete"
        project = client.get(f"/api/projects/{project_id}").json()
        assert {output["kind"] for output in project["outputs"]} == {
            "corrected_vocal",
            "finished_mix",
        }
        assert client.get(f"/api/projects/{project_id}/outputs/finished_mix").status_code == 200


def test_preference_ranker_waits_for_twenty_private_choices(tmp_path) -> None:
    ranker = PreferenceRanker(tmp_path / "preference.json")
    ratings = [
        {
            "winner": LABELS[index % len(LABELS)],
            "context": {
                "strength": 0.35 + index * 0.02,
                "retune_ms": 150 - index,
                "pitch_span": 8 + index % 4,
            },
        }
        for index in range(20)
    ]
    assert not ranker.train(ratings[:19])
    assert ranker.train(ratings)
    assert ranker.recommend({"strength": 0.55, "retune_ms": 110}) in LABELS


def test_chord_aware_sequence_prefers_harmonic_candidate() -> None:
    notes = [
        NoteEvent(start=0, end=1, midi=63.55, confidence=0.9),
        NoteEvent(start=1, end=2, midi=67.1, confidence=0.9),
    ]
    chords = [ChordEvent(start=0, end=2, label="C", confidence=0.8)]
    targets = chord_aware_targets(
        notes,
        chords,
        CorrectionConfig(key="C", scale="major"),
        max_correction_cents=90,
    )
    assert targets == [64, 67]
