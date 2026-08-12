from __future__ import annotations

import subprocess
from pathlib import Path

from voiceforge.quality import QualityRunner


def test_quality_runner_requires_every_offline_checkpoint(tmp_path) -> None:
    runner = QualityRunner(tmp_path)
    assert not runner.clap_ready
    assert not runner.singmos_ready

    runner.python.parent.mkdir(parents=True)
    runner.python.touch()
    (runner.clap_model / "pytorch_model.bin").parent.mkdir(parents=True)
    (runner.clap_model / "pytorch_model.bin").touch()
    (runner.singmos_repo / "hubconf.py").parent.mkdir(parents=True)
    (runner.singmos_repo / "hubconf.py").touch()
    runner.singmos_checkpoint.touch()
    assert runner.clap_ready
    assert not runner.singmos_ready

    runner.s3prl_checkpoint.parent.mkdir(parents=True)
    runner.s3prl_checkpoint.touch()
    assert runner.singmos_ready


def test_quality_runner_uses_offline_subprocess(monkeypatch, tmp_path) -> None:
    runner = QualityRunner(tmp_path)
    runner.python.parent.mkdir(parents=True)
    runner.python.touch()
    (runner.clap_model / "pytorch_model.bin").parent.mkdir(parents=True)
    (runner.clap_model / "pytorch_model.bin").touch()
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, '{"descriptors": ["warm and rich"]}\n', "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.clap(Path("voice.wav"), "intimate")
    assert result["descriptors"] == ["warm and rich"]
    environment = captured["env"]
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
