from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


class QualityModelError(RuntimeError):
    pass


class QualityRunner:
    """Run incompatible perception models one at a time in an isolated environment."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.python = data_root / "runtimes" / "perception" / "bin" / "python"
        self.worker = Path(__file__).with_name("quality_worker.py")
        self.models = data_root / "models"
        self.clap_model = self.models / "clap" / "larger_clap_music_and_speech"
        self.singmos_repo = self.models / "singmos" / "repository"
        self.singmos_checkpoint = (
            self.models / "singmos" / "ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth"
        )
        self.s3prl_checkpoint = (
            self.models
            / "s3prl"
            / "download"
            / "70a1c0d7bd4d235fe73bc359f70cc91a165d2405f6608fb01c504b8ba1244ac6.wav2vec_vox_new.pt"
        )

    @property
    def clap_ready(self) -> bool:
        return self.python.is_file() and (self.clap_model / "pytorch_model.bin").is_file()

    @property
    def singmos_ready(self) -> bool:
        return (
            self.python.is_file()
            and (self.singmos_repo / "hubconf.py").is_file()
            and self.singmos_checkpoint.is_file()
            and self.s3prl_checkpoint.is_file()
        )

    def _run(self, mode: str, audio_path: Path, vibe: str | None = None) -> dict[str, Any]:
        if not self.python.is_file():
            raise QualityModelError("The perception runtime is not installed.")
        command = [
            str(self.python),
            str(self.worker),
            mode,
            "--audio",
            str(audio_path),
            "--model-root",
            str(self.models),
        ]
        if vibe:
            command.extend(["--vibe", vibe])
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HOME": str(self.models / "huggingface"),
                "TORCH_HOME": str(self.models / "torch"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
                env=environment,
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired as error:
            raise QualityModelError(f"{mode} timed out after ten minutes") from error
        except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError) as error:
            detail = getattr(error, "stderr", "") or str(error)
            raise QualityModelError(f"{mode} failed: {detail[-800:]}") from error
        if not isinstance(payload, dict):
            raise QualityModelError(f"{mode} returned an invalid result")
        return payload

    def clap(self, audio_path: Path, vibe: str | None = None) -> dict[str, Any]:
        if not self.clap_ready:
            raise QualityModelError("The CLAP checkpoint is not installed.")
        return self._run("clap", audio_path, vibe)

    def singmos(self, audio_path: Path) -> dict[str, Any]:
        if not self.singmos_ready:
            raise QualityModelError("The SingMOS-Pro checkpoint is not installed.")
        return self._run("singmos", audio_path)
