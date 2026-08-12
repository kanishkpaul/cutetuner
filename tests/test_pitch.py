import numpy as np

from voiceforge.config import CorrectionConfig
from voiceforge.pitch import estimate_pitch, hz_to_midi, smooth_pitch, target_pitch
from voiceforge.song import detect_key, parse_key


def test_estimate_220_hz_sine() -> None:
    sr = 44_100
    audio = 0.4 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    _, f0, confidence = estimate_pitch(audio, sr, CorrectionConfig())
    voiced = f0 > 0
    assert abs(np.median(f0[voiced]) - 220) < 1.0
    assert np.median(confidence[voiced]) > 0.9


def test_g_minor_snaps_a_sharp_to_b_flat() -> None:
    target = target_pitch(
        np.array([466.1637615]), CorrectionConfig(key="G", scale="minor", strength=1.0)
    )
    assert abs(hz_to_midi(target)[0] - 70.0) < 0.01


def test_smoothing_preserves_unvoiced_frames() -> None:
    smoothed = smooth_pitch(
        np.array([220.0, 440.0, 0.0, 440.0]), np.arange(4) * 0.01, CorrectionConfig(retune_ms=80)
    )
    assert smoothed[2] == 0
    assert 220 < smoothed[1] < 440


def test_key_parser() -> None:
    assert parse_key("Bb major") == ("BB", "major")


def test_detects_simple_c_major_backing(tmp_path) -> None:
    from voiceforge.audio import write_wav

    sr = 44_100
    time = np.arange(sr * 2) / sr
    chord = sum(np.sin(2 * np.pi * hz * time) for hz in (130.81, 164.81, 196.0, 261.63))
    path = tmp_path / "c-major.wav"
    write_wav(path, chord / 4, sr)
    estimate = detect_key(path)
    assert estimate.key == "C"
    assert estimate.scale == "major"


def test_auto_cli_routes_local_llm_flag(monkeypatch, capsys) -> None:
    from voiceforge import cli

    received = {}

    def fake_auto(vocal, song, output, **kwargs):
        received.update(vocal=vocal, song=song, output=output, **kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli, "auto_tune_file", fake_auto)
    monkeypatch.setattr(
        "sys.argv",
        ["cutetuner", "auto", "vocal.wav", "song.wav", "out.wav", "--vibe", "warm", "--local-llm"],
    )
    cli.main()
    assert received == {
        "vocal": "vocal.wav",
        "song": "song.wav",
        "output": "out.wav",
        "vibe": "warm",
        "key_override": None,
        "use_local_llm": True,
    }
    assert '"ok": true' in capsys.readouterr().out


def test_local_llm_rejects_non_loopback_endpoint(monkeypatch) -> None:
    from voiceforge.director import local_llm_profile

    monkeypatch.setenv("VOICEFORGE_LLM_URL", "https://example.com/v1")
    monkeypatch.setenv("VOICEFORGE_LLM_MODEL", "remote")
    assert local_llm_profile("warm", {}) is None
