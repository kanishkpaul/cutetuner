from __future__ import annotations

import argparse
import json
from pathlib import Path

DESCRIPTORS = {
    "intimate and close": "an intimate close-miked singing voice",
    "clear and present": "a clear present singing voice",
    "warm and rich": "a warm rich singing voice",
    "dark and rounded": "a dark rounded singing voice",
    "bright and airy": "a bright airy singing voice",
    "breathy and fragile": "a breathy fragile singing voice",
    "restrained and emotional": "a restrained emotional singing performance",
    "energetic and forceful": "an energetic forceful singing performance",
    "raw and natural": "a raw natural unprocessed singing voice",
    "polished and modern": "a polished modern studio singing voice",
}


def clap_analysis(audio_path: Path, model_root: Path, vibe: str | None) -> dict[str, object]:
    import librosa
    import numpy as np
    import torch
    from transformers import ClapModel, ClapProcessor

    model_path = model_root / "clap" / "larger_clap_music_and_speech"
    processor = ClapProcessor.from_pretrained(model_path, local_files_only=True)
    model = ClapModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    audio, _ = librosa.load(audio_path, sr=48_000, mono=True, duration=30)
    audio_inputs = processor(audio=audio, sampling_rate=48_000, return_tensors="pt")
    prompts = list(DESCRIPTORS.values())
    labels = list(DESCRIPTORS)
    if vibe:
        prompts.append(f"a singing performance that feels {vibe}")
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.inference_mode():
        audio_features = model.get_audio_features(**audio_inputs)
        text_features = model.get_text_features(**text_inputs)
    audio_features = getattr(audio_features, "pooler_output", audio_features)
    text_features = getattr(text_features, "pooler_output", text_features)
    audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    similarities = (audio_features @ text_features.T).squeeze(0).cpu().numpy()
    descriptor_scores = similarities[: len(labels)]
    top = np.argsort(descriptor_scores)[::-1][:3]
    result: dict[str, object] = {
        "descriptors": [labels[int(index)] for index in top],
        "scores": {labels[int(index)]: round(float(descriptor_scores[index]), 4) for index in top},
        "confidence": round(
            float(
                np.clip(
                    (descriptor_scores[top[0]] - descriptor_scores[top[-1]]) * 2 + 0.55, 0.35, 0.9
                )
            ),
            3,
        ),
        "source": "LAION larger CLAP music and speech",
    }
    if vibe:
        result["vibe_similarity"] = round(float(np.clip((similarities[-1] + 1) / 2, 0, 1)), 3)
    return result


def _representative_chunks(audio, sample_rate: int, seconds: int = 15):
    import numpy as np

    width = sample_rate * seconds
    if len(audio) <= width:
        return [audio]
    starts = np.linspace(0, len(audio) - width, min(3, max(1, len(audio) // width)), dtype=int)
    return [audio[int(start) : int(start) + width] for start in starts]


def singmos_analysis(audio_path: Path, model_root: Path) -> dict[str, object]:
    import librosa
    import numpy as np
    import torch
    import torchaudio
    from s3prl.util.download import set_dir as set_s3prl_download_dir

    # S3PRL 0.4.18 still calls this legacy no-op while importing optional upstreams.
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda _backend: None
    set_s3prl_download_dir(model_root / "s3prl" / "download")

    repository = model_root / "singmos" / "repository"
    checkpoint = (
        model_root / "singmos" / "ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth"
    )
    predictor = torch.hub.load(
        str(repository),
        "singmos_pro",
        source="local",
        trust_repo=True,
        pretrained=False,
        model_path=str(checkpoint),
    )
    predictor.eval()
    audio, _ = librosa.load(audio_path, sr=16_000, mono=True)
    scores: list[float] = []
    with torch.inference_mode():
        for chunk in _representative_chunks(audio, 16_000):
            wave = torch.from_numpy(np.asarray(chunk, dtype=np.float32)).unsqueeze(0)
            length = torch.tensor([wave.shape[1]], dtype=torch.long)
            scores.append(float(predictor(wave, length).item()))
    return {
        "score": round(float(np.mean(scores)), 3),
        "confidence": round(min(0.82, 0.62 + 0.06 * len(scores)), 3),
        "windows": len(scores),
        "source": "SingMOS-Pro v1.1.2 prediction",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("clap", "singmos"))
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--vibe")
    args = parser.parse_args()
    if args.mode == "clap":
        result = clap_analysis(args.audio, args.model_root, args.vibe)
    else:
        result = singmos_analysis(args.audio, args.model_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
