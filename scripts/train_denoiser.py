"""Train a small enhancement mask on consented, clean recordings only.

Put 10-30 minutes of your best dry takes in data/clean. The script corrupts them
synthetically, so it does not need another person's voice or a cloud data set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from voiceforge.audio import read_mono
from voiceforge.neural import make_tiny_mask_net


def chunks(root: Path, sample_rate: int, samples: int) -> list[np.ndarray]:
    clips: list[np.ndarray] = []
    for path in sorted(root.glob("*.wav")):
        audio, _ = read_mono(path, sample_rate)
        for start in range(0, max(0, len(audio) - samples + 1), samples):
            segment = audio[start : start + samples]
            if np.max(np.abs(segment)) > 0.02:
                clips.append(segment)
    return clips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    args = parser.parse_args()
    import torch
    from torch.nn import functional

    length = args.sample_rate * 2
    clean = chunks(Path(args.clean_dir), args.sample_rate, length)
    if len(clean) < 8:
        raise SystemExit("Need at least eight non-silent 2-second chunks of your own dry vocals.")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_tiny_mask_net().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    window = torch.hann_window(512, device=device)
    rng = np.random.default_rng(7)
    for epoch in range(args.epochs):
        losses = []
        for target_np in rng.permutation(clean):
            target = torch.tensor(target_np, device=device).unsqueeze(0)
            noise = torch.randn_like(target) * float(rng.uniform(0.004, 0.03))
            noisy = target + noise
            target_spec = torch.stft(target, 512, 128, window=window, return_complex=True)
            noisy_spec = torch.stft(noisy, 512, 128, window=window, return_complex=True)
            mask = model(torch.log1p(noisy_spec.abs()).unsqueeze(1)).squeeze(1)
            estimate = noisy_spec * mask
            loss = functional.l1_loss(estimate.abs(), target_spec.abs())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} spectral_l1={np.mean(losses):.5f}")
    Path("models").mkdir(exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "sample_rate": args.sample_rate},
        "models/tiny-denoiser.pt",
    )


if __name__ == "__main__":
    main()
