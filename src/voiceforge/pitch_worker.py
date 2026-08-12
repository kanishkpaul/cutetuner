from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one bounded torchcrepe inference chunk")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-rate", required=True, type=int)
    parser.add_argument("--hop-length", required=True, type=int)
    args = parser.parse_args()

    try:
        os.nice(10)
    except OSError:
        pass

    import torch
    import torchcrepe

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    audio = np.load(args.input, allow_pickle=False).astype(np.float32, copy=False)
    samples = torch.from_numpy(audio).unsqueeze(0)
    activation_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for frames in torchcrepe.preprocess(
            samples,
            args.sample_rate,
            args.hop_length,
            batch_size=32,
            device="cpu",
            pad=True,
        ):
            activations = torchcrepe.infer(frames, model="full", device="cpu")
            activation_parts.append(activations.cpu().numpy().astype(np.float32))
    np.save(args.output, np.concatenate(activation_parts, axis=0), allow_pickle=False)


if __name__ == "__main__":
    main()
