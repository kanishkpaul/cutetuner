"""Optional tiny denoiser for *your own* vocal recordings.

The pitch corrector is intentionally independent of this module. Train a model only
after it has a defined, measurable job such as reducing room noise or harshness.
"""


def make_tiny_mask_net():
    try:
        from torch import nn
    except ImportError as error:
        raise RuntimeError("Install neural extras first: uv sync --extra neural") from error

    class TinyMaskNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid(),
            )

        def forward(self, log_magnitude):
            return self.layers(log_magnitude)

    return TinyMaskNet()
