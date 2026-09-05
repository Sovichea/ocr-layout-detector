from __future__ import annotations

from torch import nn


OUTPUT_STRIDE = 4


class TinyLayoutNet(nn.Module):
    """Tiny fully convolutional region and text-line detector.

    Output channels at stride 4:
      0. line-center logit
      1. log line height in output-grid pixels
      2. text-region logit

    The model uses only Conv2D and ReLU before the output heads so the
    inference graph remains straightforward to reproduce in a small
    native or WebAssembly runtime.
    """

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 12, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, 3, padding=8, dilation=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(12, 3, 1)

    def forward(self, x):
        return self.head(self.body(x))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
