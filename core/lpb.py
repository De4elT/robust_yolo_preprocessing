import torch
import torch.nn as nn

class LPB(nn.Module):
    """
    Learnable Preprocessing Block (LPB)

    This version is designed for thesis-grade experiments:
    - residual mapping (identity-preserving): y = x + g(x) * delta(x)
    - optional gating g(x) in [0, 1] (per-channel), making the block conservative on clean inputs
    - no debug prints (prints inside forward kill training speed and distort FPS measurements)

    Default behaviour keeps backward compatibility:
      LPB(channels=3)

    Notes:
    - If you want the block to start very close to identity, we initialize delta to zeros.
      That allows you to train baseline YOLO weights first (no LPB effect) and then fine-tune LPB.
    """

    def __init__(self, channels=3, hidden=16, gated=True):
        super().__init__()
        self.gated = gated

        self.feat = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.delta = nn.Conv2d(hidden, channels, kernel_size=3, padding=1)

        if gated:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.gate = None

        # init: start close to identity
        nn.init.zeros_(self.delta.weight)
        if self.delta.bias is not None:
            nn.init.zeros_(self.delta.bias)

    def forward(self, x):
        d = self.delta(self.feat(x))
        if self.gate is None:
            return x + d
        g = self.gate(x)
        return x + g * d
