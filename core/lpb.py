import torch
import torch.nn as nn

class LPB(nn.Module):
    """
    Learnable Preprocessing Block (LPB) — prosty blok konwolucyjny uczony razem z detektorem YOLO.
    Cel: poprawa odporności na zakłócenia obrazu.
    """
    def __init__(self, channels=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)
