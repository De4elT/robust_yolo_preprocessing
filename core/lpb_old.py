import torch
import torch.nn as nn

class LPB(nn.Module):
    """
    Learnable Preprocessing Block
    """

    def __init__(self, channels=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        print(f"[LPB] Input shape: {x.shape}")
        x = self.block(x)
        print(f"[LPB] Output shape: {x.shape}")
        return x
