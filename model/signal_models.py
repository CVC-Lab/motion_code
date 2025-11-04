import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

# ------------------------------
# Simple 1D encoder for a window
# ------------------------------
class SignalEncoder1D(nn.Module):
    """
    Map a lag-aligned window (1, wlen, C) -> (1, D).
    Lightweight Conv1d + global avg-pool + linear.
    """
    def __init__(self, C_in: int, D: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(C_in, 32, kernel_size=7, padding=3, dtype=torch.float64),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, dtype=torch.float64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, dtype=torch.float64),
            nn.GELU(),
        )
        self.head = nn.Linear(64, D, dtype=torch.float64)

    def forward(self, win_1_w_c: torch.Tensor) -> torch.Tensor:
        # win_1_w_c: (1, w, C)
        x = win_1_w_c.movedim(-1, 1)      # -> (1, C, w)
        h = self.conv(x)                   # (1, 64, w)
        h = h.mean(dim=-1)                 # global avg pool -> (1, 64)
        z = self.head(h)                   # (1, D)
        return z

class SignalDecoder1D(nn.Module):
    """
    Simple MLP decoder: z (1,D) -> y_hat (1, window_size, C_out).
    For windows shorter than window_size, we truncate to wlen.
    """
    def __init__(self, D: int, C_out: int, window_size: int):
        super().__init__()
        self.D = int(D)
        self.C_out = int(C_out)
        self.window_size = int(window_size)
        self.net = nn.Sequential(
            nn.Linear(D, 4*D, dtype=torch.float64), nn.GELU(),
            nn.Linear(4*D, 8*D, dtype=torch.float64), nn.GELU(),
            nn.Linear(8*D, self.window_size * self.C_out, dtype=torch.float64),
        )

    def forward(self, z: torch.Tensor, wlen: Optional[int] = None) -> torch.Tensor:
        """
        z:   (1, D)
        out: (1, wlen, C_out) with wlen==window_size if not given
        """
        y = self.net(z)                              # (1, W*C)
        y = y.view(1, self.window_size, self.C_out)  # (1, W, C)
        if wlen is not None and wlen < self.window_size:
            y = y[:, :wlen, :]
        return y
