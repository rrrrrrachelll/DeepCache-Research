"""Parameter-matched local and multiscale C3 correctors for experiment 12."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sacc import FiLMResidualBlock, SolverAwareCorrector


class LocalWideCorrector(SolverAwareCorrector):
    """Capacity control: larger C3 without changing spatial scale."""
    def __init__(self):
        super().__init__("c3", width=64, blocks=6)


class MultiscaleCorrector(nn.Module):
    """C3 with one 64->32->64 spatial hierarchy and a skip connection."""
    def __init__(self, high_width=48, low_width=80, scalar_features=7):
        super().__init__()
        scalar_width = 64
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_features, scalar_width), nn.SiLU(),
            nn.Linear(scalar_width, scalar_width))
        self.stem = nn.Conv2d(20, high_width, 3, padding=1)
        self.encoder = nn.ModuleList([
            FiLMResidualBlock(high_width, scalar_width) for _ in range(2)])
        self.down = nn.Conv2d(high_width, low_width, 3, stride=2, padding=1)
        self.low = nn.ModuleList([
            FiLMResidualBlock(low_width, scalar_width) for _ in range(2)])
        self.up = nn.Conv2d(low_width, high_width, 3, padding=1)
        self.fuse = nn.Conv2d(high_width * 2, high_width, 3, padding=1)
        self.decoder = FiLMResidualBlock(high_width, scalar_width)
        self.output = nn.Conv2d(high_width, 4, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def features(batch):
        return torch.cat([
            batch["latent"], batch["cached"], batch["x0_cached"],
            batch["history1"], batch["history2"]], dim=1)

    def forward(self, batch):
        condition = self.scalar_mlp(batch["scalars"])
        high = self.stem(self.features(batch))
        for block in self.encoder:
            high = block(high, condition)
        low = self.down(high)
        for block in self.low:
            low = block(low, condition)
        up = F.interpolate(low, size=high.shape[-2:], mode="bilinear",
                           align_corners=False)
        up = self.up(up)
        hidden = self.fuse(torch.cat([high, up], dim=1))
        hidden = self.decoder(hidden, condition)
        return self.output(hidden)


def build_corrector(name: str) -> nn.Module:
    if name == "local_wide":
        return LocalWideCorrector()
    if name == "multiscale":
        return MultiscaleCorrector()
    raise ValueError(name)
