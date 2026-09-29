"""Trajectory dataset, normalization, and prompt-grouped batching."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, Sampler


TENSOR_KEYS = ("latent", "cached", "x0_cached", "history1", "history2", "target", "full")
INPUT_KEYS = ("latent", "cached", "x0_cached", "history1", "history2")


class TrajectoryDataset(Dataset):
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.rows = []
        paths = sorted((self.directory / "data").glob("*.pt"))
        if not paths:
            raise FileNotFoundError(f"No trajectories in {self.directory / 'data'}")
        for path in paths:
            artifact = torch.load(path, map_location="cpu")
            for sample in artifact["samples"]:
                row = {key: value.float() for key, value in sample.items() if torch.is_tensor(value)}
                row.update(case=artifact["case"], prompt_id=artifact["prompt_id"],
                           category=artifact["category"], seed=artifact["seed"],
                           step=sample["step"])
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class PromptGroupedBatchSampler(Sampler[list[int]]):
    """Yield at most one diffusion step from each trajectory in a batch."""
    def __init__(self, dataset: TrajectoryDataset, batch_size: int, seed: int = 0):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        grouped = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            grouped[row["case"]].append(index)
        self.groups = dict(grouped)

    def __len__(self):
        return math.ceil(sum(len(v) for v in self.groups.values()) / self.batch_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        groups = {key: rng.sample(value, len(value)) for key, value in self.groups.items()}
        pending = list(groups)
        while pending:
            rng.shuffle(pending)
            batch = []
            next_pending = []
            for case in pending:
                batch.append(groups[case].pop())
                if groups[case]:
                    next_pending.append(case)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
            pending = next_pending


def compute_normalization(dataset: TrajectoryDataset) -> dict:
    result = {"tensors": {}}
    for key in TENSOR_KEYS:
        values = torch.cat([row[key] for row in dataset.rows], dim=0).double()
        mean = values.mean(dim=(0, 2, 3))
        std = values.std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-5)
        result["tensors"][key] = {"mean": mean.float(), "std": std.float()}
    scalars = torch.stack([row["scalars"] for row in dataset.rows]).double()
    result["scalars"] = {
        "mean": scalars.mean(0).float(),
        "std": scalars.std(0, unbiased=False).clamp_min(1e-5).float(),
    }
    return result


def normalize_batch(batch: dict, statistics: dict, device: str) -> dict:
    result = {}
    for key in INPUT_KEYS:
        value = batch[key].to(device)
        params = statistics["tensors"][key]
        mean = params["mean"].to(device)[None, :, None, None]
        std = params["std"].to(device)[None, :, None, None]
        result[key] = (value - mean) / std
    scalar = batch["scalars"].to(device)
    result["scalars"] = ((scalar - statistics["scalars"]["mean"].to(device)) /
                         statistics["scalars"]["std"].to(device))
    return result


def normalized_target(batch: dict, statistics: dict, device: str) -> torch.Tensor:
    value = batch["target"].to(device)
    params = statistics["tensors"]["target"]
    return ((value - params["mean"].to(device)[None, :, None, None]) /
            params["std"].to(device)[None, :, None, None])


def denormalize_target(value: torch.Tensor, statistics: dict) -> torch.Tensor:
    params = statistics["tensors"]["target"]
    return (value * params["std"].to(value.device)[None, :, None, None] +
            params["mean"].to(value.device)[None, :, None, None])
