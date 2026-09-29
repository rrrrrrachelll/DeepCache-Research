"""Capacity sanity check: deliberately overfit C3 to one 12-step trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import (TrajectoryDataset, compute_normalization, normalize_batch,
                  normalized_target)
from sacc import SolverAwareCorrector
from train import collate, evaluate, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    torch.manual_seed(29)
    torch.cuda.manual_seed_all(29)

    dataset = TrajectoryDataset(args.data_dir)
    first_case = dataset.rows[0]["case"]
    indices = [index for index, row in enumerate(dataset.rows)
               if row["case"] == first_case]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=len(indices), shuffle=False, collate_fn=collate)
    statistics = compute_normalization(dataset)
    batch = next(iter(loader))
    normalized = normalize_batch(batch, statistics, "cuda")
    target = normalized_target(batch, statistics, "cuda")
    model = SolverAwareCorrector("c3").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=0.0)
    records = []
    for step in range(args.steps + 1):
        if step % 25 == 0 or step == args.steps:
            summary = summarize(evaluate(model, loader, statistics))
            record = {"step": step, "loss": F.smooth_l1_loss(
                model(normalized), target).item(), **summary}
            records.append(record)
            print(json.dumps(record), flush=True)
        if step == args.steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(normalized)
        loss = F.smooth_l1_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    result = {"case": first_case, "samples": len(indices),
              "learning_rate": args.learning_rate, "records": records}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
