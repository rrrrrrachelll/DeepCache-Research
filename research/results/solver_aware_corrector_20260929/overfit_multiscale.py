"""Experiment 12A: one-trajectory capacity check for both architectures."""
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
from multiscale import build_corrector
from sacc import parameter_count
from train import collate, evaluate, summarize


def run(name, batch, loader, normalization, steps, learning_rate):
    torch.manual_seed(29); torch.cuda.manual_seed_all(29)
    model = build_corrector(name).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.0)
    normalized = normalize_batch(batch, normalization, "cuda")
    target = normalized_target(batch, normalization, "cuda")
    records = []
    for step in range(steps + 1):
        if step % 100 == 0 or step == steps:
            result = summarize(evaluate(model, loader, normalization))
            record = {"step": step,
                      "loss": F.smooth_l1_loss(model(normalized), target).item(),
                      **result}
            records.append(record)
            print(json.dumps({"model": name, **record}), flush=True)
        if step == steps:
            break
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(model(normalized), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final = records[-1]
    return {"parameters": parameter_count(model), "records": records,
            "passed": final["mean_recovery"] >= 0.50 and
                      final["positive_steps"] == 12}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = TrajectoryDataset(args.data_dir)
    first_case = dataset.rows[0]["case"]
    indices = [index for index, row in enumerate(dataset.rows)
               if row["case"] == first_case]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=len(indices), shuffle=False,
                        collate_fn=collate)
    batch = next(iter(loader))
    normalization = compute_normalization(dataset)
    result = {"case": first_case, "samples": len(indices), "models": {}}
    for name in ("local_wide", "multiscale"):
        result["models"][name] = run(
            name, batch, loader, normalization, args.steps,
            args.learning_rate)
    result["passed"] = all(row["passed"] for row in result["models"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"],
                      "parameters": {key: value["parameters"] for key, value
                                     in result["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
