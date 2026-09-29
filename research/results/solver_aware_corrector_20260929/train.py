"""Train C0-C3 correction models and report prompt-disjoint validation recovery."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (PromptGroupedBatchSampler, TrajectoryDataset, compute_normalization,
                  denormalize_target, normalize_batch, normalized_target)
from sacc import SolverAwareCorrector, parameter_count


def collate(rows):
    tensor_keys = ("latent", "cached", "x0_cached", "history1", "history2", "target", "full", "scalars")
    result = {key: torch.cat([row[key] for row in rows], dim=0) if key != "scalars"
              else torch.stack([row[key] for row in rows]) for key in tensor_keys}
    for key in ("case", "prompt_id", "category", "seed", "step"):
        result[key] = [row[key] for row in rows]
    return result


def evaluate(model, loader, statistics_data, shuffle_history=False):
    model.eval(); rows = []
    with torch.inference_mode():
        for batch in loader:
            normalized = normalize_batch(batch, statistics_data, "cuda")
            if shuffle_history:
                permutation = torch.arange(normalized["history1"].shape[0] - 1, -1, -1, device="cuda")
                normalized["history1"] = normalized["history1"][permutation]
                normalized["history2"] = normalized["history2"][permutation]
            prediction = denormalize_target(model(normalized).float(), statistics_data).cpu()
            target = batch["target"]
            for index in range(target.shape[0]):
                baseline = target[index].abs().mean().item()
                corrected = (target[index] - prediction[index]).abs().mean().item()
                rows.append({"case": batch["case"][index], "category": batch["category"][index],
                             "step": batch["step"][index], "baseline_l1": baseline,
                             "corrected_l1": corrected,
                             "recovery": 1 - corrected / max(baseline, 1e-12)})
    return rows


def summarize(rows):
    def grouped(key):
        values = defaultdict(list)
        for row in rows: values[str(row[key])].append(row["recovery"])
        return {group: statistics.mean(items) for group, items in sorted(values.items())}
    case_values = defaultdict(list)
    for row in rows: case_values[row["case"]].append(row["recovery"])
    return {"samples": len(rows), "mean_recovery": statistics.mean(r["recovery"] for r in rows),
            "median_recovery": statistics.median(r["recovery"] for r in rows),
            "positive_steps": sum(value > 0 for value in grouped("step").values()),
            "positive_categories": sum(value > 0 for value in grouped("category").values()),
            "positive_cases": sum(statistics.mean(value) > 0 for value in case_values.values()),
            "cases": len(case_values), "by_step": grouped("step"),
            "by_category": grouped("category")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["c0", "c1", "c2", "c3"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(29); np.random.seed(29); torch.manual_seed(29); torch.cuda.manual_seed_all(29)
    train = TrajectoryDataset(args.train_dir); validation = TrajectoryDataset(args.validation_dir)
    statistics_data = compute_normalization(train)
    torch.save(statistics_data, args.output_dir / "normalization.pt")
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    all_results = {}
    for variant in args.variants:
        sampler = PromptGroupedBatchSampler(train, args.batch_size, seed=29)
        loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate)
        model = SolverAwareCorrector(variant).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        best, best_epoch, stale = float("inf"), -1, 0
        history = []
        checkpoint = args.output_dir / f"{variant}.pt"
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch); model.train(); losses = []
            for batch in loader:
                normalized = normalize_batch(batch, statistics_data, "cuda")
                target = normalized_target(batch, statistics_data, "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    prediction = model(normalized)
                    loss = F.smooth_l1_loss(prediction.float(), target.float())
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); losses.append(loss.item())
            validation_rows = evaluate(model, validation_loader, statistics_data)
            validation_l1 = statistics.mean(row["corrected_l1"] for row in validation_rows)
            record = {"epoch": epoch, "train_loss": statistics.mean(losses),
                      "validation_l1": validation_l1,
                      "validation_recovery": statistics.mean(r["recovery"] for r in validation_rows)}
            history.append(record)
            print(json.dumps({"variant": variant, **record}), flush=True)
            if validation_l1 < best - 1e-8:
                best, best_epoch, stale = validation_l1, epoch, 0
                torch.save({"variant": variant, "state_dict": model.state_dict(),
                            "parameters": parameter_count(model), "epoch": epoch}, checkpoint)
            else:
                stale += 1
                if stale >= args.patience: break
        saved = torch.load(checkpoint, map_location="cuda"); model.load_state_dict(saved["state_dict"])
        rows = evaluate(model, validation_loader, statistics_data)
        summary = summarize(rows)
        shuffled = summarize(evaluate(model, validation_loader, statistics_data, shuffle_history=True)) \
            if variant in ("c2", "c3") else None
        all_results[variant] = {"parameters": parameter_count(model), "best_epoch": best_epoch,
                                "validation": summary, "shuffled_history": shuffled,
                                "history": history}
        (args.output_dir / f"{variant}_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
        (args.output_dir / "results.json").write_text(json.dumps(all_results, indent=2) + "\n")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
