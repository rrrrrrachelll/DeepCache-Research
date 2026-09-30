"""Experiment 12B: parameter-matched local-wide vs multiscale training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import (PromptGroupedBatchSampler, TrajectoryDataset,
                  compute_normalization, normalize_batch, normalized_target)
from multiscale import build_corrector
from sacc import parameter_count
from train import collate, evaluate, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+",
                        default=["local_wide", "multiscale"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    random.seed(29); np.random.seed(29); torch.manual_seed(29)
    torch.cuda.manual_seed_all(29)
    train = TrajectoryDataset(args.train_dir)
    validation = TrajectoryDataset(args.validation_dir)
    normalization = compute_normalization(train)
    torch.save(normalization, args.output_dir / "normalization.pt")
    train_evaluation_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    results = {}
    for name in args.models:
        sampler = PromptGroupedBatchSampler(train, args.batch_size, seed=29)
        loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate)
        torch.manual_seed(29); torch.cuda.manual_seed_all(29)
        model = build_corrector(name).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        best, best_epoch, stale = float("inf"), -1, 0
        history = []
        checkpoint = args.output_dir / f"{name}.pt"
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch); model.train(); losses = []
            for batch in loader:
                normalized = normalize_batch(batch, normalization, "cuda")
                target = normalized_target(batch, normalization, "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    prediction = model(normalized)
                    loss = F.smooth_l1_loss(prediction.float(), target.float())
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); losses.append(loss.item())
            rows = evaluate(model, validation_loader, normalization)
            validation_l1 = statistics.mean(row["corrected_l1"] for row in rows)
            record = {"epoch": epoch, "train_loss": statistics.mean(losses),
                      "validation_l1": validation_l1,
                      "validation_recovery": statistics.mean(
                          row["recovery"] for row in rows)}
            history.append(record)
            print(json.dumps({"model": name, **record}), flush=True)
            if validation_l1 < best - 1e-8:
                best, best_epoch, stale = validation_l1, epoch, 0
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "parameters": parameter_count(model)}, checkpoint)
            else:
                stale += 1
                if stale >= args.patience:
                    break
        saved = torch.load(checkpoint, map_location="cuda")
        model.load_state_dict(saved["state_dict"])
        validation_rows = evaluate(model, validation_loader, normalization)
        train_rows = evaluate(model, train_evaluation_loader, normalization)
        shuffled = summarize(evaluate(
            model, validation_loader, normalization, shuffle_history=True))
        results[name] = {
            "parameters": parameter_count(model), "best_epoch": best_epoch,
            "train": summarize(train_rows),
            "validation": summarize(validation_rows),
            "shuffled_history": shuffled, "history": history,
        }
        (args.output_dir / "results.json").write_text(
            json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
