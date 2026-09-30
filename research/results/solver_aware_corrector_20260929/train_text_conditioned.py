"""Train C3 with frozen CLIP prompt conditioning and run causal controls."""
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
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import (PromptGroupedBatchSampler, TrajectoryDataset,
                  compute_normalization, denormalize_target, normalize_batch,
                  normalized_target)
from prompt_set import prompt_hash
from sacc import SolverAwareCorrector, parameter_count
from train import collate as base_collate, summarize


class PromptConditionedCorrector(SolverAwareCorrector):
    def __init__(self, text_features=768):
        super().__init__("c3")
        self.text_mlp = nn.Sequential(
            nn.LayerNorm(text_features), nn.Linear(text_features, 64),
            nn.SiLU(), nn.Linear(64, 64))

    def forward(self, batch):
        scalar = self.scalar_mlp(self.conditioned_scalars(batch["scalars"]))
        condition = scalar + self.text_mlp(batch["text_embedding"])
        hidden = self.input(self.features(batch))
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden)


def attach_embeddings(dataset, artifact):
    if artifact["prompt_hash"] != prompt_hash():
        raise RuntimeError("Prompt embedding artifact does not match frozen prompts")
    embeddings = artifact["embeddings"]
    prompt_ids = sorted({row["prompt_id"] for row in dataset.rows})
    shift = max(1, len(prompt_ids) // 2)
    mismatched = {prompt_id: prompt_ids[(index + shift) % len(prompt_ids)]
                  for index, prompt_id in enumerate(prompt_ids)}
    if any(source == target for source, target in mismatched.items()):
        raise RuntimeError("Text-control permutation contains a fixed point")
    for row in dataset.rows:
        row["text_embedding"] = embeddings[row["prompt_id"]].float()
        row["text_embedding_shuffled"] = embeddings[mismatched[row["prompt_id"]]].float()


def collate(rows):
    result = base_collate(rows)
    result["text_embedding"] = torch.stack([row["text_embedding"] for row in rows])
    result["text_embedding_shuffled"] = torch.stack(
        [row["text_embedding_shuffled"] for row in rows])
    return result


def evaluate(model, loader, normalization, shuffle_history=False,
             shuffle_text=False):
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            normalized = normalize_batch(batch, normalization, "cuda")
            normalized["text_embedding"] = (
                batch["text_embedding_shuffled" if shuffle_text
                      else "text_embedding"].to("cuda"))
            if shuffle_history:
                permutation = torch.arange(
                    normalized["history1"].shape[0] - 1, -1, -1,
                    device="cuda")
                normalized["history1"] = normalized["history1"][permutation]
                normalized["history2"] = normalized["history2"][permutation]
            prediction = denormalize_target(
                model(normalized).float(), normalization).cpu()
            target = batch["target"]
            for index in range(target.shape[0]):
                baseline = target[index].abs().mean().item()
                corrected = (target[index] - prediction[index]).abs().mean().item()
                rows.append({
                    "case": batch["case"][index],
                    "category": batch["category"][index],
                    "step": batch["step"][index],
                    "baseline_l1": baseline,
                    "corrected_l1": corrected,
                    "recovery": 1 - corrected / max(baseline, 1e-12),
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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

    artifact = torch.load(args.embeddings, map_location="cpu")
    train = TrajectoryDataset(args.train_dir)
    validation = TrajectoryDataset(args.validation_dir)
    attach_embeddings(train, artifact)
    attach_embeddings(validation, artifact)
    normalization = compute_normalization(train)
    torch.save(normalization, args.output_dir / "normalization.pt")
    sampler = PromptGroupedBatchSampler(train, args.batch_size, seed=29)
    train_loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate)
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = PromptConditionedCorrector(artifact["dimension"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    best, best_epoch, stale = float("inf"), -1, 0
    history = []
    checkpoint = args.output_dir / "c3_text.pt"
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        losses = []
        for batch in train_loader:
            normalized = normalize_batch(batch, normalization, "cuda")
            normalized["text_embedding"] = batch["text_embedding"].to("cuda")
            target = normalized_target(batch, normalization, "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                prediction = model(normalized)
                loss = F.smooth_l1_loss(prediction.float(), target.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        validation_rows = evaluate(model, validation_loader, normalization)
        validation_l1 = statistics.mean(
            row["corrected_l1"] for row in validation_rows)
        record = {
            "epoch": epoch,
            "train_loss": statistics.mean(losses),
            "validation_l1": validation_l1,
            "validation_recovery": statistics.mean(
                row["recovery"] for row in validation_rows),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
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
    conditions = {
        "true": {},
        "shuffled_history": {"shuffle_history": True},
        "shuffled_text": {"shuffle_text": True},
        "shuffled_both": {"shuffle_history": True, "shuffle_text": True},
    }
    result = {
        "model": "c3_text",
        "parameters": parameter_count(model),
        "best_epoch": best_epoch,
        "embedding": {key: artifact[key] for key in
                      ("model", "encoder", "dimension", "prompt_hash")},
        "history": history,
        "evaluation": {},
    }
    for name, options in conditions.items():
        rows = evaluate(model, validation_loader, normalization, **options)
        result["evaluation"][name] = summarize(rows)
        (args.output_dir / f"{name}_rows.json").write_text(
            json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
