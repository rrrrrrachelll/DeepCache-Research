"""Diagnose whether pilot checkpoints learned beyond the zero-output baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import TrajectoryDataset
from sacc import SolverAwareCorrector
from train import collate, evaluate, summarize


def state_diagnostics(model: torch.nn.Module) -> dict:
    parameters = torch.cat([parameter.detach().float().cpu().flatten()
                            for parameter in model.parameters()])
    output_weight = model.output.weight.detach().float().cpu()
    output_bias = model.output.bias.detach().float().cpu()
    return {
        "parameter_l2": parameters.norm().item(),
        "output_weight_l2": output_weight.norm().item(),
        "output_weight_abs_max": output_weight.abs().max().item(),
        "output_bias_l2": output_bias.norm().item(),
    }


def prediction_diagnostics(model, loader, statistics):
    values = []
    model.eval()
    from data import denormalize_target, normalize_batch
    with torch.inference_mode():
        for batch in loader:
            normalized = normalize_batch(batch, statistics, "cuda")
            prediction = denormalize_target(model(normalized).float(), statistics)
            values.append(prediction.detach().cpu())
    prediction = torch.cat(values)
    return {"prediction_abs_mean": prediction.abs().mean().item(),
            "prediction_abs_max": prediction.abs().max().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = TrajectoryDataset(args.train_dir)
    validation = TrajectoryDataset(args.validation_dir)
    train_loader = DataLoader(train, batch_size=16, shuffle=False, collate_fn=collate)
    validation_loader = DataLoader(validation, batch_size=16, shuffle=False, collate_fn=collate)
    statistics = torch.load(args.checkpoint_dir / "normalization.pt", map_location="cpu")
    result = {}
    for variant in ("c0", "c1", "c2", "c3"):
        saved = torch.load(args.checkpoint_dir / f"{variant}.pt", map_location="cuda")
        model = SolverAwareCorrector(variant).cuda()
        model.load_state_dict(saved["state_dict"])
        result[variant] = {
            "saved_epoch": saved["epoch"],
            "state": state_diagnostics(model),
            "prediction": prediction_diagnostics(model, validation_loader, statistics),
            "train": summarize(evaluate(model, train_loader, statistics)),
            "validation": summarize(evaluate(model, validation_loader, statistics)),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
