"""Integrity and error summary for a collected split."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import torch


def aggregate(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row["guided_error"])
    return {name: {"n": len(values), "mean": statistics.mean(values),
                   "min": min(values), "max": max(values)}
            for name, values in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.input_dir / "data").glob("*.pt"))
    if not paths:
        raise FileNotFoundError("No trajectory artifacts")
    reuse_rows, refresh_rows, files = [], [], []
    for path in paths:
        artifact = torch.load(path, map_location="cpu")
        if len(artifact["trace"]) != 20 or len(artifact["samples"]) != 12:
            raise RuntimeError(f"Incomplete artifact {path}")
        for row in artifact["trace"]:
            target = refresh_rows if row["refresh"] else reuse_rows
            target.append({**row, "case": artifact["case"], "category": artifact["category"]})
        files.append({"case": artifact["case"], "mib": path.stat().st_size / 2**20,
                      "seconds": artifact["seconds"], "peak_mib": artifact["peak_allocated_mib"]})
    if max(row["guided_error"] for row in refresh_rows) != 0:
        raise RuntimeError("A refresh error is nonzero")
    summary = {
        "trajectories": len(paths), "reuse_rows": len(reuse_rows),
        "refresh_rows": len(refresh_rows),
        "mean_seconds": statistics.mean(row["seconds"] for row in files),
        "max_peak_mib": max(row["peak_mib"] for row in files),
        "total_data_mib": sum(row["mib"] for row in files),
        "reuse_error": {
            "mean": statistics.mean(row["guided_error"] for row in reuse_rows),
            "median": statistics.median(row["guided_error"] for row in reuse_rows),
            "min": min(row["guided_error"] for row in reuse_rows),
            "max": max(row["guided_error"] for row in reuse_rows),
        },
        "by_step": aggregate(reuse_rows, "step"),
        "by_age": aggregate(reuse_rows, "age"),
        "by_category": aggregate(reuse_rows, "category"),
        "cases_with_positive_error": sum(any(row["guided_error"] > 0 for row in reuse_rows
                                               if row["case"] == file["case"]) for file in files),
    }
    (args.input_dir / "collection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
