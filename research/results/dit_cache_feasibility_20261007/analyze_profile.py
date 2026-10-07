"""Aggregate Experiment 14A profiles and identify robust cache candidates."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics


CACHE_COMPONENTS = ("residual", "attention", "mlp")


def contiguous_runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    ordered = sorted(set(values))
    runs = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def load_profiles(output_dir: Path) -> dict[int, list[dict]]:
    result = {}
    for path in sorted(output_dir.glob("class_*_profile.json")):
        class_id = int(path.stem.split("_")[1])
        result[class_id] = json.loads(path.read_text())
    if not result:
        raise FileNotFoundError(f"No class profiles in {output_dir}")
    return result


def analyze(profiles: dict[int, list[dict]], cosine_threshold: float,
            change_threshold: float) -> tuple[list[dict], dict]:
    grouped = defaultdict(list)
    for class_id, rows in profiles.items():
        for row in rows:
            if "cosine" in row:
                grouped[(row["component"], row["block"], row["step"])].append((class_id, row))
    expected_classes = set(profiles)
    aggregate = []
    for (component, block, step), values in sorted(grouped.items()):
        if {class_id for class_id, _ in values} != expected_classes:
            raise RuntimeError(f"Incomplete classes for {component} block={block} step={step}")
        cosines = [row["cosine"] for _, row in values]
        changes = [row["relative_l2_change"] for _, row in values]
        aggregate.append({
            "component": component, "block": block, "step": step,
            "class_count": len(values), "mean_cosine": statistics.fmean(cosines),
            "min_cosine": min(cosines),
            "mean_relative_l2_change": statistics.fmean(changes),
            "max_relative_l2_change": max(changes),
            "robust_eligible": min(cosines) >= cosine_threshold and max(changes) <= change_threshold,
        })
    block_count = max(row["block"] for row in aggregate) + 1
    step_count = max(row["step"] for row in aggregate) + 1
    candidate_runs = []
    component_summary = {}
    for component in CACHE_COMPONENTS:
        rows = [row for row in aggregate if row["component"] == component]
        eligible = [row for row in rows if row["robust_eligible"]]
        runs = []
        for block in range(block_count):
            steps = [row["step"] for row in eligible if row["block"] == block]
            for start, end in contiguous_runs(steps):
                runs.append({"component": component, "block": block,
                             "start_step": start, "end_step": end,
                             "length": end - start + 1})
        runs.sort(key=lambda row: (-row["length"], row["block"], row["start_step"]))
        candidate_runs.extend(runs)
        component_summary[component] = {
            "eligible_coordinates": len(eligible), "total_coordinates": len(rows),
            "eligible_fraction": len(eligible) / len(rows),
            "longest_consecutive_steps": runs[0]["length"] if runs else 0,
            "mean_cosine": statistics.fmean(row["mean_cosine"] for row in rows),
            "mean_relative_l2_change": statistics.fmean(row["mean_relative_l2_change"] for row in rows),
        }
    summary = {
        "class_ids": sorted(profiles), "class_count": len(profiles),
        "block_count": block_count, "adjacent_steps": step_count - 1,
        "thresholds": {"minimum_cosine_all_classes": cosine_threshold,
                       "maximum_relative_l2_change_all_classes": change_threshold},
        "components": component_summary, "candidate_runs": candidate_runs,
        "has_robust_candidate": any(row["length"] >= 2 for row in candidate_runs),
    }
    return aggregate, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cosine-threshold", type=float, default=0.995)
    parser.add_argument("--change-threshold", type=float, default=0.10)
    args = parser.parse_args()
    aggregate, summary = analyze(load_profiles(args.output_dir), args.cosine_threshold,
                                 args.change_threshold)
    with (args.output_dir / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (args.output_dir / "profile_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
