"""Aggregate Experiment 14B runtime, quality, and cache-age diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def cache_age_stats(coordinates: set[tuple[int, int]], steps: int, blocks: int) -> dict[str, float]:
    ages = []
    longest_run = 0
    for block in range(blocks):
        age = 0
        for step in range(steps):
            if (step, block) in coordinates:
                age += 1
                ages.append(age)
                longest_run = max(longest_run, age)
            else:
                age = 0
    return {
        "mean_reuse_age": statistics.fmean(ages) if ages else 0.0,
        "max_reuse_age": max(ages, default=0),
        "longest_consecutive_run": longest_run,
    }


def analyze(results: list[dict], policies: dict, steps: int, blocks: int) -> dict:
    baselines = {row["class_id"]: row for row in results if row["strategy"] == "baseline"}
    strategy_names = sorted({row["strategy"] for row in results if row["strategy"] != "baseline"})
    summaries = {}
    for strategy in strategy_names:
        rows = [row for row in results if row["strategy"] == strategy]
        speedups = [baselines[row["class_id"]]["seconds"] / row["seconds"] for row in rows]
        coordinates = {tuple(item) for item in policies[strategy]["coordinates"]}
        summaries[strategy] = {
            "cases": len(rows),
            "mean_seconds": statistics.fmean(row["seconds"] for row in rows),
            "mean_speedup": statistics.fmean(speedups),
            "min_speedup": min(speedups),
            "mean_ssim": statistics.fmean(row["ssim"] for row in rows),
            "worst_ssim": min(row["ssim"] for row in rows),
            "mean_lpips": statistics.fmean(row["lpips"] for row in rows),
            "worst_lpips": max(row["lpips"] for row in rows),
            "mean_dists": statistics.fmean(row["dists"] for row in rows),
            "worst_dists": max(row["dists"] for row in rows),
            "mean_pixel_mae": statistics.fmean(row["pixel_mae"] for row in rows),
            "skipped_block_fraction": rows[0]["skipped_block_calls"] / (steps * blocks),
            **cache_age_stats(coordinates, steps, blocks),
        }

    paired = {}
    for tier in ("conservative_25", "target_35", "aggressive_50"):
        selected = summaries[tier]
        shuffled = summaries[f"{tier}_shuffled"]
        paired[tier] = {
            "selected_minus_shuffled_ssim": selected["mean_ssim"] - shuffled["mean_ssim"],
            "selected_minus_shuffled_lpips": selected["mean_lpips"] - shuffled["mean_lpips"],
            "selected_minus_shuffled_dists": selected["mean_dists"] - shuffled["mean_dists"],
            "selected_minus_shuffled_speedup": selected["mean_speedup"] - shuffled["mean_speedup"],
        }
    return {
        "baseline": {
            "cases": len(baselines),
            "mean_seconds": statistics.fmean(row["seconds"] for row in baselines.values()),
            "mean_peak_allocated_mib": statistics.fmean(row["peak_allocated_mib"] for row in baselines.values()),
        },
        "strategies": summaries,
        "paired_selected_vs_shuffled": paired,
        "decision": {
            "any_selected_strategy_quality_preserving": any(
                summaries[name]["mean_ssim"] >= 0.90
                and summaries[name]["mean_lpips"] <= 0.10
                and summaries[name]["mean_dists"] <= 0.10
                for name in ("conservative_25", "target_35", "aggressive_50")
            ),
            "target_35_reaches_1_5x_mean_speedup": summaries["target_35"]["mean_speedup"] >= 1.5,
            "stop_static_whole_block_cache": True,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.output_dir / "results.json").read_text())
    policies = json.loads((args.output_dir / "policies.json").read_text())
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    analysis = analyze(results, policies, manifest["steps"], manifest["block_count"])
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
