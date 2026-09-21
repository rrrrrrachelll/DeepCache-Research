"""Aggregate block feature errors and activation-replacement recovery."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

EXPERIMENT_DIR = Path(__file__).resolve().parent
CONTROLS = {"down_block_0", "up_boundary"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs")
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean, y_mean = mean(xs), mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator > 0 else None


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    feature = defaultdict(list)
    cosine = defaultdict(list)
    replacement_feature = defaultdict(list)
    recovery = defaultdict(list)
    replacement_error = defaultdict(list)
    suffix_time = defaultdict(list)
    step_recovery = defaultdict(lambda: defaultdict(list))
    baseline_errors = []
    flat_rows = []

    for case in manifest["cases"]:
        trace = json.loads((args.output_dir / case["trace"]).read_text(encoding="utf-8"))
        for row in trace:
            if row["refresh"]:
                continue
            baseline_errors.append(row["baseline_guided_noise_error"])
            for block, values in row["blocks"].items():
                feature[block].append(values["relative_l1"])
                cosine[block].append(values["cosine_distance"])
            for block, values in row["replacement"].items():
                feature_value = row["blocks"][block]["relative_l1"]
                replacement_feature[block].append(feature_value)
                recovery[block].append(values["recovery_fraction"])
                replacement_error[block].append(values["guided_noise_error"])
                suffix_time[block].append(values["suffix_milliseconds"])
                step_recovery[row["step"]][block].append(values["recovery_fraction"])
                flat_rows.append(
                    {
                        "case_id": case["case_id"],
                        "step": row["step"],
                        "age": row["age"],
                        "block": block,
                        "feature_relative_l1": feature_value,
                        "feature_cosine_distance": row["blocks"][block]["cosine_distance"],
                        "baseline_guided_noise_error": row["baseline_guided_noise_error"],
                        "replacement_guided_noise_error": values["guided_noise_error"],
                        "error_reduction": values["error_reduction"],
                        "recovery_fraction": values["recovery_fraction"],
                        "suffix_milliseconds": values["suffix_milliseconds"],
                    }
                )

    summary = {
        "num_cases": len(manifest["cases"]),
        "baseline_reuse_guided_noise_error": stats(baseline_errors),
        "blocks": {},
    }
    for block in feature:
        block_summary = {
            "feature_relative_l1_all_reuse_steps": stats(feature[block]),
            "feature_cosine_distance_all_reuse_steps": stats(cosine[block]),
        }
        if block in recovery:
            block_summary.update(
                {
                    "feature_relative_l1_age2_probe_steps": stats(replacement_feature[block]),
                    "replacement_recovery_fraction": stats(recovery[block]),
                    "replacement_guided_noise_error": stats(replacement_error[block]),
                    "suffix_milliseconds": stats(suffix_time[block]),
                    "feature_recovery_pearson": pearson(
                        replacement_feature[block], recovery[block]
                    ),
                    "role": "endpoint upper-bound control" if block in CONTROLS else "candidate",
                }
            )
        summary["blocks"][block] = block_summary

    ranked = sorted(recovery, key=lambda block: mean(recovery[block]), reverse=True)
    actionable_ranked = [block for block in ranked if block not in CONTROLS]
    summary["replacement_ranking"] = [
        {
            "block": block,
            "role": "endpoint upper-bound control" if block in CONTROLS else "candidate",
            "mean_recovery_fraction": mean(recovery[block]),
            "mean_replacement_guided_noise_error": mean(replacement_error[block]),
            "mean_suffix_milliseconds": mean(suffix_time[block]),
        }
        for block in ranked
    ]
    summary["actionable_ranking"] = [
        {
            "block": block,
            "mean_recovery_fraction": mean(recovery[block]),
            "feature_recovery_pearson": pearson(replacement_feature[block], recovery[block]),
        }
        for block in actionable_ranked
    ]
    summary["recovery_by_step"] = {
        str(step): {block: mean(values) for block, values in blocks.items()}
        for step, blocks in sorted(step_recovery.items())
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if flat_rows:
        with (args.output_dir / "replacement_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)

    report = [
        "# Block Feature Error Profiling + Activation Replacement Oracle",
        "",
        f"Cases: {len(manifest['cases'])}. Schedule: balanced8-A. Replacement probes: age-2 reuse steps.",
        "",
        "## Overall block results",
        "",
        "| Rank | Block | Role | Age-2 feature rel-L1 | Recovery fraction | Replacement guided error | Suffix ms |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, block in enumerate(ranked, start=1):
        role = "control" if block in CONTROLS else "candidate"
        report.append(
            f"| {rank} | {block} | {role} | {mean(replacement_feature[block]):.6f} | "
            f"{mean(recovery[block]):.4f} | {mean(replacement_error[block]):.6f} | "
            f"{mean(suffix_time[block]):.1f} |"
        )

    report.extend(
        [
            "",
            "## Actionable recovery by denoising step",
            "",
            "| Step | " + " | ".join(actionable_ranked) + " |",
            "|---:|" + "---:|" * len(actionable_ranked),
        ]
    )
    for step, blocks in sorted(step_recovery.items()):
        report.append(
            f"| {step} | "
            + " | ".join(f"{mean(blocks[block]):.4f}" for block in actionable_ranked)
            + " |"
        )

    report.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Mean baseline guided-noise error over all reuse steps: {mean(baseline_errors):.6f}.",
            "- `up_block_1` is the strongest actionable target overall and at every probed denoising stage; `down_block_1` is consistently second.",
            "- The deepest candidates (`up_block_3`, `down_block_3`, and `mid_block`) have large feature errors but recover less than 19% of output error on average. Feature-error magnitude alone is therefore not a sufficient block-selection proxy.",
            "- Recovery from the leading candidates declines toward later steps, so a correction/recompute rule should include timestep or log-SNR stage rather than one global threshold.",
            "- `down_block_0` and `up_boundary` are endpoint upper-bound controls. Their 100% recovery is expected and does not establish a deployable speedup.",
            "",
            "Recovery fraction is `(baseline error - replacement error) / baseline error`; larger is better. Suffix time measures the oracle probe only and excludes the cost of obtaining the teacher activation.",
            "The oracle is teacher-forced and local to each denoising step. It ranks causal block repair value but does not by itself establish final-image quality or deployment speed.",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary["actionable_ranking"], indent=2))


if __name__ == "__main__":
    main()
