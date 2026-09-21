"""Aggregate the up_block_1 mechanism ablation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

KINDS = ("activation_only", "suffix_only", "activation_plus_suffix")
EXPERIMENT_DIR = Path(__file__).resolve().parent


def stats(values):
    return {"n": len(values), "mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs")
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    recovery = defaultdict(list)
    errors = defaultdict(list)
    timing = defaultdict(list)
    by_step = defaultdict(lambda: defaultdict(list))
    rows = []
    baselines = []
    for case in manifest["cases"]:
        trace = json.loads((args.output_dir / case["trace"]).read_text())
        for row in trace:
            if row["refresh"]:
                continue
            baselines.append(row["baseline_guided_noise_error"])
            for kind, values in row["probes"].items():
                recovery[kind].append(values["recovery_fraction"])
                errors[kind].append(values["guided_noise_error"])
                timing[kind].append(values["milliseconds"])
                by_step[row["step"]][kind].append(values["recovery_fraction"])
                rows.append({"case_id": case["case_id"], "step": row["step"], "age": row["age"],
                             "kind": kind, "baseline_guided_noise_error": row["baseline_guided_noise_error"],
                             **values})
    combined_suffix_overhead_seconds = mean(timing["activation_plus_suffix"]) * 5 / 1000.0
    balanced8_seconds = 29.84
    uniform9_seconds = 31.73
    estimated_combined_seconds = balanced8_seconds + combined_suffix_overhead_seconds
    summary = {
        "num_cases": len(manifest["cases"]),
        "baseline_reuse_guided_noise_error": stats(baselines),
        "probes": {kind: {"recovery_fraction": stats(recovery[kind]),
                           "guided_noise_error": stats(errors[kind]),
                           "milliseconds": stats(timing[kind])} for kind in KINDS},
        "recovery_by_step": {str(step): {kind: mean(values) for kind, values in kinds.items()}
                             for step, kinds in sorted(by_step.items())},
        "runtime_implication": {
            "balanced8_a_seconds_from_existing_dev": balanced8_seconds,
            "uniform9_seconds_from_existing_dev": uniform9_seconds,
            "five_probe_suffix_overhead_seconds": combined_suffix_overhead_seconds,
            "estimated_balanced8_plus_full_suffix_seconds": estimated_combined_seconds,
            "note": "Lower-bound propagation estimate; excludes residual-predictor cost.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "probe_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    report = ["# up_block_1 Mechanism Ablation", "",
              f"Cases: {len(manifest['cases'])}. Probes: {len(rows)}. All probes use the same teacher-forced DPM20 trajectory.", "",
              "| Probe | Recovery fraction | Guided-noise error | Probe time (ms) |",
              "|---|---:|---:|---:|"]
    for kind in KINDS:
        report.append(f"| {kind} | {mean(recovery[kind]):.4f} | {mean(errors[kind]):.6f} | {mean(timing[kind]):.1f} |")
    report += ["", "## Recovery by denoising step", "",
               "| Step | activation only | suffix only | activation + suffix |",
               "|---:|---:|---:|---:|"]
    for step, values in sorted(by_step.items()):
        report.append(f"| {step} | {mean(values['activation_only']):.4f} | {mean(values['suffix_only']):.4f} | {mean(values['activation_plus_suffix']):.4f} |")
    ao, so, both = (mean(recovery[k]) for k in KINDS)
    if so > ao and so >= 0.8 * both:
        conclusion = "Suffix recomputation explains most of the recoverable error; prioritize a fixed selective-recompute implementation before training a residual predictor."
    elif ao > so and ao >= 0.8 * both:
        conclusion = "Activation replacement explains most of the recoverable error; proceed to a lightweight residual predictor at up_block_1."
    else:
        conclusion = "Activation correction and suffix recomputation are complementary. Do not train the residual predictor yet; first scan shorter propagation cut points after up_block_1."
    report += ["", "## Runtime implication", "",
               f"Five full-suffix probes add approximately **{combined_suffix_overhead_seconds:.2f} s** of GPU work per image. Added to the existing balanced8-A mean ({balanced8_seconds:.2f} s), this gives a lower-bound estimate of **{estimated_combined_seconds:.2f} s**, already slower than uniform9 ({uniform9_seconds:.2f} s) before paying for a residual predictor.",
               "", "## Decision", "", conclusion, "",
               "The next experiment should keep teacher activation replacement as the upper bound and vary how many downstream modules are recomputed, measuring recovery per millisecond.",
               "", "Probe time is an oracle-pass diagnostic, not end-to-end deployment latency. Teacher activations are unavailable at inference."]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary["probes"], indent=2))


if __name__ == "__main__":
    main()
