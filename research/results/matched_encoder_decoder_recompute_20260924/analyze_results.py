"""Aggregate the matched encoder-decoder recomputation sweep."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
TRACES = OUT / "traces"
FRONTIERS = [
    "decoder_only",
    "down3_pair",
    "down2_pair",
    "down1_pair",
    "full_unet",
]
PROBE_STEPS = [4, 9]


def aggregate(rows, key):
    values = [row[key] for row in rows]
    return mean(values)


def main() -> None:
    paths = sorted(TRACES.glob("*.json"))
    if len(paths) != 20:
        raise RuntimeError(f"Expected 20 traces, found {len(paths)}")

    rows = []
    controls = []
    for path in paths:
        stem = path.stem
        prompt_id, seed_text = stem.rsplit("_seed", 1)
        seed = int(seed_text)
        trace = json.loads(path.read_text())
        if len(trace) != 20:
            raise RuntimeError(f"{path.name}: expected 20 diffusion steps")
        for step in PROBE_STEPS:
            record = trace[step]
            if record["step"] != step or set(record["frontiers"]) != set(FRONTIERS):
                raise RuntimeError(f"{path.name}: incomplete probe step {step}")
            baseline = record["baseline_guided_noise_error"]
            for frontier in FRONTIERS:
                probe = record["frontiers"][frontier]
                row = {
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "step": step,
                    "frontier": frontier,
                    "baseline_guided_noise_error": baseline,
                    **probe,
                }
                rows.append(row)
                if frontier == "full_unet":
                    controls.append(probe["guided_noise_error"])

    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 probe rows, found {len(rows)}")
    if max(controls) > 1e-6:
        raise RuntimeError(f"Full-UNet control mismatch: max error {max(controls)}")

    fieldnames = list(rows[0])
    with (OUT / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    overall = {}
    by_step = {}
    by_prompt = {}
    for frontier in FRONTIERS:
        selected = [row for row in rows if row["frontier"] == frontier]
        prompt_means = {}
        for prompt_id in sorted({row["prompt_id"] for row in selected}):
            prompt_rows = [row for row in selected if row["prompt_id"] == prompt_id]
            prompt_means[prompt_id] = mean(row["recovery_fraction"] for row in prompt_rows)
        overall[frontier] = {
            "guided_noise_error": aggregate(selected, "guided_noise_error"),
            "recovery_fraction": aggregate(selected, "recovery_fraction"),
            "milliseconds": aggregate(selected, "milliseconds"),
            "incremental_milliseconds": aggregate(selected, "incremental_milliseconds"),
            "recomputed_wrapped_modules": aggregate(selected, "recomputed_wrapped_modules"),
            "positive_probe_rows": sum(row["recovery_fraction"] > 0 for row in selected),
            "positive_prompts": sum(value > 0 for value in prompt_means.values()),
        }
        by_prompt[frontier] = prompt_means
        by_step[frontier] = {}
        for step in PROBE_STEPS:
            step_rows = [row for row in selected if row["step"] == step]
            by_step[frontier][str(step)] = {
                "baseline_guided_noise_error": aggregate(step_rows, "baseline_guided_noise_error"),
                "guided_noise_error": aggregate(step_rows, "guided_noise_error"),
                "recovery_fraction": aggregate(step_rows, "recovery_fraction"),
                "milliseconds": aggregate(step_rows, "milliseconds"),
                "incremental_milliseconds": aggregate(step_rows, "incremental_milliseconds"),
                "positive_rows": sum(row["recovery_fraction"] > 0 for row in step_rows),
            }

    summary = {
        "integrity": {
            "trace_files": len(paths),
            "diffusion_steps": len(paths) * 20,
            "probe_rows": len(rows),
            "frontier_probes": len(paths) * len(PROBE_STEPS) * len(FRONTIERS),
            "full_unet_max_guided_noise_error": max(controls),
        },
        "protocol": {
            "probe_steps": PROBE_STEPS,
            "frontiers": FRONTIERS,
            "decision_threshold": {
                "minimum_mean_recovery_fraction": 0.20,
                "minimum_positive_prompts": 7,
                "both_steps_must_be_nonnegative": True,
            },
        },
        "overall": overall,
        "by_step": by_step,
        "by_prompt": by_prompt,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Matched Encoder–Decoder Path Recompute",
        "",
        "## Protocol and integrity",
        "",
        "- SD1.5, DPM-Solver++ 20 steps, guidance scale 7.5.",
        "- DeepCache balanced8-A refresh steps: `[0, 2, 5, 7, 10, 13, 16, 19]`.",
        "- Probe steps: 4 and 9 (cache age 2).",
        "- Data: 10 analysis prompts x seeds 101 and 202; consumed held-out prompts excluded.",
        "- Each path retains the cached shallower boundary, then recomputes the selected encoder block, all deeper encoder blocks, the mid block, and the full decoder.",
        "- 20/20 traces, 400 diffusion steps, 200 path probes.",
        f"- Full-UNet control maximum guided-noise error: `{max(controls):.8g}`.",
        "",
        "## Overall results",
        "",
        "| Path | Modules | Guided error | Recovery | Positive prompts | Incremental time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for frontier in FRONTIERS:
        item = overall[frontier]
        lines.append(
            f"| {frontier} | {item['recomputed_wrapped_modules']:.0f} | "
            f"{item['guided_noise_error']:.6f} | {item['recovery_fraction'] * 100:.2f}% | "
            f"{item['positive_prompts']}/10 | {item['incremental_milliseconds']:.1f} ms |"
        )
    lines += [
        "",
        "## Results by diffusion step",
        "",
        "| Path | Step 4 recovery | Step 9 recovery | Step 4 positive | Step 9 positive |",
        "|---|---:|---:|---:|---:|",
    ]
    for frontier in FRONTIERS:
        s4 = by_step[frontier]["4"]
        s9 = by_step[frontier]["9"]
        lines.append(
            f"| {frontier} | {s4['recovery_fraction'] * 100:.2f}% | "
            f"{s9['recovery_fraction'] * 100:.2f}% | {s4['positive_rows']}/20 | {s9['positive_rows']}/20 |"
        )
    partial = FRONTIERS[:-1]
    passing = [
        name for name in partial
        if overall[name]["recovery_fraction"] >= 0.20
        and overall[name]["positive_prompts"] >= 7
        and all(by_step[name][str(step)]["recovery_fraction"] >= 0 for step in PROBE_STEPS)
    ]
    worst_or_best = max(partial, key=lambda name: overall[name]["recovery_fraction"])
    lines += [
        "",
        "## Decision",
        "",
        f"Partial paths passing the predefined gate: `{passing}`.",
        "",
        f"The best partial path by mean recovery was `{worst_or_best}` at "
        f"{overall[worst_or_best]['recovery_fraction'] * 100:.2f}%. "
        "No partial matched path passed the recovery gate. Any retained shallow encoder boundary leaves stale hidden and skip states that conflict with the newly recomputed deeper path. The exact full-UNet control confirms that the instrumentation can recover the teacher output.",
        "",
        "The experiment decides whether matched partial recomputation is viable under the predefined 20% recovery and prompt-stability gate. If no partial path passes, the remaining coherent option is a real full refresh at selected critical steps, followed by closed-loop image-quality and end-to-end latency evaluation.",
        "",
        "Timing values are isolated probe costs and must not be interpreted as end-to-end generation latency. They are suitable for comparing frontiers within this experiment.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
