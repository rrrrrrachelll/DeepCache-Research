"""Analyze the completed stage-aware closed-loop development experiment."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
METRICS = ("ssim", "psnr_db", "mean_absolute_rgb_error", "seconds")
DIRECTIONS = {"ssim": 1, "psnr_db": 1, "mean_absolute_rgb_error": -1, "seconds": -1}


def paired_prompt_bootstrap(rows, left, right, metric, draws=10000):
    prompts = sorted({r["prompt_id"] for r in rows})
    differences = []
    by_prompt = {}
    for prompt in prompts:
        a = [r[metric] for r in rows if r["prompt_id"] == prompt and r["mode"] == left]
        b = [r[metric] for r in rows if r["prompt_id"] == prompt and r["mode"] == right]
        assert len(a) == len(b) == 2
        by_prompt[prompt] = float(np.mean(a) - np.mean(b))
        differences.append(by_prompt[prompt])
    values = np.array(differences)
    rng = np.random.default_rng(20260915)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return dict(left=left, right=right, metric=metric,
                mean_difference=float(values.mean()),
                prompt_bootstrap_95ci=np.quantile(samples, [.025, .975]).tolist(),
                prompts_better=int(sum(DIRECTIONS[metric] * v > 0 for v in values)),
                prompts_worse=int(sum(DIRECTIONS[metric] * v < 0 for v in values)),
                by_prompt=by_prompt)


def main():
    complete = json.loads((RUN / "complete.json").read_text())
    rows = json.loads((RUN / "metrics.json").read_text())
    summary = json.loads((RUN / "summary.json").read_text())
    assert complete == {"cases": 20, "generations": 80,
                        "modes": ["dpm20", "fixed3", "uniform9", "stage_aware9"]}
    assert len(rows) == 80
    assert all(r["nsfw_flag"] is False for r in rows)
    for case in {r["case"] for r in rows}:
        selected = [r for r in rows if r["case"] == case]
        assert len(selected) == 4
        assert len({r["initial_latent_sha256"] for r in selected}) == 1
    comparisons = []
    for metric in METRICS:
        comparisons.append(paired_prompt_bootstrap(
            rows, "stage_aware9", "uniform9", metric))
    for metric in METRICS:
        comparisons.append(paired_prompt_bootstrap(
            rows, "stage_aware9", "fixed3", metric))
    result = dict(comparisons=comparisons, development_prompts=10,
                  seeds_per_prompt=2, held_out_used=False)
    (HERE / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (HERE / "paired_comparisons.csv").open("w", newline="") as handle:
        fields = ("left", "right", "metric", "mean_difference",
                  "ci_low", "ci_high", "prompts_better", "prompts_worse")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in comparisons:
            writer.writerow({
                "left": item["left"], "right": item["right"],
                "metric": item["metric"],
                "mean_difference": item["mean_difference"],
                "ci_low": item["prompt_bootstrap_95ci"][0],
                "ci_high": item["prompt_bootstrap_95ci"][1],
                "prompts_better": item["prompts_better"],
                "prompts_worse": item["prompts_worse"],
            })

    modes = ["dpm20", "fixed3", "uniform9", "stage_aware9"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = ["#777777", "#5588aa", "#d28e2b", "#4a9b67"]
    axes[0].bar(modes, [summary[m]["mean_seconds"] for m in modes], color=colors)
    axes[0].set(ylabel="Mean seconds", title="Closed-loop runtime")
    axes[1].bar(modes, [summary[m]["mean_ssim"] for m in modes], color=colors)
    axes[1].set(ylabel="SSIM vs uncached DPM20", title="Reference similarity")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=.2)
    fig.savefig(HERE / "runtime_quality.png", dpi=160)
    plt.close(fig)

    stage_uniform = {x["metric"]: x for x in comparisons
                     if x["right"] == "uniform9"}
    lines = [
        "# Stage-aware V2 development experiment", "",
        "Closed-loop SD1.5/DPM-Solver++ generation on 10 development prompts "
        "and two seeds. Held-out prompts were not used.", "",
        "## Aggregate results", "",
        "| Mode | Full calls | Mean seconds | Speedup | SSIM | PSNR (dB) | Mean RGB error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in modes:
        item = summary[mode]
        psnr = "infinite" if item["mean_psnr_db"] is None else f'{item["mean_psnr_db"]:.3f}'
        lines.append(
            f'| {mode} | {item["mean_full_calls"]:.0f} | {item["mean_seconds"]:.2f} | '
            f'{item["speedup_vs_dpm20"]:.2f}x | {item["mean_ssim"]:.4f} | '
            f'{psnr} | {item["mean_absolute_rgb_error"]:.4f} |')
    lines += [
        "", "## Equal-budget comparison", "",
        "Stage-aware9 minus uniform9; both use exactly nine full UNet calls. "
        "Intervals resample whole prompt groups, keeping the two seeds together.", "",
        "| Metric | Mean difference | 95% prompt-bootstrap interval | Prompts better/worse |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        item = stage_uniform[metric]
        low, high = item["prompt_bootstrap_95ci"]
        lines.append(
            f'| {metric} | {item["mean_difference"]:.6f} | '
            f'[{low:.6f}, {high:.6f}] | {item["prompts_better"]}/{item["prompts_worse"]} |')
    lines += [
        "", "![Runtime and reference similarity](runtime_quality.png)", "",
        "## Limits", "",
        "- This is policy selection on development prompts, not held-out validation.",
        "- SSIM and PSNR measure similarity to uncached DPM20, not absolute image quality.",
        "- Each case has one timed run per mode; timing differences need repeated measurements.",
        "- The experiment tests one stage-aware allocation at one nine-call budget.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
