"""Prompt-grouped analysis of balanced eight-call development schedules."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
STAGE_REFERENCE = HERE.parent / "stage_aware_v2_dev_20260915" / "run"
DROP_REFERENCE = HERE.parent / "refresh_position_ablation_20260917" / "run"
MODES = ("drop_middle8", "balanced8_a", "balanced8_b")
CANDIDATES = MODES[1:]
METRICS = ("ssim", "psnr_db", "mean_absolute_rgb_error")


def quality_difference(left, right, metric):
    if metric in ("ssim", "psnr_db"):
        return left-right
    if metric == "mean_absolute_rgb_error":
        return right-left
    raise KeyError(metric)


def grouped_difference(rows, left, right, metric, draws=10000):
    prompts = sorted({r["prompt_id"] for r in rows})
    by_prompt = {}
    for prompt in prompts:
        a = {r["case"]: r[metric] for r in rows
             if r["prompt_id"] == prompt and r["mode"] == left}
        b = {r["case"]: r[metric] for r in rows
             if r["prompt_id"] == prompt and r["mode"] == right}
        assert len(a) == len(b) == 2 and a.keys() == b.keys()
        by_prompt[prompt] = float(np.mean(
            [quality_difference(a[case], b[case], metric) for case in sorted(a)]))
    values = np.asarray(list(by_prompt.values()))
    rng = np.random.default_rng(20260920 + len(left) + len(metric))
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return dict(left=left, right=right, metric=metric,
                mean_difference=float(values.mean()),
                prompt_bootstrap_95ci=np.quantile(sampled, [.025, .975]).tolist(),
                prompts_better=int(sum(values > 0)), prompts_worse=int(sum(values < 0)),
                by_prompt=by_prompt)


def main():
    complete = json.loads((RUN / "complete.json").read_text())
    measured = json.loads((RUN / "metrics.json").read_text())
    stage_rows = json.loads((STAGE_REFERENCE / "metrics.json").read_text())
    drop_rows = json.loads((DROP_REFERENCE / "metrics.json").read_text())
    drop = [r for r in drop_rows if r["mode"] == "drop_middle8"]
    uniform = [r for r in stage_rows if r["mode"] == "uniform9"]
    assert complete == {"cases": 20, "generations": 40, "modes": list(CANDIDATES)}
    assert len(measured) == 40 and len(drop) == len(uniform) == 20
    assert all(r["nsfw_flag"] is not True for r in measured)
    all_eight = drop + measured
    all_rows = all_eight + uniform
    for case in {r["case"] for r in all_rows}:
        selected = [r for r in all_rows if r["case"] == case]
        assert len(selected) == 4
        assert len({r["initial_latent_sha256"] for r in selected}) == 1

    vs_drop = [grouped_difference(all_eight, mode, "drop_middle8", metric)
               for mode in CANDIDATES for metric in METRICS]
    vs_uniform = [grouped_difference(all_rows, mode, "uniform9", metric)
                  for mode in MODES for metric in METRICS]
    pairwise = [grouped_difference(all_eight, "balanced8_a", "balanced8_b", metric)
                for metric in METRICS]

    stage_summary = json.loads((STAGE_REFERENCE / "summary.json").read_text())
    drop_summary = json.loads((DROP_REFERENCE / "summary.json").read_text())
    measured_summary = json.loads((RUN / "summary.json").read_text())
    summary = {"uniform9": stage_summary["uniform9"],
               "drop_middle8": drop_summary["drop_middle8"], **measured_summary}
    best = max(MODES, key=lambda mode: summary[mode]["mean_ssim"])
    result = dict(development_prompts=10, seeds_per_prompt=2, held_out_used=False,
                  consumed_held_out_reused=False, comparisons_vs_drop_middle8=vs_drop,
                  comparisons_vs_uniform9=vs_uniform, balanced_pairwise=pairwise,
                  selected_eight_call_schedule=best, summary=summary)
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

    with (HERE / "paired_comparisons.csv").open("w", newline="") as handle:
        fields = ("left", "right", "metric", "mean_difference", "ci_low", "ci_high",
                  "prompts_better", "prompts_worse")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in vs_drop + vs_uniform + pairwise:
            writer.writerow(dict(left=item["left"], right=item["right"], metric=item["metric"],
                mean_difference=item["mean_difference"], ci_low=item["prompt_bootstrap_95ci"][0],
                ci_high=item["prompt_bootstrap_95ci"][1],
                prompts_better=item["prompts_better"], prompts_worse=item["prompts_worse"]))

    names = list(MODES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(names, [summary[m]["mean_ssim"] for m in names],
                color=["#9c755f", "#4c78a8", "#59a14f"])
    axes[0].axhline(summary["uniform9"]["mean_ssim"], color="black", linestyle="--",
                    label="uniform9")
    axes[0].set(ylabel="SSIM vs DPM20", title="Strict eight-call fidelity")
    axes[0].legend(); axes[0].tick_params(axis="x", rotation=20); axes[0].grid(axis="y", alpha=.2)
    axes[1].bar(names, [summary[m]["mean_seconds"] for m in names],
                color=["#9c755f", "#4c78a8", "#59a14f"])
    axes[1].set(ylabel="Seconds", title="Generation time")
    axes[1].tick_params(axis="x", rotation=20); axes[1].grid(axis="y", alpha=.2)
    fig.savefig(HERE / "balanced8_comparison.png", dpi=160); plt.close(fig)

    lines = ["# Balanced eight-call development comparison", "",
             "Ten analysis prompts with two seeds were evaluated. Existing DPM20, "
             "uniform9, and drop-middle8 results were reused; consumed held-out prompts "
             "were not touched.", "", "## Aggregate results", "",
             "| Mode | Max gap | Full calls | Seconds | SSIM | PSNR (dB) | RGB error |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    max_gap = {"uniform9": 3, "drop_middle8": 5, "balanced8_a": 3, "balanced8_b": 3}
    calls = {"uniform9": 9, "drop_middle8": 8, "balanced8_a": 8, "balanced8_b": 8}
    for mode in ("uniform9",) + MODES:
        item = summary[mode]
        lines.append(f'| {mode} | {max_gap[mode]} | {calls[mode]} | '
                     f'{item["mean_seconds"]:.2f} | {item["mean_ssim"]:.4f} | '
                     f'{item["mean_psnr_db"]:.3f} | {item["mean_absolute_rgb_error"]:.4f} |')
    lines += ["", "## Paired improvement over drop-middle8", "",
              "Positive differences favor the balanced schedule. Intervals resample "
              "whole prompt groups.", "",
              "| Candidate | SSIM difference | 95% CI | Prompts better/worse |",
              "| --- | ---: | ---: | ---: |"]
    for mode in CANDIDATES:
        item = next(x for x in vs_drop if x["left"] == mode and x["metric"] == "ssim")
        low, high = item["prompt_bootstrap_95ci"]
        lines.append(f'| {mode} | {item["mean_difference"]:.5f} | '
                     f'[{low:.5f}, {high:.5f}] | '
                     f'{item["prompts_better"]}/{item["prompts_worse"]} |')
    lines += ["", f'Best observed strict eight-call schedule: **{best}**.', "",
              "![Balanced eight-call comparison](balanced8_comparison.png)", "",
              "## Boundary", "",
              "This is development-set schedule selection. A future final claim needs a "
              "newly reserved validation set; the previously consumed held-out prompts "
              "must not be reused."]
    (HERE / "REPORT.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
