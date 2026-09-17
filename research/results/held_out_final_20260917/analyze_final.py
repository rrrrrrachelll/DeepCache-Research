"""Analyze the frozen one-time held-out validation."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
MODES = ("dpm20", "fixed3", "uniform9", "stage_aware9", "drop_middle8")
CACHED_MODES = MODES[1:]


def quality_difference(left, right, metric):
    if metric in ("ssim", "psnr_db"):
        return left - right
    if metric == "mean_absolute_rgb_error":
        return right - left
    raise KeyError(metric)


def paired_prompt_difference(rows, left, right, metric, draws=20000):
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
    rng = np.random.default_rng(20260917 + len(metric) + len(left) + len(right))
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return dict(left=left, right=right, metric=metric,
                mean_difference=float(values.mean()),
                prompt_bootstrap_95ci=np.quantile(sampled, [.025, .975]).tolist(),
                prompts_better=int(sum(values > 0)), prompts_worse=int(sum(values < 0)),
                by_prompt=by_prompt)


def evaluate_decision(primary, summary, mean_loss_margin=.01,
                      prompt_loss_threshold=.03, max_bad_prompts=1):
    bad_prompts = sorted(prompt for prompt, value in primary["by_prompt"].items()
                         if value < -prompt_loss_threshold)
    checks = {
        "mean_ssim_loss_at_most_0.01": primary["mean_difference"] >= -mean_loss_margin,
        "at_most_one_prompt_loses_over_0.03": len(bad_prompts) <= max_bad_prompts,
        "exactly_eight_full_calls": summary["drop_middle8"]["mean_full_calls"] == 8,
        "faster_than_uniform9": (summary["drop_middle8"]["mean_seconds"] <
                                 summary["uniform9"]["mean_seconds"]),
    }
    return dict(accepted=all(checks.values()), checks=checks,
                prompts_losing_over_0_03=bad_prompts)


def main():
    complete = json.loads((RUN / "complete.json").read_text())
    rows = json.loads((RUN / "metrics.json").read_text())
    summary = json.loads((RUN / "summary.json").read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    assert complete == {"cases": 10, "generations": 50, "modes": list(MODES)}
    assert manifest["held_out_used"] is True
    assert len(rows) == 50 and all(r["nsfw_flag"] is not True for r in rows)
    for case in {r["case"] for r in rows}:
        selected = [r for r in rows if r["case"] == case]
        assert len(selected) == 5
        assert len({r["initial_latent_sha256"] for r in selected}) == 1

    contrasts = [paired_prompt_difference(rows, mode, "uniform9", metric)
                 for mode in ("fixed3", "stage_aware9", "drop_middle8")
                 for metric in ("ssim", "psnr_db", "mean_absolute_rgb_error")]
    primary = next(x for x in contrasts
                   if x["left"] == "drop_middle8" and x["metric"] == "ssim")
    decision = evaluate_decision(primary, summary)
    result = dict(held_out_prompts=5, seeds_per_prompt=2,
                  frozen_primary_contrast=primary, contrasts_vs_uniform9=contrasts,
                  decision=decision, summary=summary)
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

    with (HERE / "paired_contrasts.csv").open("w", newline="") as handle:
        fields = ("left", "right", "metric", "mean_difference", "ci_low", "ci_high",
                  "prompts_better", "prompts_worse")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in contrasts:
            writer.writerow(dict(left=item["left"], right=item["right"],
                                 metric=item["metric"], mean_difference=item["mean_difference"],
                                 ci_low=item["prompt_bootstrap_95ci"][0],
                                 ci_high=item["prompt_bootstrap_95ci"][1],
                                 prompts_better=item["prompts_better"],
                                 prompts_worse=item["prompts_worse"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    names = list(CACHED_MODES)
    axes[0].bar(names, [summary[m]["mean_ssim"] for m in names], color="#4c78a8")
    axes[0].set(ylabel="SSIM vs DPM20", title="Held-out fidelity")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=.2)
    axes[1].bar(names, [summary[m]["speedup_vs_dpm20"] for m in names], color="#59a14f")
    axes[1].set(ylabel="Speedup vs DPM20", title="Held-out runtime")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=.2)
    fig.savefig(HERE / "held_out_quality_speed.png", dpi=160)
    plt.close(fig)

    lines = ["# One-time held-out final validation", "",
             "Five previously unused prompts with two seeds were evaluated once under "
             "the frozen protocol. Positive paired differences favor the method in the "
             "left column.", "", "## Aggregate results", "",
             "| Mode | Full calls | Seconds | Speedup | SSIM | PSNR (dB) | RGB error |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode in MODES:
        item = summary[mode]
        psnr = "—" if item["mean_psnr_db"] is None else f'{item["mean_psnr_db"]:.3f}'
        lines.append(f'| {mode} | {item["mean_full_calls"]:.0f} | '
                     f'{item["mean_seconds"]:.2f} | {item["speedup_vs_dpm20"]:.2f}× | '
                     f'{item["mean_ssim"]:.4f} | {psnr} | '
                     f'{item["mean_absolute_rgb_error"]:.4f} |')
    low, high = primary["prompt_bootstrap_95ci"]
    lines += ["", "## Frozen primary decision", "",
              f'- Paired SSIM difference, drop_middle8 − uniform9: '
              f'**{primary["mean_difference"]:.5f}** (prompt-bootstrap 95% CI '
              f'[{low:.5f}, {high:.5f}]).',
              f'- Prompt groups better/worse: {primary["prompts_better"]}/'
              f'{primary["prompts_worse"]}.',
              f'- Prompt groups losing more than 0.03: '
              f'{", ".join(decision["prompts_losing_over_0_03"]) or "none"}.',
              f'- Frozen decision: **{"ACCEPT" if decision["accepted"] else "REJECT"} '
              f'drop_middle8**.', "", "| Frozen check | Pass |", "| --- | ---: |"]
    for name, passed in decision["checks"].items():
        lines.append(f'| {name} | {"yes" if passed else "no"} |')
    lines += ["", "![Held-out quality and speed](held_out_quality_speed.png)", "",
              "## Interpretation boundary", "",
              "These held-out prompts are now consumed. This result may select between "
              "the frozen candidates, but it must not be used to tune another schedule "
              "and then claim validation on the same prompts."]
    (HERE / "REPORT.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
