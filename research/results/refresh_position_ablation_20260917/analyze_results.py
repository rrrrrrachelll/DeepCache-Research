"""Prompt-grouped analysis of marginal refresh-position value."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
REFERENCE = HERE.parent / "stage_aware_v2_dev_20260915" / "run"
MODES = ("drop_early8", "drop_middle8", "drop_late8")
METRICS = ("ssim", "psnr_db", "mean_absolute_rgb_error", "seconds")


def benefit_value(uniform, dropped, metric):
    if metric in ("ssim", "psnr_db"):
        return uniform - dropped
    if metric == "mean_absolute_rgb_error":
        return dropped - uniform
    if metric == "seconds":
        return uniform - dropped
    raise KeyError(metric)


def grouped_benefit(rows, uniform_rows, mode, metric, draws=10000):
    prompts = sorted({r["prompt_id"] for r in rows})
    values, by_prompt = [], {}
    for prompt in prompts:
        dropped = {r["case"]: r[metric] for r in rows
                   if r["prompt_id"] == prompt and r["mode"] == mode}
        uniform = {r["case"]: r[metric] for r in uniform_rows
                   if r["prompt_id"] == prompt}
        assert len(dropped) == len(uniform) == 2 and dropped.keys() == uniform.keys()
        value = float(np.mean([benefit_value(uniform[case], dropped[case], metric)
                               for case in sorted(uniform)]))
        values.append(value)
        by_prompt[prompt] = value
    values = np.array(values)
    rng = np.random.default_rng(20260917)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return dict(mode=mode, metric=metric, mean_benefit=float(values.mean()),
                prompt_bootstrap_95ci=np.quantile(sampled, [.025, .975]).tolist(),
                prompts_positive=int(sum(values > 0)),
                prompts_negative=int(sum(values < 0)), by_prompt=by_prompt)


def paired_benefit_difference(benefits, left, right, metric, draws=10000):
    a = next(x for x in benefits if x["mode"] == left and x["metric"] == metric)
    b = next(x for x in benefits if x["mode"] == right and x["metric"] == metric)
    prompts = sorted(a["by_prompt"])
    values = np.array([a["by_prompt"][p]-b["by_prompt"][p] for p in prompts])
    rng = np.random.default_rng(20260917 + len(metric))
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return dict(left=left, right=right, metric=metric,
                mean_difference=float(values.mean()),
                prompt_bootstrap_95ci=np.quantile(sampled, [.025, .975]).tolist())


def main():
    complete = json.loads((RUN / "complete.json").read_text())
    rows = json.loads((RUN / "metrics.json").read_text())
    reference_rows = json.loads((REFERENCE / "metrics.json").read_text())
    uniform_rows = [r for r in reference_rows if r["mode"] == "uniform9"]
    dpm_rows = [r for r in reference_rows if r["mode"] == "dpm20"]
    assert complete == {"cases": 20, "generations": 60, "modes": list(MODES)}
    assert len(rows) == 60 and len(uniform_rows) == len(dpm_rows) == 20
    assert all(r["nsfw_flag"] is not True for r in rows)
    for case in {r["case"] for r in rows}:
        selected = [r for r in rows if r["case"] == case]
        assert len(selected) == 3
        assert len({r["initial_latent_sha256"] for r in selected}) == 1
        assert selected[0]["initial_latent_sha256"] == next(
            r["initial_latent_sha256"] for r in uniform_rows if r["case"] == case)

    benefits = [grouped_benefit(rows, uniform_rows, mode, metric)
                for mode in MODES for metric in METRICS]
    pairwise = [paired_benefit_difference(benefits, left, right, metric)
                for metric in ("ssim", "psnr_db", "mean_absolute_rgb_error")
                for left, right in (("drop_early8", "drop_middle8"),
                                    ("drop_early8", "drop_late8"),
                                    ("drop_middle8", "drop_late8"))]
    result = dict(development_prompts=10, seeds_per_prompt=2, held_out_used=False,
                  benefits=benefits, paired_benefit_differences=pairwise)
    (HERE / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (HERE / "marginal_benefits.csv").open("w", newline="") as handle:
        fields = ("mode", "metric", "mean_benefit", "ci_low", "ci_high",
                  "prompts_positive", "prompts_negative")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in benefits:
            writer.writerow(dict(
                mode=item["mode"], metric=item["metric"],
                mean_benefit=item["mean_benefit"],
                ci_low=item["prompt_bootstrap_95ci"][0],
                ci_high=item["prompt_bootstrap_95ci"][1],
                prompts_positive=item["prompts_positive"],
                prompts_negative=item["prompts_negative"]))

    positions = ["early step 2", "middle step 10", "late step 17"]
    ssim = [next(x for x in benefits if x["mode"] == mode and x["metric"] == "ssim")
            for mode in MODES]
    means = np.array([x["mean_benefit"] for x in ssim])
    intervals = np.array([x["prompt_bootstrap_95ci"] for x in ssim])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(positions, means, color=["#4c78a8", "#f2a541", "#59a14f"],
           yerr=np.vstack([means-intervals[:, 0], intervals[:, 1]-means]),
           capsize=5)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(ylabel="SSIM benefit of restoring deleted refresh",
           title="Marginal closed-loop value of one refresh")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(HERE / "refresh_position_value.png", dpi=160)
    plt.close(fig)

    summary = json.loads((RUN / "summary.json").read_text())
    reference_summary = json.loads((REFERENCE / "summary.json").read_text())
    lines = [
        "# Refresh-position ablation", "",
        "Closed-loop development experiment on 10 prompts and two seeds. Each "
        "eight-call mode removes exactly one refresh from uniform9. Held-out "
        "prompts were not used.", "",
        "## Aggregate results", "",
        "| Mode | Deleted step | Full calls | Seconds | SSIM | PSNR (dB) | RGB error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f'| uniform9 | none | 9 | {reference_summary["uniform9"]["mean_seconds"]:.2f} | '
        f'{reference_summary["uniform9"]["mean_ssim"]:.4f} | '
        f'{reference_summary["uniform9"]["mean_psnr_db"]:.3f} | '
        f'{reference_summary["uniform9"]["mean_absolute_rgb_error"]:.4f} |',
    ]
    deleted = {"drop_early8": 2, "drop_middle8": 10, "drop_late8": 17}
    for mode in MODES:
        item = summary[mode]
        lines.append(
            f'| {mode} | {deleted[mode]} | 8 | {item["mean_seconds"]:.2f} | '
            f'{item["mean_ssim"]:.4f} | {item["mean_psnr_db"]:.3f} | '
            f'{item["mean_absolute_rgb_error"]:.4f} |')
    lines += [
        "", "## Marginal refresh benefit", "",
        "Positive values mean restoring the deleted refresh improves fidelity. "
        "Intervals resample whole prompt groups.", "",
        "| Deleted refresh | SSIM benefit | 95% CI | PSNR benefit | RGB-error reduction | Prompts SSIM +/− |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        selected = {x["metric"]: x for x in benefits if x["mode"] == mode}
        s = selected["ssim"]
        low, high = s["prompt_bootstrap_95ci"]
        lines.append(
            f'| step {deleted[mode]} | {s["mean_benefit"]:.5f} | '
            f'[{low:.5f}, {high:.5f}] | '
            f'{selected["psnr_db"]["mean_benefit"]:.4f} | '
            f'{selected["mean_absolute_rgb_error"]["mean_benefit"]:.5f} | '
            f'{s["prompts_positive"]}/{s["prompts_negative"]} |')
    lines += [
        "", "![Marginal refresh-position value](refresh_position_value.png)", "",
        "## Limits", "",
        "- This is development-set policy analysis, not held-out validation.",
        "- Marginal effects are conditional on the other uniform9 refresh positions.",
        "- SSIM and PSNR measure similarity to uncached DPM20, not absolute quality.",
        "- Each case has one timed run per dropout mode.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
