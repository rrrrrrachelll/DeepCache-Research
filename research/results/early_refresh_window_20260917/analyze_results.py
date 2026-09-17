"""Prompt-grouped analysis of the early refresh-window sweep."""
import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
REFERENCE = HERE.parent / "stage_aware_v2_dev_20260915" / "run"
MODES = ("early1", "early2", "early3", "early4")
MEASURED = ("early1", "early3", "early4")
METRICS = ("ssim", "psnr_db", "mean_absolute_rgb_error")


def quality_difference(candidate, reference, metric):
    if metric in ("ssim", "psnr_db"):
        return candidate - reference
    if metric == "mean_absolute_rgb_error":
        return reference - candidate
    raise KeyError(metric)


def grouped_difference(candidate_rows, reference_rows, mode, metric, draws=10000):
    prompts = sorted({r["prompt_id"] for r in candidate_rows})
    values, by_prompt = [], {}
    for prompt in prompts:
        candidate = {r["case"]: r[metric] for r in candidate_rows
                     if r["prompt_id"] == prompt and r["mode"] == mode}
        reference = {r["case"]: r[metric] for r in reference_rows
                     if r["prompt_id"] == prompt}
        assert len(candidate) == len(reference) == 2 and candidate.keys() == reference.keys()
        value = float(np.mean([quality_difference(candidate[case], reference[case], metric)
                               for case in sorted(reference)]))
        values.append(value)
        by_prompt[prompt] = value
    values = np.asarray(values)
    rng = np.random.default_rng(20260917 + int(mode[-1]) + len(metric))
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return dict(mode=mode, reference="early2", metric=metric,
                mean_difference=float(values.mean()),
                prompt_bootstrap_95ci=np.quantile(sampled, [.025, .975]).tolist(),
                prompts_better=int(sum(values > 0)), prompts_worse=int(sum(values < 0)),
                by_prompt=by_prompt)


def paired_mode_difference(all_rows, left, right, metric, draws=10000):
    prompts = sorted({r["prompt_id"] for r in all_rows})
    values = []
    for prompt in prompts:
        a = {r["case"]: r[metric] for r in all_rows
             if r["prompt_id"] == prompt and r["mode"] == left}
        b = {r["case"]: r[metric] for r in all_rows
             if r["prompt_id"] == prompt and r["mode"] == right}
        assert len(a) == len(b) == 2 and a.keys() == b.keys()
        values.append(float(np.mean([quality_difference(a[c], b[c], metric)
                                     for c in sorted(a)])))
    values = np.asarray(values)
    rng = np.random.default_rng(20261000 + int(left[-1])*10 + int(right[-1]) + len(metric))
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return dict(left=left, right=right, metric=metric,
                mean_difference=float(values.mean()),
                prompt_bootstrap_95ci=np.quantile(sampled, [.025, .975]).tolist())


def grouped_mean(rows, mode, metric="ssim", draws=10000):
    prompts = sorted({r["prompt_id"] for r in rows})
    values = np.asarray([np.mean([r[metric] for r in rows
                                  if r["prompt_id"] == prompt and r["mode"] == mode])
                         for prompt in prompts])
    rng = np.random.default_rng(20261100 + int(mode[-1]))
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return dict(mode=mode, mean=float(values.mean()),
                prompt_bootstrap_95ci=np.quantile(sampled, [.025, .975]).tolist())


def main():
    complete = json.loads((RUN / "complete.json").read_text())
    measured_rows = json.loads((RUN / "metrics.json").read_text())
    reference_rows = json.loads((REFERENCE / "metrics.json").read_text())
    early2 = [dict(r, mode="early2", second_refresh=2)
              for r in reference_rows if r["mode"] == "uniform9"]
    assert complete == {"cases": 20, "generations": 60, "modes": list(MEASURED)}
    assert len(measured_rows) == 60 and len(early2) == 20
    assert all(r["nsfw_flag"] is not True for r in measured_rows)
    all_rows = measured_rows + early2
    for case in {r["case"] for r in all_rows}:
        selected = [r for r in all_rows if r["case"] == case]
        assert len(selected) == 4
        assert len({r["initial_latent_sha256"] for r in selected}) == 1

    differences = [grouped_difference(measured_rows, early2, mode, metric)
                   for mode in MEASURED for metric in METRICS]
    pairwise = [paired_mode_difference(all_rows, left, right, metric)
                for metric in METRICS for left, right in itertools.combinations(MODES, 2)]
    means = [grouped_mean(all_rows, mode) for mode in MODES]
    result = dict(development_prompts=10, seeds_per_prompt=2, held_out_used=False,
                  reference_mode="early2", differences_vs_early2=differences,
                  pairwise_differences=pairwise, grouped_ssim_means=means)
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

    with (HERE / "early_window_differences.csv").open("w", newline="") as handle:
        fields = ("mode", "reference", "metric", "mean_difference", "ci_low",
                  "ci_high", "prompts_better", "prompts_worse")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in differences:
            writer.writerow(dict(mode=item["mode"], reference=item["reference"],
                                 metric=item["metric"],
                                 mean_difference=item["mean_difference"],
                                 ci_low=item["prompt_bootstrap_95ci"][0],
                                 ci_high=item["prompt_bootstrap_95ci"][1],
                                 prompts_better=item["prompts_better"],
                                 prompts_worse=item["prompts_worse"]))

    x = np.arange(1, 5)
    y = np.asarray([item["mean"] for item in means])
    ci = np.asarray([item["prompt_bootstrap_95ci"] for item in means])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.errorbar(x, y, yerr=np.vstack([y-ci[:, 0], ci[:, 1]-y]), marker="o",
                linewidth=2, capsize=5, color="#4c78a8")
    ax.set(xticks=x, xlabel="Second full-UNet refresh step", ylabel="SSIM vs DPM20",
           title="Equal-budget early refresh-window sweep")
    ax.grid(alpha=.25)
    fig.savefig(HERE / "early_refresh_window.png", dpi=160)
    plt.close(fig)

    run_summary = json.loads((RUN / "summary.json").read_text())
    ref_summary = json.loads((REFERENCE / "summary.json").read_text())["uniform9"]
    aggregate = {
        "early1": run_summary["early1"], "early2": ref_summary,
        "early3": run_summary["early3"], "early4": run_summary["early4"],
    }
    lines = [
        "# Early refresh-window sweep", "",
        "Closed-loop development experiment on 10 prompts and two seeds. Every "
        "schedule uses nine full UNet calls; only the second refresh moves among "
        "steps 1, 2, 3, and 4. Held-out prompts were not used.", "",
        "## Aggregate results", "",
        "| Second refresh | Seconds | SSIM | PSNR (dB) | RGB error |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        item = aggregate[mode]
        lines.append(f'| {mode[-1]} | {item["mean_seconds"]:.2f} | '
                     f'{item["mean_ssim"]:.4f} | {item["mean_psnr_db"]:.3f} | '
                     f'{item["mean_absolute_rgb_error"]:.4f} |')
    lines += ["", "## Difference from step 2", "",
              "Positive values mean the candidate is closer to uncached DPM20. "
              "Intervals resample whole prompt groups.", "",
              "| Candidate step | SSIM difference | 95% CI | Prompts better/worse |",
              "| ---: | ---: | ---: | ---: |"]
    for mode in MEASURED:
        item = next(x for x in differences if x["mode"] == mode and x["metric"] == "ssim")
        low, high = item["prompt_bootstrap_95ci"]
        lines.append(f'| {mode[-1]} | {item["mean_difference"]:.5f} | '
                     f'[{low:.5f}, {high:.5f}] | '
                     f'{item["prompts_better"]}/{item["prompts_worse"]} |')
    lines += ["", "![Early refresh-window sweep](early_refresh_window.png)", "",
              "## Limits", "",
              "- This is development-set policy analysis, not held-out validation.",
              "- SSIM and PSNR measure similarity to uncached DPM20, not absolute quality.",
              "- Step 2 reuses the prior uniform9 run; the other positions were newly measured.",
              "- Each case has one timed run per newly measured mode."]
    (HERE / "REPORT.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
