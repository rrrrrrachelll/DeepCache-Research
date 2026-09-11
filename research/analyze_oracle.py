"""Descriptive, prompt-clustered analysis; no threshold fitting or held-out usage."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREDICTORS = ["conv_previous", "conv_cache", "delta_lambda", "lambda_distance", "age"]
TARGETS = ["boundary_error", "mid_error", "guided_noise_error"]


def correlation(x, y, method="spearman"):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float((spearmanr if method == "spearman" else pearsonr)(x, y).statistic)


def within_step_ranks(x, y, steps):
    # Remove the shared sampling-stage schedule without fitting to image targets.
    rx, ry = np.zeros(len(x)), np.zeros(len(y))
    for step in sorted(set(steps)):
        mask = steps == step
        a, b = rankdata(x[mask]), rankdata(y[mask])
        rx[mask], ry[mask] = a-a.mean(), b-b.mean()
    return correlation(rx, ry, "pearson")


def summarize(rows, bootstrap=1000):
    data = [r for r in rows if not r["refresh"]]
    steps = np.array([r["index"] for r in data])
    groups = np.array([r["prompt_id"] for r in data])
    unique = sorted(set(groups))
    rng = np.random.default_rng(20260911)
    clusters = [np.flatnonzero(groups == g) for g in unique]
    samples = [np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
               for _ in range(bootstrap)]
    output = []
    for target in TARGETS:
        y = np.array([r[target] for r in data])
        for predictor in PREDICTORS:
            x = np.array([r[predictor] for r in data])
            estimates = [correlation(x[ix], y[ix]) for ix in samples]
            estimates = [v for v in estimates if v is not None]
            item = dict(target=target, predictor=predictor, n=len(data),
                        pearson=correlation(x, y, "pearson"), spearman=correlation(x, y),
                        prompt_bootstrap_95ci=np.quantile(estimates, [.025, .975]).tolist() if estimates else None,
                        within_step_rank_correlation=within_step_ranks(x, y, steps),
                        by_age={}, by_stage={}, by_prompt={})
            for age in sorted(set(r["age"] for r in data)):
                mask = np.array([r["age"] == age for r in data])
                item["by_age"][str(age)] = correlation(x[mask], y[mask])
            for label, low, high in [("early", 0, 6), ("middle", 7, 13), ("late", 14, 19)]:
                mask = (steps >= low) & (steps <= high)
                item["by_stage"][label] = correlation(x[mask], y[mask])
            for group in unique:
                mask = groups == group
                item["by_prompt"][group] = correlation(x[mask], y[mask])
            output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    complete = json.loads((args.results / "complete.json").read_text())
    rows = []
    for path in sorted(args.results.glob("*_steps.json")):
        rows.extend(json.loads(path.read_text()))
    assert len(rows) == complete["rows"]
    assert all(np.isfinite(r[k]) for r in rows for k in PREDICTORS + TARGETS)
    stats = summarize(rows)
    result = dict(samples=complete["samples"], prompts=len({r["prompt_id"] for r in rows}),
                  reused_steps=sum(not r["refresh"] for r in rows), statistics=stats,
                  caveats=["Fixed interval 3, uncached teacher trajectory only",
                           "Repeated timesteps are not independent samples; CI resamples prompts",
                           "Within-step rank correlation removes shared timestep trends",
                           "Exploratory analysis, not validation of an adaptive policy",
                           "Held-out prompts are not generated or analyzed"])
    (args.results / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    with (args.results / "correlations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "predictor", "pearson", "spearman", "within_step_rank_correlation"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in stats)
    data = [r for r in rows if not r["refresh"]]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, (target, predictor) in zip(axes.flat, [(t, p) for t in ["boundary_error", "guided_noise_error"]
                                                  for p in ["conv_previous", "conv_cache", "lambda_distance"]]):
        dots = axis.scatter([r[predictor] for r in data], [r[target] for r in data],
                            c=[r["index"] for r in data], cmap="viridis", s=16, alpha=.7)
        stat = next(s for s in stats if s["target"] == target and s["predictor"] == predictor)
        axis.set(xlabel=predictor, ylabel=target, title=f'Spearman = {stat["spearman"]:.3f}')
        axis.grid(alpha=.2)
    fig.colorbar(dots, ax=axes, label="DPM20 step index", shrink=.7)
    fig.savefig(args.results / "proxy_scatter.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, key in zip(axes, ["boundary_error", "guided_noise_error"]):
        for prompt in sorted({r["prompt_id"] for r in rows}):
            selected = [r for r in rows if r["prompt_id"] == prompt]
            values = [np.mean([r[key] for r in selected if r["index"] == i]) for i in range(20)]
            axis.plot(range(20), values, alpha=.7, linewidth=1, label=prompt)
        axis.set(xlabel="Step index (refresh steps have zero error)", ylabel=key)
        axis.grid(alpha=.2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(args.results / "error_by_step.png", dpi=160)
    plt.close(fig)
    def fmt(x):
        return "N/A (constant within step)" if x is None else f"{x:.3f}"
    lines = ["# Oracle Cache Error Analysis", "",
             f'{result["prompts"]} analysis prompts, {result["samples"]} images, {result["reused_steps"]} reuse observations.',
             "", "Refresh steps are excluded from all correlations. Confidence intervals resample whole prompts.", "",
             "| Target | Proxy | Spearman | 95% prompt CI | Within-step rank correlation |",
             "|---|---|---:|---|---:|"]
    for row in stats:
        ci = row["prompt_bootstrap_95ci"]
        lines.append(f'| {row["target"]} | {row["predictor"]} | {fmt(row["spearman"])} | '
                     f'[{ci[0]:.3f}, {ci[1]:.3f}] | {fmt(row["within_step_rank_correlation"])} |')
    lines += ["", "## Interpretation Limits", ""] + ["- " + x for x in result["caveats"]]
    lines += ["- The saved images are uncached reference images, not adaptive-policy results.",
              "- Diagnostic runtime includes oracle passes, hooks and reductions; it is not an inference speed benchmark.",
              "- Local guided-noise error is not final perceptual image quality or a solver truncation-error estimate."]
    (args.results / "REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
