"""Fixed-interval Oracle sweep: age/step errors and prompt-held-out prediction."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "oracle_20260911"
INTERVALS = (2, 3, 4, 5)
TARGETS = ("guided_noise_error", "boundary_error")
MODELS = {
    "intercept": (),
    "age": ("age",),
    "lambda_distance": ("lambda_distance",),
    "age+lambda_distance": ("age", "lambda_distance"),
}


def load_runs():
    runs = {3: BASE, 2: HERE / "interval_2", 4: HERE / "interval_4",
            5: HERE / "interval_5"}
    rows = []
    reference = None
    image_hashes = None
    for interval in INTERVALS:
        directory = runs[interval]
        manifest = json.loads((directory / "manifest.json").read_text())
        complete = json.loads((directory / "complete.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        assert manifest["interval"] == interval
        assert complete == {"samples": 20, "rows": 400}
        config = {key: manifest[key] for key in ("model", "prompts", "cases", "steps",
                                                  "branch", "guidance_scale",
                                                  "resolution", "dtype", "scheduler")}
        # Diffusers stores this semantically unordered set as a list; ordering
        # can vary between processes because the source is a Python set.
        config["scheduler"]["_use_default_values"] = sorted(
            config["scheduler"].get("_use_default_values", []))
        if reference is None:
            reference = config
        else:
            assert config == reference, f"Protocol mismatch at interval {interval}"
        hashes = {m["case"]: (m["initial_latent_sha256"],
                             hashlib.sha256((directory / (m["case"] + ".png")).read_bytes()).hexdigest())
                  for m in metrics}
        assert len(hashes) == 20
        if image_hashes is None:
            image_hashes = hashes
        else:
            assert hashes == image_hashes, f"Reference image or latent mismatch at interval {interval}"
        for path in sorted(directory.glob("*_steps.json")):
            trace = json.loads(path.read_text())
            assert len(trace) == 20
            for step, row in enumerate(trace):
                assert row["index"] == step
                assert row["refresh"] == (step % interval == 0)
                assert row["age"] == step % interval
                assert row["prompt_id"] in {p["id"] for p in manifest["prompts"]["analysis"]}
                assert np.isfinite([row[k] for k in (*TARGETS, "age", "lambda_distance",
                                                     "conv_previous", "conv_cache")]).all()
                rows.append({"interval": interval, **row})
    assert len(rows) == 1600
    return rows


def fit_predict(train, test, features, target):
    y = np.array([r[target] for r in train], dtype=float)
    if not features:
        return np.full(len(test), y.mean())
    x_train = np.array([[r[f] for f in features] for r in train], dtype=float)
    x_test = np.array([[r[f] for f in features] for r in test], dtype=float)
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack([np.ones(len(train)), (x_train - center) / scale])
    coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    return np.column_stack([np.ones(len(test)), (x_test - center) / scale]) @ coeff


def leave_one_prompt_out(rows):
    reuse = [r for r in rows if not r["refresh"]]
    prompts = sorted({r["prompt_id"] for r in reuse})
    assert len(prompts) == 10
    predictions = []
    for target in TARGETS:
        for held_out in prompts:
            train = [r for r in reuse if r["prompt_id"] != held_out]
            test = [r for r in reuse if r["prompt_id"] == held_out]
            actual = np.array([r[target] for r in test])
            for model, features in MODELS.items():
                estimate = fit_predict(train, test, features, target)
                for row, truth, prediction in zip(test, actual, estimate):
                    predictions.append(dict(target=target, model=model, prompt_id=held_out,
                                            interval=row["interval"], index=row["index"],
                                            age=row["age"], actual=float(truth),
                                            predicted=float(prediction)))
    results = []
    for target in TARGETS:
        for model in MODELS:
            selected = [r for r in predictions if r["target"] == target and r["model"] == model]
            residual = np.array([r["predicted"] - r["actual"] for r in selected])
            folds = {}
            for prompt in prompts:
                fold = np.array([r["predicted"] - r["actual"] for r in selected
                                 if r["prompt_id"] == prompt])
                folds[prompt] = dict(rmse=float(np.sqrt(np.mean(fold**2))),
                                     mae=float(np.mean(np.abs(fold))))
            results.append(dict(target=target, model=model, n=len(selected),
                                rmse=float(np.sqrt(np.mean(residual**2))),
                                mae=float(np.mean(np.abs(residual))), folds=folds))
    return results, predictions



def paired_model_comparisons(results, draws=5000):
    """Prompt-bootstrap intervals for paired LOPO RMSE differences."""
    rng = np.random.default_rng(20260914)
    comparisons = []
    for target in TARGETS:
        by_model = {r["model"]: r for r in results if r["target"] == target}
        for simple, richer in (("age", "lambda_distance"),
                               ("age", "age+lambda_distance"),
                               ("lambda_distance", "age+lambda_distance")):
            prompts = sorted(by_model[simple]["folds"])
            squared_simple = np.array([by_model[simple]["folds"][p]["rmse"] ** 2 for p in prompts])
            squared_richer = np.array([by_model[richer]["folds"][p]["rmse"] ** 2 for p in prompts])
            samples = rng.integers(0, len(prompts), size=(draws, len(prompts)))
            delta = (np.sqrt(squared_richer[samples].mean(axis=1))
                     - np.sqrt(squared_simple[samples].mean(axis=1)))
            comparisons.append(dict(target=target, simple=simple, richer=richer,
                                    delta_rmse=by_model[richer]["rmse"] - by_model[simple]["rmse"],
                                    prompt_bootstrap_95ci=np.quantile(delta, [.025, .975]).tolist()))
    return comparisons


def age_step_stats(rows):
    cells = defaultdict(list)
    for row in rows:
        if not row["refresh"]:
            cells[(row["age"], row["index"])].append(row["guided_noise_error"])
    return [dict(age=age, step=step, n=len(values), mean=float(np.mean(values)),
                 median=float(np.median(values)))
            for (age, step), values in sorted(cells.items())]



def stage_age_stats(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["refresh"]:
            continue
        stage = "early" if row["index"] <= 6 else "middle" if row["index"] <= 13 else "late"
        for target in TARGETS:
            groups[(stage, row["age"], target)].append(row[target])
    return [dict(stage=stage, age=age, target=target, n=len(values),
                 mean=float(np.mean(values)), median=float(np.median(values)))
            for (stage, age, target), values in sorted(groups.items())]


def plots(cells, results):
    grid = np.full((4, 20), np.nan)
    counts = np.zeros((4, 20), dtype=int)
    for cell in cells:
        grid[cell["age"] - 1, cell["step"]] = cell["mean"]
        counts[cell["age"] - 1, cell["step"]] = cell["n"]
    fig, ax = plt.subplots(figsize=(15, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, aspect="auto", origin="lower")
    for age in range(4):
        for step in range(20):
            if counts[age, step]:
                ax.text(step, age, str(counts[age, step]), ha="center", va="center",
                        color="white" if grid[age, step] > np.nanmax(grid) * .45 else "black",
                        fontsize=7)
    ax.set(xlabel="DPM-Solver++ step index", ylabel="Cache age (reuse calls)",
           title="Mean guided-noise relative L1 error | age, step (cell text = n)")
    ax.set_xticks(range(20))
    ax.set_yticks(range(4), labels=range(1, 5))
    fig.colorbar(image, ax=ax, label="Mean guided-noise error")
    fig.savefig(HERE / "guided_error_by_age_step.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        subset = [r for r in results if r["target"] == target]
        ax.bar(range(len(subset)), [r["rmse"] for r in subset], color="#426f9a")
        ax.set_xticks(range(len(subset)), [r["model"].replace("lambda_distance", "λ distance")
                                            for r in subset], rotation=18)
        ax.set(ylabel="Leave-one-prompt-out RMSE (relative L1)",
               title=target.replace("_", " "))
        ax.grid(axis="y", alpha=.2)
    fig.savefig(HERE / "lopo_model_rmse.png", dpi=160)
    plt.close(fig)


def main():
    rows = load_runs()
    cells = age_step_stats(rows)
    stages = stage_age_stats(rows)
    results, predictions = leave_one_prompt_out(rows)
    comparisons = paired_model_comparisons(results)
    (HERE / "analysis.json").write_text(json.dumps(
        {"intervals": INTERVALS, "samples": 80, "rows": len(rows),
         "reuse_rows": sum(not r["refresh"] for r in rows),
         "age_step": cells, "stage_age": stages, "lopo": results,
         "paired_model_comparisons": comparisons}, indent=2, allow_nan=False) + "\n")
    with (HERE / "age_step.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("age", "step", "n", "mean", "median"))
        writer.writeheader()
        writer.writerows(cells)
    with (HERE / "lopo_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    with (HERE / "stage_age.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stage", "age", "target", "n", "mean", "median"))
        writer.writeheader()
        writer.writerows(stages)
    plots(cells, results)
    lines = ["# Fixed-interval Oracle sweep", "",
             "Intervals 2, 3, 4, 5; interval 3 reuses the completed 2026-09-11 run.",
             f"80 paired images, {len(rows)} UNet steps, "
             f"{sum(not r['refresh'] for r in rows)} reuse observations. "
             "The same reference image and latent hash matched across all intervals.",
             "", "## Leave-one-prompt-out prediction", "",
             "Linear least-squares models with an intercept were trained on nine prompts "
             "and evaluated on the excluded prompt, repeating for all ten prompts. "
             "Both seeds and all intervals of each held-out prompt stay together. "
             "RMSE and MAE pool out-of-prompt predictions; refresh steps are excluded.",
             "", "| Target | Model | RMSE | MAE |",
             "| --- | --- | ---: | ---: |"]
    for row in results:
        lines.append(f'| {row["target"]} | {row["model"]} | {row["rmse"]:.5f} | {row["mae"]:.5f} |')
    lines += ["", "## Paired model differences", "",
              "Negative delta RMSE favors the richer model. Percentile intervals "
              "resample the ten held-out prompt groups 5,000 times.",
              "", "| Target | Simple | Richer | Delta RMSE | 95% prompt-bootstrap interval |",
              "| --- | --- | --- | ---: | ---: |"]
    for row in comparisons:
        low, high = row["prompt_bootstrap_95ci"]
        lines.append(f'| {row["target"]} | {row["simple"]} | {row["richer"]} | '
                     f'{row["delta_rmse"]:.5f} | [{low:.5f}, {high:.5f}] |')
    lines += ["", "## Age and step", "",
              "![Mean guided-noise error conditional on age and step](guided_error_by_age_step.png)",
              "", "Cell labels show observation counts. Gray cells have no reuse observations; "
              "refresh-step zeros are excluded. The errors are local to the uncached "
              "teacher trajectory, not final generated-image quality.",
              "", "## Stage and age averages", "",
              "Stages follow the earlier Oracle analysis: early steps 0-6, "
              "middle 7-13, late 14-19. Values are relative L1 means on reuse rows.",
              "", "| Stage | Age | n | Guided-noise error | Boundary error |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for stage in ("early", "middle", "late"):
        for age in range(1, 5):
            guided = next((r for r in stages if r["stage"] == stage and
                           r["age"] == age and r["target"] == "guided_noise_error"), None)
            boundary = next((r for r in stages if r["stage"] == stage and
                             r["age"] == age and r["target"] == "boundary_error"), None)
            if guided is not None:
                lines.append(f'| {stage} | {age} | {guided["n"]} | '
                             f'{guided["mean"]:.5f} | {boundary["mean"]:.5f} |')
    lines += ["", "## Interpretation limits", "",
              "- All intervals share the same ten development prompts; five held-out prompts were not used.",
              "- Age and lambda distance are correlated; linear predictive value does not identify a causal driver.",
              "- This sweep does not test a V2 policy or compare final image quality at equal compute budgets."]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:22]))


if __name__ == "__main__":
    main()
