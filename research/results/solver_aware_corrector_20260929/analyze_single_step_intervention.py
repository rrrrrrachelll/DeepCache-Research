"""Analyze Experiment 13A.5 with paired final-image metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL
import lpips
from DISTS_pytorch import DISTS

from research.results.solver_aware_corrector_20260929.analyze_branch_frontier_train12 import (
    gaussian_similarity, image_tensor, save_json)
from research.results.solver_aware_corrector_20260929.run_single_step_intervention import POLICIES


def load_interventions(data_dir: Path) -> dict[tuple[str, str], dict]:
    artifacts = {}
    for path in sorted(data_dir.glob("*.pt")):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        key = (artifact["prompt_id"], artifact["policy"])
        if key in artifacts:
            raise RuntimeError(f"Duplicate intervention artifact {key}")
        artifacts[key] = artifact
    return artifacts


def decode_images(input_dir: Path, artifacts: dict, model_path: str) -> Path:
    images_dir = input_dir / "images"
    images_dir.mkdir(exist_ok=True)
    missing = [artifact for artifact in artifacts.values()
               if not (images_dir / f"{artifact['case']}.png").exists()]
    if not missing:
        print("decode resume: all images exist", flush=True)
        return images_dir
    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float16,
        local_files_only=True).to("cuda").eval()
    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
    for index, artifact in enumerate(sorted(missing, key=lambda item: item["case"]), 1):
        latent = artifact["final_latent"].to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            decoded = vae.decode(latent / scaling).sample
        image = ((decoded[0].float().cpu() / 2 + 0.5).clamp(0, 1)
                 .permute(1, 2, 0).numpy())
        Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(
            images_dir / f"{artifact['case']}.png")
        print(f"decode [{index}/{len(missing)}] {artifact['case']}", flush=True)
    del vae
    torch.cuda.empty_cache()
    return images_dir


def compute_metrics(input_dir: Path, reference_dir: Path, images_dir: Path,
                    artifacts: dict) -> list[dict]:
    metrics_path = input_dir / "metrics.json"
    if metrics_path.exists():
        rows = json.loads(metrics_path.read_text())
        if len(rows) == 132:
            print("metrics resume: 132 rows", flush=True)
            return rows

    reference_metrics = json.loads((reference_dir / "metrics.json").read_text())
    baseline = {(row["prompt_id"], row["branch_id"]): row
                for row in reference_metrics}
    perceptual = lpips.LPIPS(net="alex").eval().to("cuda")
    dists_model = DISTS().eval().to("cuda")
    rows = []
    total = len(artifacts)
    for prompt_id, policy in sorted(artifacts):
        artifact = artifacts[prompt_id, policy]
        full_artifact = torch.load(
            reference_dir / "data" / f"{prompt_id}__full.pt",
            map_location="cpu", weights_only=False)
        full_image = np.asarray(Image.open(
            reference_dir / "images" / f"{prompt_id}__full.png").convert("RGB"))
        candidate = np.asarray(Image.open(
            images_dir / f"{artifact['case']}.png").convert("RGB"))
        full_tensor = image_tensor(full_image).to("cuda")
        candidate_tensor = image_tensor(candidate).to("cuda")
        traditional = gaussian_similarity(full_image, candidate)
        with torch.inference_mode():
            lpips_value = perceptual(
                full_tensor * 2 - 1, candidate_tensor * 2 - 1).item()
            dists_value = dists_model(full_tensor, candidate_tensor).item()
        base = baseline[prompt_id, 0]
        row = {
            "prompt_id": prompt_id,
            "category": artifact["category"],
            "policy": policy,
            "seconds": artifact["seconds"],
            "speedup_vs_dpm20": full_artifact["summary"]["wall_seconds"] / artifact["seconds"],
            **traditional,
            "lpips": lpips_value,
            "dists": dists_value,
            "ssim_gain_vs_branch0": traditional["ssim"] - base["ssim"],
            "psnr_gain_vs_branch0": traditional["psnr_db"] - base["psnr_db"],
            "lpips_reduction_vs_branch0": base["lpips"] - lpips_value,
            "dists_reduction_vs_branch0": base["dists"] - dists_value,
            "rgb_error_reduction_vs_branch0": (
                base["mean_absolute_rgb_error"] - traditional["mean_absolute_rgb_error"]),
        }
        rows.append(row)
        save_json(metrics_path, rows)
        print(f"metrics [{len(rows)}/{total}] {prompt_id} {policy}", flush=True)
    del perceptual, dists_model
    torch.cuda.empty_cache()
    return rows


def bootstrap_ci(values: list[float], draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(1305)
    array = np.asarray(values, dtype=np.float64)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(rows: list[dict], reference_dir: Path) -> dict:
    reference_metrics = json.loads((reference_dir / "metrics.json").read_text())
    base_rows = [row for row in reference_metrics if row["branch_id"] == 0]
    base_lpips = statistics.mean(row["lpips"] for row in base_rows)
    base_dists = statistics.mean(row["dists"] for row in base_rows)
    by_policy = {policy: [row for row in rows if row["policy"] == policy]
                 for policy in POLICIES}
    control_ssim = {row["prompt_id"]: row["ssim"] for row in by_policy["b8_s01"]}
    result = {}
    for policy, selected in by_policy.items():
        ssim_gains = [row["ssim_gain_vs_branch0"] for row in selected]
        lpips_reductions = [row["lpips_reduction_vs_branch0"] for row in selected]
        dists_reductions = [row["dists_reduction_vs_branch0"] for row in selected]
        is_control = policy == "b8_s01"
        if policy.startswith("b1_"):
            control_gain = None
            control_pass = True
        else:
            control_diffs = [row["ssim"] - control_ssim[row["prompt_id"]]
                             for row in selected]
            control_gain = statistics.mean(control_diffs)
            control_pass = control_gain > 0
        item = {
            "cases": len(selected),
            "mean_seconds": statistics.mean(row["seconds"] for row in selected),
            "mean_speedup_vs_dpm20": statistics.mean(
                row["speedup_vs_dpm20"] for row in selected),
            "mean_ssim": statistics.mean(row["ssim"] for row in selected),
            "mean_ssim_gain_vs_branch0": statistics.mean(ssim_gains),
            "ssim_gain_bootstrap_95ci": bootstrap_ci(ssim_gains),
            "positive_ssim_cases": sum(value > 0 for value in ssim_gains),
            "mean_lpips": statistics.mean(row["lpips"] for row in selected),
            "lpips_relative_reduction_vs_branch0": statistics.mean(
                lpips_reductions) / base_lpips,
            "mean_dists": statistics.mean(row["dists"] for row in selected),
            "dists_relative_reduction_vs_branch0": statistics.mean(
                dists_reductions) / base_dists,
            "mean_ssim_gain_vs_b8_s01_control": control_gain,
            "gates": {
                "ssim_positive_at_least_9_of_12": sum(value > 0 for value in ssim_gains) >= 9,
                "mean_ssim_gain_at_least_0_005": statistics.mean(ssim_gains) >= 0.005,
                "lpips_reduction_at_least_3_percent": (
                    statistics.mean(lpips_reductions) / base_lpips >= 0.03),
                "dists_reduction_at_least_2_percent": (
                    statistics.mean(dists_reductions) / base_dists >= 0.02),
                "speedup_at_least_1_50": statistics.mean(
                    row["speedup_vs_dpm20"] for row in selected) >= 1.50,
                "beats_early_control_when_applicable": control_pass,
            },
            "is_early_control": is_control,
        }
        item["passes_all_candidate_gates"] = (
            not is_control and all(item["gates"].values()))
        result[policy] = item
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    artifacts = load_interventions(args.input_dir / "data")
    if len(artifacts) != 132:
        raise RuntimeError(f"Expected 132 artifacts, found {len(artifacts)}")
    for artifact in artifacts.values():
        reference = torch.load(
            args.reference_dir / "data" /
            f"{artifact['prompt_id']}__branch_00.pt",
            map_location="cpu", weights_only=False)
        if artifact["initial_latent_sha256"] != reference["initial_latent_sha256"]:
            raise RuntimeError(f"Latent mismatch for {artifact['case']}")
    images_dir = decode_images(args.input_dir, artifacts, manifest["model_path"])
    rows = compute_metrics(args.input_dir, args.reference_dir, images_dir, artifacts)
    summary = summarize(rows, args.reference_dir)
    save_json(args.input_dir / "policy_summary.json", summary)
    with (args.input_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
