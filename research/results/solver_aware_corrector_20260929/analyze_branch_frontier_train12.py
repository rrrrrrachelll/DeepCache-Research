"""Decode and analyze formal Experiment 13A outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
import lpips
from DISTS_pytorch import DISTS
from scipy.stats import spearmanr


BRANCHES = (0, 1, 3, 8, 10)
REUSE_STEPS = (1, 3, 4, 6, 8, 9, 11, 12, 14, 15, 17, 18)


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_artifacts(data_dir: Path) -> dict[tuple[str, str], dict]:
    artifacts = {}
    for path in sorted(data_dir.glob("*.pt")):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        key = (artifact["prompt_id"], artifact["summary"]["mode"])
        if key in artifacts:
            raise RuntimeError(f"Duplicate artifact {key}")
        artifacts[key] = artifact
    return artifacts


def decode_images(input_dir: Path, artifacts: dict, model_path: str) -> Path:
    images_dir = input_dir / "images"
    images_dir.mkdir(exist_ok=True)
    missing = [(key, value) for key, value in artifacts.items()
               if not (images_dir / f"{value['case']}.png").exists()]
    if not missing:
        print("decode resume: all images already exist", flush=True)
        return images_dir

    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float16,
        local_files_only=True).to("cuda")
    vae.eval()
    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
    for index, (_, artifact) in enumerate(missing, 1):
        latent = artifact["final_latent"].to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            image = vae.decode(latent / scaling).sample
        image = ((image[0].float().cpu() / 2 + 0.5).clamp(0, 1)
                 .permute(1, 2, 0).numpy())
        rgb = np.rint(image * 255).astype(np.uint8)
        Image.fromarray(rgb).save(images_dir / f"{artifact['case']}.png")
        print(f"decode [{index}/{len(missing)}] {artifact['case']}", flush=True)
    del vae
    torch.cuda.empty_cache()
    return images_dir


def gaussian_similarity(reference: np.ndarray, candidate: np.ndarray) -> dict:
    a = torch.from_numpy(reference.copy()).permute(2, 0, 1)[None].float() / 255
    b = torch.from_numpy(candidate.copy()).permute(2, 0, 1)[None].float() / 255
    x = torch.arange(11).float() - 5
    g = torch.exp(-x.square() / (2 * 1.5 ** 2))
    g /= g.sum()
    kernel = (g[:, None] * g[None, :]).expand(3, 1, 11, 11)
    blur = lambda value: F.conv2d(value, kernel, groups=3)
    ma, mb = blur(a), blur(b)
    va, vb = blur(a * a) - ma * ma, blur(b * b) - mb * mb
    cov = blur(a * b) - ma * mb
    ssim = (((2 * ma * mb + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) /
            ((ma * ma + mb * mb + 0.01 ** 2) *
             (va + vb + 0.03 ** 2))).mean().item()
    mse = (a - b).square().mean().item()
    return {
        "ssim": ssim,
        "psnr_db": -10 * math.log10(mse) if mse else None,
        "mean_absolute_rgb_error": (a - b).abs().mean().item(),
    }


def image_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float() / 255


def compute_metrics(input_dir: Path, images_dir: Path, artifacts: dict) -> list[dict]:
    metrics_path = input_dir / "metrics.json"
    if metrics_path.exists():
        rows = json.loads(metrics_path.read_text())
        if len(rows) == 60:
            print("metrics resume: 60 rows", flush=True)
            return rows

    perceptual = lpips.LPIPS(net="alex").eval().to("cuda")
    dists = DISTS().eval().to("cuda")
    prompt_ids = sorted({prompt_id for prompt_id, _ in artifacts})
    rows = []
    total = len(prompt_ids) * len(BRANCHES)
    for prompt_id in prompt_ids:
        reference_artifact = artifacts[prompt_id, "full"]
        reference = np.asarray(Image.open(
            images_dir / f"{reference_artifact['case']}.png").convert("RGB"))
        reference_tensor = image_tensor(reference).to("cuda")
        for branch_id in BRANCHES:
            mode = f"branch_{branch_id:02d}"
            artifact = artifacts[prompt_id, mode]
            candidate = np.asarray(Image.open(
                images_dir / f"{artifact['case']}.png").convert("RGB"))
            candidate_tensor = image_tensor(candidate).to("cuda")
            traditional = gaussian_similarity(reference, candidate)
            with torch.inference_mode():
                lpips_value = perceptual(
                    reference_tensor * 2 - 1,
                    candidate_tensor * 2 - 1).item()
                dists_value = dists(reference_tensor, candidate_tensor).item()
            row = {
                "prompt_id": prompt_id,
                "category": artifact["category"],
                "mode": mode,
                "branch_id": branch_id,
                **traditional,
                "lpips": lpips_value,
                "dists": dists_value,
                "mean_reuse_guided_error": artifact["summary"]["mean_reuse_guided_error"],
                "deployment_ms": artifact["summary"]["deployment_ms"],
                "estimated_speedup": artifact["summary"]["estimated_speedup"],
            }
            rows.append(row)
            save_json(metrics_path, rows)
            print(f"metrics [{len(rows)}/{total}] {prompt_id} {mode}", flush=True)
    del perceptual, dists
    torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict]) -> dict:
    result = {}
    for branch_id in BRANCHES:
        selected = [row for row in rows if row["branch_id"] == branch_id]
        result[f"branch_{branch_id:02d}"] = {
            "cases": len(selected),
            "mean_ssim": statistics.mean(row["ssim"] for row in selected),
            "mean_psnr_db": statistics.mean(row["psnr_db"] for row in selected),
            "mean_lpips": statistics.mean(row["lpips"] for row in selected),
            "mean_dists": statistics.mean(row["dists"] for row in selected),
            "mean_absolute_rgb_error": statistics.mean(
                row["mean_absolute_rgb_error"] for row in selected),
            "mean_reuse_guided_error": statistics.mean(
                row["mean_reuse_guided_error"] for row in selected),
            "mean_deployment_ms": statistics.mean(row["deployment_ms"] for row in selected),
            "mean_estimated_speedup": statistics.mean(row["estimated_speedup"] for row in selected),
        }
    return result


def step_statistics(artifacts: dict) -> list[dict]:
    prompt_ids = sorted({prompt_id for prompt_id, _ in artifacts})
    output = []
    for branch_id in (1, 3, 8, 10):
        per_prompt_gains = {}
        for prompt_id in prompt_ids:
            zero = artifacts[prompt_id, "branch_00"]
            candidate = artifacts[prompt_id, f"branch_{branch_id:02d}"]
            per_prompt_gains[prompt_id] = {
                step: zero["trace"][step]["guided_error"] -
                candidate["trace"][step]["guided_error"]
                for step in REUSE_STEPS}
        top4 = {prompt_id: {step for _, step in sorted(
            ((gain, step) for step, gain in gains.items()), reverse=True)[:4]}
            for prompt_id, gains in per_prompt_gains.items()}
        for step in REUSE_STEPS:
            gains, costs = [], []
            for prompt_id in prompt_ids:
                zero = artifacts[prompt_id, "branch_00"]
                candidate = artifacts[prompt_id, f"branch_{branch_id:02d}"]
                gains.append(per_prompt_gains[prompt_id][step])
                costs.append(candidate["trace"][step]["deployment_ms"] -
                             zero["trace"][step]["deployment_ms"])
            mean_gain, mean_cost = statistics.mean(gains), statistics.mean(costs)
            output.append({
                "branch_id": branch_id,
                "step": step,
                "mean_error_reduction": mean_gain,
                "median_error_reduction": statistics.median(gains),
                "positive_cases": sum(value > 0 for value in gains),
                "top4_cases": sum(step in top4[prompt_id] for prompt_id in prompt_ids),
                "mean_incremental_ms": mean_cost,
                "error_reduction_per_second": mean_gain / mean_cost * 1000,
            })
    return output


def correlations(rows: list[dict]) -> dict:
    noise = [row["mean_reuse_guided_error"] for row in rows]
    result = {}
    for metric in ("ssim", "psnr_db", "lpips", "dists", "mean_absolute_rgb_error"):
        values = [row[metric] for row in rows]
        statistic = spearmanr(noise, values)
        result[metric] = {
            "spearman_rho_with_noise_error": float(statistic.statistic),
            "pvalue": float(statistic.pvalue),
            "cases": len(rows),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    artifacts = load_artifacts(args.input_dir / "data")
    if len(artifacts) != 72:
        raise RuntimeError(f"Expected 72 artifacts, found {len(artifacts)}")
    prompt_ids = sorted({key[0] for key in artifacts})
    if len(prompt_ids) != 12:
        raise RuntimeError("Expected 12 prompts")
    for prompt_id in prompt_ids:
        hashes = {artifacts[prompt_id, mode]["initial_latent_sha256"]
                  for mode in manifest["modes"]}
        if len(hashes) != 1:
            raise RuntimeError(f"Latent pairing failed for {prompt_id}")

    images_dir = decode_images(args.input_dir, artifacts, manifest["model_path"])
    rows = compute_metrics(args.input_dir, images_dir, artifacts)
    summary = summarize(rows)
    steps = step_statistics(artifacts)
    corr = correlations(rows)
    save_json(args.input_dir / "metric_summary.json", summary)
    save_json(args.input_dir / "step_statistics.json", steps)
    save_json(args.input_dir / "proxy_correlations.json", corr)
    with (args.input_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": summary, "correlations": corr}, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
