"""Formal Experiment 13A: cross-category static branch frontier."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.results.solver_aware_corrector_20260929.branch_frontier import (
    BranchFrontierHelper, MODEL_ID, branch_location, make_scheduler,
    parse_branches, save_json)
from research.results.solver_aware_corrector_20260929.prompt_set import (
    build_prompts, prompt_hash)
from research.results.solver_aware_corrector_20260929.sacc import REFRESH_BALANCED8


DEFAULT_BRANCHES = (0, 1, 3, 8, 10)


def category_prompts(prompt_index: int = 1) -> list[dict]:
    suffix = f"_{prompt_index:02d}"
    prompts = [row for row in build_prompts()
               if row["split"] == "train" and row["id"].endswith(suffix)]
    categories = {row["category"] for row in prompts}
    if len(prompts) != 12 or len(categories) != 12:
        raise RuntimeError("Expected one train prompt from each of 12 categories")
    return prompts


def make_initial_latent(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((1, 4, 64, 64), generator=generator,
                       device="cuda", dtype=torch.float16)


def run_full(pipe, scheduler_config, prompt: str, latent: torch.Tensor):
    pipe.scheduler = make_scheduler(scheduler_config)
    pipe.scheduler.set_timesteps(20, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(prompt, latents=latent.clone(), num_inference_steps=20,
                      guidance_scale=7.5, height=512, width=512,
                      output_type="latent")
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return result.images.detach().half().cpu(), seconds, (
        torch.cuda.max_memory_allocated() / 2**20)


def run_branch(pipe, scheduler_config, prompt: str, latent: torch.Tensor,
               branch_id: int):
    pipe.scheduler = make_scheduler(scheduler_config)
    pipe.scheduler.set_timesteps(20, device="cuda")
    helper = BranchFrontierHelper(pipe, branch_id)
    helper.enable()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            result = pipe(prompt, latents=latent.clone(), num_inference_steps=20,
                          guidance_scale=7.5, height=512, width=512,
                          output_type="latent")
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        peak_mib = torch.cuda.max_memory_allocated() / 2**20
    finally:
        helper.disable()

    if len(helper.trace) != 20:
        raise RuntimeError(f"Incomplete branch {branch_id} trajectory")
    refresh_rows = [row for row in helper.trace if row["refresh"]]
    reuse_rows = [row for row in helper.trace if not row["refresh"]]
    if len(refresh_rows) != 8 or len(reuse_rows) != 12:
        raise RuntimeError("Expected exactly 8 refresh and 12 reuse calls")
    if any(row["guided_error"] != 0.0 for row in refresh_rows):
        raise RuntimeError("Refresh error must be zero")
    if not torch.isfinite(result.images).all():
        raise RuntimeError("Non-finite final latent")

    summary = {
        "branch_id": branch_id,
        **branch_location(branch_id),
        "mean_reuse_guided_error": sum(row["guided_error"] for row in reuse_rows) / 12,
        "max_reuse_guided_error": max(row["guided_error"] for row in reuse_rows),
        "deployment_ms": sum(row["deployment_ms"] for row in helper.trace),
        "mean_reuse_ms": sum(row["deployment_ms"] for row in reuse_rows) / 12,
        "mean_full_ms": sum(row["full_ms"] for row in helper.trace) / 20,
        "estimated_speedup": (
            sum(row["full_ms"] for row in helper.trace) /
            sum(row["deployment_ms"] for row in helper.trace)),
        "wall_seconds_with_shadow": seconds,
        "peak_allocated_mib": peak_mib,
    }
    return result.images.detach().half().cpu(), helper.trace, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branches", type=parse_branches, default=DEFAULT_BRANCHES)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--prompt-index", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    prompts = category_prompts(args.prompt_index)

    model_path = snapshot_download(MODEL_ID, local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    scheduler_config = dict(pipe.scheduler.config)

    source_files = [Path(__file__), HERE / "branch_frontier.py",
                    HERE / "sacc.py", HERE / "prompt_set.py"]
    modes = ["full"] + [f"branch_{branch_id:02d}" for branch_id in args.branches]
    manifest = {
        "experiment": "Experiment 13A formal cross-category branch frontier",
        "model": MODEL_ID,
        "model_path": model_path,
        "steps": 20,
        "guidance_scale": 7.5,
        "refresh_steps": list(REFRESH_BALANCED8),
        "branches": list(args.branches),
        "seed": args.seed,
        "prompt_index": args.prompt_index,
        "prompt_ids": [row["id"] for row in prompts],
        "prompt_hash": prompt_hash(),
        "modes": modes,
        "execution_order": "Latin rotation of full and branch modes by category index",
        "trajectory": "cached output advances scheduler; same-latent full shadow",
        "gpu": torch.cuda.get_device_name(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "diffusers", "transformers", "numpy", "Pillow")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
        "completed": [],
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ("experiment", "steps", "refresh_steps", "branches", "seed",
                    "prompt_index", "prompt_ids", "prompt_hash", "modes"):
            if old[key] != manifest[key]:
                raise RuntimeError(f"Resume manifest mismatch: {key}")
        manifest = old
    else:
        save_json(manifest_path, manifest)

    # A short warm-up initializes CUDA kernels. It is never saved or timed.
    pipe.scheduler = make_scheduler(scheduler_config)
    warm_latent = make_initial_latent(args.seed)
    with torch.inference_mode():
        pipe(prompts[0]["prompt"], latents=warm_latent, num_inference_steps=2,
             guidance_scale=7.5, height=512, width=512, output_type="latent")
    torch.cuda.synchronize()

    completed = set(manifest["completed"])
    summary_path = args.output_dir / "summary.jsonl"
    total = len(prompts) * len(modes)
    ordinal = 0
    for prompt_number, prompt in enumerate(prompts):
        rotated_modes = modes[prompt_number % len(modes):] + modes[:prompt_number % len(modes)]
        for mode in rotated_modes:
            ordinal += 1
            case = f"{prompt['id']}__{mode}"
            output_path = data_dir / f"{case}.pt"
            if case in completed:
                if not output_path.exists():
                    raise RuntimeError(f"Manifest records missing case {case}")
                print(f"formal [{ordinal}/{total}] resume {case}", flush=True)
                continue
            if output_path.exists():
                raise RuntimeError(f"Untracked existing output {output_path}")

            latent = make_initial_latent(args.seed)
            latent_hash = hashlib.sha256(latent.cpu().numpy().tobytes()).hexdigest()
            if mode == "full":
                final_latent, seconds, peak_mib = run_full(
                    pipe, scheduler_config, prompt["prompt"], latent)
                trace = []
                result_summary = {
                    "mode": mode, "wall_seconds": seconds,
                    "peak_allocated_mib": peak_mib,
                }
            else:
                branch_id = int(mode.split("_")[1])
                final_latent, trace, branch_summary = run_branch(
                    pipe, scheduler_config, prompt["prompt"], latent, branch_id)
                result_summary = {"mode": mode, **branch_summary}

            artifact = {
                "case": case,
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "prompt": prompt["prompt"],
                "seed": args.seed,
                "initial_latent_sha256": latent_hash,
                "summary": result_summary,
                "trace": trace,
                "final_latent": final_latent,
            }
            torch.save(artifact, output_path)
            completed.add(case)
            manifest["completed"] = sorted(completed)
            save_json(manifest_path, manifest)
            log_row = {
                "ordinal": ordinal, "total": total, "case": case,
                "category": prompt["category"], **result_summary,
            }
            with summary_path.open("a") as handle:
                handle.write(json.dumps(log_row, allow_nan=False) + "\n")
            print(json.dumps(log_row), flush=True)

    save_json(args.output_dir / "complete.json", {
        "requested_cases": total,
        "completed_cases": len(completed),
        "complete": len(completed) == total,
    })


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
