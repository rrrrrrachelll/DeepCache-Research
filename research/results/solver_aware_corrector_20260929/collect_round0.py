"""Collect on-policy DeepCache trajectories with same-latent full teachers."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.results.solver_aware_corrector_20260929.prompt_set import SEEDS, build_prompts, prompt_hash
from research.results.solver_aware_corrector_20260929.sacc import REFRESH_BALANCED8, Round0CollectorHelper


MODEL_ID = "runwayml/stable-diffusion-v1-5"


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scheduler(config):
    return DPMSolverMultistepScheduler.from_config(
        config, algorithm_type="dpmsolver++", solver_order=2,
        solver_type="midpoint", timestep_spacing="linspace", use_karras_sigmas=False)


def cases_for(split, limit=0):
    prompts = [row for row in build_prompts() if row["split"] == split]
    cases = [(row, seed) for row in prompts for seed in SEEDS[split]]
    return cases[:limit] if limit else cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(SEEDS), required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    cases = cases_for(args.split, args.limit)

    model_path = snapshot_download(MODEL_ID, local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    config = dict(pipe.scheduler.config)
    manifest_path = args.output_dir / "manifest.json"
    source_files = [Path(__file__), HERE / "sacc.py", HERE / "prompt_set.py", HERE / "PROTOCOL.md"]
    manifest = {
        "experiment": "SACC round-0 on-policy pair collection",
        "split": args.split, "limit": args.limit, "cases": len(cases),
        "model": MODEL_ID, "model_path": model_path, "steps": 20,
        "guidance_scale": 7.5, "height": 512, "width": 512,
        "refresh_steps": list(REFRESH_BALANCED8), "cache_branch_id": 0,
        "trajectory": "cached output advances scheduler; full output is same-latent shadow teacher",
        "prompt_hash": prompt_hash(), "seeds": list(SEEDS[args.split]),
        "held_out_used": False, "old_analysis_prompts_used": False,
        "gpu": torch.cuda.get_device_name(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "diffusers", "transformers", "numpy", "Pillow")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "completed": [],
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in ("experiment", "split", "limit", "cases", "prompt_hash", "refresh_steps"):
            if existing[key] != manifest[key]:
                raise RuntimeError(f"Resume manifest mismatch: {key}")
        manifest = existing
    else:
        save_json(manifest_path, manifest)

    completed = set(manifest["completed"])
    for case_index, (prompt, seed) in enumerate(cases, 1):
        case = f"{prompt['id']}_seed{seed}"
        output_path = data_dir / f"{case}.pt"
        if case in completed:
            if not output_path.exists():
                raise RuntimeError(f"Manifest records missing case {case}")
            print(f"collect [{case_index}/{len(cases)}] resume {case}", flush=True)
            continue
        if output_path.exists():
            raise RuntimeError(f"Untracked existing output {output_path}")
        pipe.scheduler = scheduler(config)
        pipe.scheduler.set_timesteps(20, device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        latent = torch.randn((1, 4, 64, 64), generator=generator,
                             device="cuda", dtype=torch.float16)
        latent_hash = hashlib.sha256(latent.cpu().numpy().tobytes()).hexdigest()
        helper = Round0CollectorHelper(pipe)
        helper.enable()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                result = pipe(prompt["prompt"], latents=latent.clone(),
                    num_inference_steps=20, guidance_scale=7.5,
                    height=512, width=512, output_type="latent")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() / 2**20
        finally:
            helper.disable()
        if len(helper.trace) != 20 or len(helper.samples) != 12:
            raise RuntimeError("Incomplete trajectory")
        artifact = {
            "case": case, "prompt_id": prompt["id"], "category": prompt["category"],
            "split": args.split, "prompt": prompt["prompt"], "seed": seed,
            "initial_latent_sha256": latent_hash, "seconds": elapsed,
            "peak_allocated_mib": peak, "trace": helper.trace,
            "samples": helper.samples,
            "final_latent": result.images.detach().half().cpu(),
        }
        torch.save(artifact, output_path)
        completed.add(case)
        manifest["completed"] = sorted(completed)
        save_json(manifest_path, manifest)
        mean_error = sum(row["guided_error"] for row in helper.trace if not row["refresh"]) / 12
        message = {"case": case, "seconds": elapsed, "peak_mib": peak,
                   "reuse_mean_guided_error": mean_error}
        print(json.dumps(message), flush=True)
        with (args.output_dir / "run.log").open("a") as handle:
            handle.write(json.dumps(message) + "\n")

    save_json(args.output_dir / "complete.json", {
        "split": args.split, "requested_cases": len(cases),
        "completed_cases": len(completed), "complete": len(completed) == len(cases),
    })


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
