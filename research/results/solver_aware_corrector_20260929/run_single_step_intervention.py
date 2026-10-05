"""Experiment 13A.5: direct final-image effects of one-step branch switches."""
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

from DeepCache.extension.deepcache import DeepCacheSDHelper
from research.results.solver_aware_corrector_20260929.branch_frontier import (
    MODEL_ID, branch_location, make_scheduler, save_json)
from research.results.solver_aware_corrector_20260929.prompt_set import prompt_hash
from research.results.solver_aware_corrector_20260929.run_branch_frontier_train12 import (
    category_prompts, make_initial_latent)
from research.results.solver_aware_corrector_20260929.sacc import REFRESH_BALANCED8


POLICIES = {
    "b1_s15": {15: 1},
    "b1_s18": {18: 1},
    "b8_s01": {1: 8},
    "b8_s09": {9: 8},
    "b8_s12": {12: 8},
    "b8_s15": {15: 8},
    "b8_s18": {18: 8},
    "b10_s09": {9: 10},
    "b10_s12": {12: 10},
    "b10_s15": {15: 10},
    "b10_s18": {18: 10},
}


class StageBranchHelper(DeepCacheSDHelper):
    """Use branch 0 except for explicitly selected reuse steps."""

    def __init__(self, pipe, branch_by_step: dict[int, int],
                 refresh_steps=REFRESH_BALANCED8):
        super().__init__(pipe)
        self.branch_by_step = dict(branch_by_step)
        self.refresh_steps = tuple(int(step) for step in refresh_steps)
        self.refresh_set = frozenset(self.refresh_steps)
        if any(step in self.refresh_set for step in self.branch_by_step):
            raise ValueError("An intervention step cannot be a full refresh")
        if any(step < 0 or step >= 20 for step in self.branch_by_step):
            raise ValueError("Intervention steps must be in [0, 19]")
        if any(branch < 0 or branch > 11 for branch in self.branch_by_step.values()):
            raise ValueError("Native SD1.5 branch IDs must be in [0, 11]")
        self.set_params(cache_interval=1, cache_branch_id=0)
        self.trace: list[dict] = []
        self.current_branch = 0

    def enable(self):
        self.timesteps = tuple(int(timestep) for timestep in self.pipe.scheduler.timesteps)
        if len(self.timesteps) != 20 or len(set(self.timesteps)) != 20:
            raise ValueError("Expected a fixed 20-step schedule")
        super().enable()

    def is_refresh(self) -> bool:
        return self.cur_timestep in self.refresh_set

    def is_skip_step(self, block_i, layer_i, blocktype="down") -> bool:
        if self.is_refresh():
            return False
        location = branch_location(self.current_branch)
        cache_block = location["block_id"]
        cache_layer = location["layer_id"]
        if block_i > cache_block or blocktype == "mid":
            return True
        if block_i < cache_block:
            return False
        return layer_i >= cache_layer if blocktype == "down" else layer_i > cache_layer

    def wrap_unet_forward(self):
        original = self.pipe.unet.forward
        self.function_dict["unet_forward"] = original

        def forward(*args, **kwargs):
            step = len(self.trace)
            timestep_arg = args[1] if len(args) > 1 else kwargs["timestep"]
            timestep = int(timestep_arg.flatten()[0]) if torch.is_tensor(timestep_arg) else int(timestep_arg)
            if step >= 20 or timestep != self.timesteps[step]:
                raise RuntimeError(f"Unexpected timestep {timestep} at call {step}")
            self.cur_timestep = step
            refresh = self.is_refresh()
            self.current_branch = 0 if refresh else self.branch_by_step.get(step, 0)
            result = original(*args, **kwargs)
            self.trace.append({
                "step": step,
                "timestep": timestep,
                "refresh": refresh,
                "branch_id": self.current_branch,
                **branch_location(self.current_branch),
            })
            return result

        self.pipe.unet.forward = forward


def run_policy(pipe, scheduler_config, prompt: str, latent: torch.Tensor,
               branch_by_step: dict[int, int]):
    pipe.scheduler = make_scheduler(scheduler_config)
    pipe.scheduler.set_timesteps(20, device="cuda")
    helper = StageBranchHelper(pipe, branch_by_step)
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
        raise RuntimeError("Incomplete intervention trajectory")
    if sum(row["refresh"] for row in helper.trace) != 8:
        raise RuntimeError("Expected exactly eight full refreshes")
    observed = {row["step"]: row["branch_id"] for row in helper.trace
                if not row["refresh"] and row["branch_id"] != 0}
    if observed != branch_by_step:
        raise RuntimeError(f"Branch intervention mismatch: {observed} != {branch_by_step}")
    if not torch.isfinite(result.images).all():
        raise RuntimeError("Non-finite final latent")
    return result.images.detach().half().cpu(), helper.trace, seconds, peak_mib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    prompts = category_prompts(1)

    reference_manifest = json.loads((args.reference_dir / "manifest.json").read_text())
    if reference_manifest["prompt_ids"] != [row["id"] for row in prompts]:
        raise RuntimeError("Reference prompt set mismatch")
    if reference_manifest["seed"] != args.seed:
        raise RuntimeError("Reference seed mismatch")

    model_path = snapshot_download(MODEL_ID, local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    scheduler_config = dict(pipe.scheduler.config)

    source_files = [Path(__file__), HERE / "branch_frontier.py",
                    HERE / "run_branch_frontier_train12.py", HERE / "prompt_set.py"]
    policy_names = list(POLICIES)
    manifest = {
        "experiment": "Experiment 13A.5 single-step closed-loop intervention",
        "model": MODEL_ID,
        "model_path": model_path,
        "steps": 20,
        "guidance_scale": 7.5,
        "refresh_steps": list(REFRESH_BALANCED8),
        "base_branch": 0,
        "policies": POLICIES,
        "seed": args.seed,
        "prompt_ids": [row["id"] for row in prompts],
        "prompt_hash": prompt_hash(),
        "reference_dir": str(args.reference_dir.resolve()),
        "execution_order": "Latin rotation of intervention policies by category index",
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
        for key in ("experiment", "steps", "refresh_steps", "base_branch",
                    "policies", "seed", "prompt_ids", "prompt_hash"):
            if old[key] != manifest[key]:
                raise RuntimeError(f"Resume manifest mismatch: {key}")
        manifest = old
    else:
        save_json(manifest_path, manifest)

    pipe.scheduler = make_scheduler(scheduler_config)
    with torch.inference_mode():
        pipe(prompts[0]["prompt"], latents=make_initial_latent(args.seed),
             num_inference_steps=2, guidance_scale=7.5,
             height=512, width=512, output_type="latent")
    torch.cuda.synchronize()

    integrity_path = args.output_dir / "integrity.json"
    if not integrity_path.exists():
        reference = torch.load(
            args.reference_dir / "data" / f"{prompts[0]['id']}__branch_00.pt",
            map_location="cpu", weights_only=False)
        control, trace, seconds, peak_mib = run_policy(
            pipe, scheduler_config, prompts[0]["prompt"],
            make_initial_latent(args.seed), {})
        max_abs = float((control.float() - reference["final_latent"].float()).abs().max())
        save_json(integrity_path, {
            "prompt_id": prompts[0]["id"],
            "branch0_exact_match": bool(torch.equal(control, reference["final_latent"])),
            "max_abs_difference": max_abs,
            "calls": len(trace), "seconds": seconds, "peak_allocated_mib": peak_mib,
        })
        if max_abs != 0.0:
            raise RuntimeError("Dynamic helper does not reproduce static branch 0")

    completed = set(manifest["completed"])
    total = len(prompts) * len(policy_names)
    summary_path = args.output_dir / "summary.jsonl"
    ordinal = 0
    for prompt_number, prompt in enumerate(prompts):
        rotated = (policy_names[prompt_number % len(policy_names):] +
                   policy_names[:prompt_number % len(policy_names)])
        for policy_name in rotated:
            ordinal += 1
            case = f"{prompt['id']}__{policy_name}"
            output_path = data_dir / f"{case}.pt"
            if case in completed:
                if not output_path.exists():
                    raise RuntimeError(f"Manifest records missing case {case}")
                print(f"intervention [{ordinal}/{total}] resume {case}", flush=True)
                continue
            if output_path.exists():
                raise RuntimeError(f"Untracked existing output {output_path}")

            latent = make_initial_latent(args.seed)
            latent_hash = hashlib.sha256(latent.cpu().numpy().tobytes()).hexdigest()
            final_latent, trace, seconds, peak_mib = run_policy(
                pipe, scheduler_config, prompt["prompt"], latent,
                POLICIES[policy_name])
            artifact = {
                "case": case,
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "prompt": prompt["prompt"],
                "seed": args.seed,
                "policy": policy_name,
                "branch_by_step": POLICIES[policy_name],
                "initial_latent_sha256": latent_hash,
                "seconds": seconds,
                "peak_allocated_mib": peak_mib,
                "trace": trace,
                "final_latent": final_latent,
            }
            torch.save(artifact, output_path)
            completed.add(case)
            manifest["completed"] = sorted(completed)
            save_json(manifest_path, manifest)
            row = {"ordinal": ordinal, "total": total, "case": case,
                   "category": prompt["category"], "policy": policy_name,
                   "seconds": seconds, "peak_allocated_mib": peak_mib}
            with summary_path.open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)

    save_json(args.output_dir / "complete.json", {
        "requested_cases": total,
        "completed_cases": len(completed),
        "complete": len(completed) == total,
    })


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
