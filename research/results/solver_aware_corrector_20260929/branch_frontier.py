"""Experiment 13A: probe DeepCache's native cache-depth frontier."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DeepCache.extension.deepcache import DeepCacheSDHelper
from research.results.solver_aware_corrector_20260929.prompt_set import build_prompts, prompt_hash
from research.results.solver_aware_corrector_20260929.sacc import (
    REFRESH_BALANCED8, guided_noise, relative_l1, tensor_output)


MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_BRANCHES = tuple(range(12))


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_scheduler(config):
    return DPMSolverMultistepScheduler.from_config(
        config, algorithm_type="dpmsolver++", solver_order=2,
        solver_type="midpoint", timestep_spacing="linspace",
        use_karras_sigmas=False)


def branch_location(branch_id: int) -> dict[str, int]:
    if branch_id < 0:
        raise ValueError("branch_id must be non-negative")
    return {"block_id": branch_id // 3, "layer_id": branch_id % 3}


def selected_train_prompt() -> dict:
    """Use the first train prompt in the deterministic prompt table."""
    return next(row for row in build_prompts() if row["split"] == "train")


class BranchFrontierHelper(DeepCacheSDHelper):
    """Run a cached trajectory while measuring same-latent full teachers."""

    def __init__(self, pipe, branch_id: int,
                 refresh_steps=REFRESH_BALANCED8, guidance_scale: float = 7.5):
        super().__init__(pipe)
        self.branch_id = int(branch_id)
        self.set_params(cache_interval=1, cache_branch_id=self.branch_id)
        self.refresh_steps = tuple(int(value) for value in refresh_steps)
        self.refresh_set = frozenset(self.refresh_steps)
        self.guidance_scale = float(guidance_scale)
        self.mode = "idle"
        self.trace: list[dict] = []
        self.last_refresh_step = 0
        self.step_recomputed: set[str] = set()
        self.step_skipped: set[str] = set()

    def enable(self):
        self.timesteps = tuple(int(t) for t in self.pipe.scheduler.timesteps)
        if len(self.timesteps) != 20 or len(set(self.timesteps)) != 20:
            raise ValueError("Expected a fixed schedule of 20 unique timesteps")
        if not self.refresh_steps or self.refresh_steps[0] != 0:
            raise ValueError("The first call must refresh an empty cache")
        super().enable()

    def is_refresh(self) -> bool:
        return self.cur_timestep in self.refresh_set

    def is_skip_step(self, block_i, layer_i, blocktype="down") -> bool:
        if self.is_refresh():
            return False
        cache_block = self.params["cache_block_id"]
        cache_layer = self.params["cache_layer_id"]
        if block_i > cache_block or blocktype == "mid":
            return True
        if block_i < cache_block:
            return False
        return layer_i >= cache_layer if blocktype == "down" else layer_i > cache_layer

    @staticmethod
    def _label(key) -> str:
        return ":".join(str(part) for part in key)

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype="down"):
        key = (blocktype, block_name, block_i, layer_i)
        original = block.forward
        self.function_dict[key] = original

        def forward(*args, **kwargs):
            if self.mode == "full":
                result = original(*args, **kwargs)
                if self.is_refresh():
                    self.cached_output[key] = result
                return result
            if self.mode == "cache":
                if self.is_skip_step(block_i, layer_i, blocktype):
                    self.step_skipped.add(self._label(key))
                    if key not in self.cached_output:
                        raise RuntimeError(f"Missing cached output for {key}")
                    return self.cached_output[key]
                self.step_recomputed.add(self._label(key))
                result = original(*args, **kwargs)
                self.cached_output[key] = result
                return result
            return original(*args, **kwargs)

        block.forward = forward

    @staticmethod
    def _time_call(function, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        end.synchronize()
        return result, float(start.elapsed_time(end))

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
            cache_identity_before = {key: id(value) for key, value in self.cached_output.items()}
            scheduler_state_before = (
                self.pipe.scheduler._step_index,
                tuple(id(value) if value is not None else None
                      for value in self.pipe.scheduler.model_outputs),
            )

            self.mode = "full"
            full_result, full_ms = self._time_call(original, *args, **kwargs)
            full = guided_noise(tensor_output(full_result), self.guidance_scale)
            self.step_recomputed = set()
            self.step_skipped = set()

            if refresh:
                cached_result = full_result
                cached = full
                cache_ms = full_ms
                self.last_refresh_step = step
            else:
                cache_identity_after = {key: id(value) for key, value in self.cached_output.items()}
                if cache_identity_before != cache_identity_after:
                    raise RuntimeError("Shadow full call mutated the DeepCache state")
                scheduler_state_after = (
                    self.pipe.scheduler._step_index,
                    tuple(id(value) if value is not None else None
                          for value in self.pipe.scheduler.model_outputs),
                )
                if scheduler_state_before != scheduler_state_after:
                    raise RuntimeError("Shadow full call mutated the scheduler state")
                self.mode = "cache"
                cached_result, cache_ms = self._time_call(original, *args, **kwargs)
                cached = guided_noise(tensor_output(cached_result), self.guidance_scale)

            row = {
                "step": step,
                "timestep": timestep,
                "refresh": refresh,
                "age": step - self.last_refresh_step,
                "branch_id": self.branch_id,
                **branch_location(self.branch_id),
                "guided_error": relative_l1(full, cached),
                "full_ms": full_ms,
                "deployment_ms": cache_ms,
                "recomputed_modules": sorted(self.step_recomputed),
                "skipped_modules": sorted(self.step_skipped),
            }
            if refresh and row["guided_error"] != 0.0:
                raise RuntimeError("Refresh step must have zero cache error")
            self.trace.append(row)
            self.mode = "idle"
            return cached_result

        self.pipe.unet.forward = forward


def parse_branches(value: str) -> tuple[int, ...]:
    result = tuple(int(part) for part in value.split(",") if part.strip())
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError("branches must be unique non-negative integers")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branches", type=parse_branches,
                        default=DEFAULT_BRANCHES)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    prompt = selected_train_prompt()
    model_path = snapshot_download(MODEL_ID, local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    scheduler_config = dict(pipe.scheduler.config)

    source_files = [Path(__file__), HERE / "sacc.py", HERE / "prompt_set.py"]
    manifest = {
        "experiment": "Experiment 13A branch legality smoke",
        "model": MODEL_ID,
        "model_path": model_path,
        "steps": 20,
        "guidance_scale": 7.5,
        "refresh_steps": list(REFRESH_BALANCED8),
        "branches": list(args.branches),
        "seed": args.seed,
        "prompt_id": prompt["id"],
        "prompt": prompt["prompt"],
        "prompt_hash": prompt_hash(),
        "trajectory": "cached output advances scheduler; full output is same-latent shadow",
        "timing": "sum of refresh full calls and reuse cached calls; shadow excluded",
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
        existing = json.loads(manifest_path.read_text())
        for key in ("experiment", "steps", "refresh_steps", "branches", "seed",
                    "prompt_id", "prompt_hash"):
            if existing[key] != manifest[key]:
                raise RuntimeError(f"Resume manifest mismatch: {key}")
        manifest = existing
    else:
        save_json(manifest_path, manifest)

    completed = set(manifest["completed"])
    summary = []
    for index, branch_id in enumerate(args.branches, 1):
        case = f"branch_{branch_id:02d}"
        output_path = data_dir / f"{case}.pt"
        if case in completed:
            artifact = torch.load(output_path, map_location="cpu", weights_only=False)
            summary.append(artifact["summary"])
            print(f"probe [{index}/{len(args.branches)}] resume {case}", flush=True)
            continue
        if output_path.exists():
            raise RuntimeError(f"Untracked existing output {output_path}")

        pipe.scheduler = make_scheduler(scheduler_config)
        pipe.scheduler.set_timesteps(20, device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        initial_latent = torch.randn((1, 4, 64, 64), generator=generator,
                                     device="cuda", dtype=torch.float16)
        latent_hash = hashlib.sha256(initial_latent.cpu().numpy().tobytes()).hexdigest()
        helper = BranchFrontierHelper(pipe, branch_id)
        helper.enable()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()
        try:
            with torch.inference_mode():
                result = pipe(
                    prompt["prompt"], latents=initial_latent.clone(),
                    num_inference_steps=20, guidance_scale=7.5,
                    height=512, width=512, output_type="latent")
            torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - wall_start
            peak_mib = torch.cuda.max_memory_allocated() / 2**20
        finally:
            helper.disable()

        if len(helper.trace) != 20:
            raise RuntimeError(f"Incomplete trajectory: {len(helper.trace)} calls")
        refresh_rows = [row for row in helper.trace if row["refresh"]]
        reuse_rows = [row for row in helper.trace if not row["refresh"]]
        if len(refresh_rows) != 8 or len(reuse_rows) != 12:
            raise RuntimeError("Expected exactly 8 refresh and 12 reuse calls")
        if any(row["guided_error"] != 0.0 for row in refresh_rows):
            raise RuntimeError("Non-zero refresh error")
        if not torch.isfinite(result.images).all():
            raise RuntimeError("Non-finite final latent")
        deployment_ms = sum(row["deployment_ms"] for row in helper.trace)
        summary_row = {
            "branch_id": branch_id,
            **branch_location(branch_id),
            "legal": True,
            "refresh_calls": len(refresh_rows),
            "reuse_calls": len(reuse_rows),
            "mean_reuse_guided_error": sum(row["guided_error"] for row in reuse_rows) / len(reuse_rows),
            "max_reuse_guided_error": max(row["guided_error"] for row in reuse_rows),
            "deployment_ms": deployment_ms,
            "mean_reuse_ms": sum(row["deployment_ms"] for row in reuse_rows) / len(reuse_rows),
            "mean_full_ms": sum(row["full_ms"] for row in helper.trace) / len(helper.trace),
            "wall_seconds_with_shadow": wall_seconds,
            "peak_allocated_mib": peak_mib,
            "recomputed_module_count": len(set().union(
                *(set(row["recomputed_modules"]) for row in reuse_rows))),
            "skipped_module_count": len(set().union(
                *(set(row["skipped_modules"]) for row in reuse_rows))),
        }
        artifact = {
            "summary": summary_row,
            "initial_latent_sha256": latent_hash,
            "trace": helper.trace,
            "final_latent": result.images.detach().half().cpu(),
        }
        torch.save(artifact, output_path)
        summary.append(summary_row)
        completed.add(case)
        manifest["completed"] = sorted(completed)
        save_json(manifest_path, manifest)
        save_json(args.output_dir / "summary.json", sorted(summary, key=lambda row: row["branch_id"]))
        print(json.dumps(summary_row), flush=True)

    save_json(args.output_dir / "complete.json", {
        "requested_branches": list(args.branches),
        "completed_branches": sorted(int(case.split("_")[1]) for case in completed),
        "complete": all(f"branch_{branch_id:02d}" in completed for branch_id in args.branches),
    })


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
