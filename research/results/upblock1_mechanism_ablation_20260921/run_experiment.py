"""Run the up_block_1 activation-versus-suffix mechanism ablation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mechanism_oracle import UpBlock1MechanismOracle

MODEL_ID = "runwayml/stable-diffusion-v1-5"
REFRESH_STEPS = [0, 2, 5, 7, 10, 13, 16, 19]
PROBE_STEPS = [4, 9, 12, 15, 18]
SEEDS = [0, 1]
PROMPTS = [
    {"id": "portrait", "text": "A cinematic portrait photo of an elderly sailor, detailed weathered face, dramatic side lighting, 85mm lens"},
    {"id": "animal", "text": "A red fox sitting in a snowy forest at sunrise, highly detailed fur, soft golden light"},
    {"id": "interior", "text": "A cozy reading room with wooden bookshelves, a fireplace, warm lamps, realistic interior photography"},
    {"id": "architecture", "text": "A futuristic glass museum beside a calm lake, symmetrical architecture, blue hour, ultra realistic"},
    {"id": "texture", "text": "Macro photograph of colorful mineral crystals, intricate texture, sharp focus, studio lighting"},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def scheduler(config):
    return DPMSolverMultistepScheduler.from_config(
        config, algorithm_type="dpmsolver++", solver_order=2
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [(prompt, seed) for prompt in PROMPTS for seed in SEEDS]
    if args.limit is not None:
        cases = cases[: args.limit]

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    ).to("cuda")
    config = dict(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    manifest = {
        "model": MODEL_ID,
        "scheduler": "DPM-Solver++ order 2",
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "cache_branch_id": 0,
        "refresh_schedule": REFRESH_STEPS,
        "probe_steps": PROBE_STEPS,
        "target": "logical up_block_1 (pipe.unet.up_blocks[2])",
        "split": "same five development prompts as block feature oracle; held-out excluded",
        "cases": [],
    }

    for index, (prompt, seed) in enumerate(cases, 1):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        latent = torch.randn((1, 4, 64, 64), generator=generator, device="cuda", dtype=torch.float16)
        kwargs = dict(
            prompt=prompt["text"], latents=latent.clone(), num_inference_steps=20,
            guidance_scale=7.5, height=512, width=512, output_type="np",
        )
        pristine = None
        if index == 1:
            pipe.scheduler = scheduler(config)
            with torch.inference_mode():
                pristine = pipe(**kwargs).images.copy()

        pipe.scheduler = scheduler(config)
        pipe.scheduler.set_timesteps(20, device="cuda")
        helper = UpBlock1MechanismOracle(pipe, REFRESH_STEPS, PROBE_STEPS)
        helper.enable()
        started = time.perf_counter()
        with torch.inference_mode():
            result = pipe(**{**kwargs, "latents": latent.clone()})
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        helper.disable()
        if pristine is not None and not np.array_equal(pristine, result.images):
            raise RuntimeError("Oracle instrumentation changed the uncached trajectory")

        case_id = f"{prompt['id']}_seed{seed}"
        trace_name = f"{case_id}.json"
        (args.output_dir / trace_name).write_text(json.dumps(helper.trace, indent=2), encoding="utf-8")
        Image.fromarray((result.images[0] * 255).round().clip(0, 255).astype(np.uint8)).save(
            args.output_dir / f"{case_id}.png"
        )
        manifest["cases"].append({
            "case_id": case_id, "prompt_id": prompt["id"], "prompt": prompt["text"],
            "seed": seed, "elapsed_seconds": elapsed,
            "pristine_exact_match": pristine is not None, "trace": trace_name,
        })
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[{index}/{len(cases)}] {case_id}: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
