"""Run the development-set block feature and activation-replacement oracle."""

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

from block_replacement_oracle import BlockReplacementOracleHelper

MODEL_ID = "runwayml/stable-diffusion-v1-5"
REFRESH_STEPS = [0, 2, 5, 7, 10, 13, 16, 19]  # balanced8-A
REPLACEMENT_STEPS = [4, 9, 12, 15, 18]  # every age-2 reuse step
SEEDS = [0, 1]

# Five diverse prompts from the original 10-prompt analysis split.  Held-out
# prompts are deliberately excluded because this experiment informs design.
PROMPTS = [
    {
        "id": "portrait",
        "text": "A cinematic portrait photo of an elderly sailor, detailed weathered face, dramatic side lighting, 85mm lens",
    },
    {
        "id": "animal",
        "text": "A red fox sitting in a snowy forest at sunrise, highly detailed fur, soft golden light",
    },
    {
        "id": "interior",
        "text": "A cozy reading room with wooden bookshelves, a fireplace, warm lamps, realistic interior photography",
    },
    {
        "id": "architecture",
        "text": "A futuristic glass museum beside a calm lake, symmetrical architecture, blue hour, ultra realistic",
    },
    {
        "id": "texture",
        "text": "Macro photograph of colorful mineral crystals, intricate texture, sharp focus, studio lighting",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs")
    parser.add_argument("--limit", type=int, default=None, help="Limit prompt/seed cases for smoke tests")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    return parser.parse_args()


def load_pipeline() -> StableDiffusionPipeline:
    local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=local_only,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2,
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [(prompt, seed) for prompt in PROMPTS for seed in SEEDS]
    if args.limit is not None:
        cases = cases[: args.limit]

    pipe = load_pipeline()
    scheduler_config = dict(pipe.scheduler.config)
    manifest = {
        "model": MODEL_ID,
        "scheduler": "DPM-Solver++ order 2",
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "cache_branch_id": 0,
        "refresh_schedule": REFRESH_STEPS,
        "replacement_steps": REPLACEMENT_STEPS,
        "split": "analysis subset (5 prompts x 2 seeds); no held-out prompts",
        "cases": [],
    }

    for case_index, (prompt, seed) in enumerate(cases, start=1):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        latent = torch.randn(
            (1, 4, args.height // 8, args.width // 8),
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        generation_args = dict(
            prompt=prompt["text"],
            num_inference_steps=20,
            guidance_scale=7.5,
            latents=latent.clone(),
            height=args.height,
            width=args.width,
            output_type="np",
        )
        pristine = None
        if case_index == 1:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                scheduler_config, algorithm_type="dpmsolver++", solver_order=2
            )
            with torch.inference_mode():
                pristine = pipe(**generation_args).images.copy()

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            scheduler_config, algorithm_type="dpmsolver++", solver_order=2
        )
        pipe.scheduler.set_timesteps(20, device="cuda")
        helper = BlockReplacementOracleHelper(
            pipe,
            refresh_steps=REFRESH_STEPS,
            replacement_steps=REPLACEMENT_STEPS,
            guidance_scale=7.5,
        )
        helper.set_params(cache_interval=1, cache_branch_id=0)
        helper.enable()
        started = time.perf_counter()
        with torch.inference_mode():
            result = pipe(**{**generation_args, "latents": latent.clone()})
        elapsed = time.perf_counter() - started
        helper.disable()
        if pristine is not None and not np.array_equal(pristine, result.images):
            raise RuntimeError("Oracle instrumentation changed the uncached trajectory")

        case_id = f"{prompt['id']}_seed{seed}"
        trace_path = args.output_dir / f"{case_id}.json"
        trace_path.write_text(json.dumps(helper.trace, indent=2), encoding="utf-8")
        image = Image.fromarray((result.images[0] * 255).round().clip(0, 255).astype(np.uint8))
        image.save(args.output_dir / f"{case_id}.png")
        manifest["cases"].append(
            {
                "case_id": case_id,
                "prompt_id": prompt["id"],
                "prompt": prompt["text"],
                "seed": seed,
                "elapsed_seconds": elapsed,
                "pristine_exact_match": pristine is not None,
                "trace": trace_path.name,
            }
        )
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"[{case_index}/{len(cases)}] {case_id}: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
