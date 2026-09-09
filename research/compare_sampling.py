"""Paired SD1.5 sampling pilot; leaves original entry points untouched."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DeepCache import DeepCacheSDHelper

CASES = [
    ("astronaut_42", "a photo of an astronaut on a moon", 42),
    ("astronaut_43", "a photo of an astronaut on a moon", 43),
    ("landscape_42", "a photograph of a mountain lake surrounded by pine trees, reflections in clear water, daylight", 42),
]
MODES = [("pndm50", 50, False), ("dpm20", 20, False),
         ("pndm50_cache", 50, True), ("dpm20_cache", 20, True)]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def similarity(reference, candidate):
    # Gaussian SSIM: 11x11 window, sigma 1.5, population moments, valid crop.
    a = torch.from_numpy(reference.copy()).permute(2, 0, 1)[None].float() / 255
    b = torch.from_numpy(candidate.copy()).permute(2, 0, 1)[None].float() / 255
    x = torch.arange(11).float() - 5
    g = torch.exp(-x.square() / (2 * 1.5 ** 2))
    g /= g.sum()
    kernel = (g[:, None] * g[None, :]).expand(3, 1, 11, 11)
    blur = lambda v: F.conv2d(v, kernel, groups=3)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a*a)-ma*ma, blur(b*b)-mb*mb, blur(a*b)-ma*mb
    score = ((2*ma*mb+0.01**2)*(2*cov+0.03**2) /
             ((ma*ma+mb*mb+0.01**2)*(va+vb+0.03**2))).mean().item()
    mse = (a-b).square().mean().item()
    return {"psnr_db": -10*np.log10(mse) if mse else None, "ssim": score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model_path = snapshot_download("runwayml/stable-diffusion-v1-5", local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16,
                                                  local_files_only=True).to("cuda:0")
    original_class, original_config = type(pipe.scheduler), dict(pipe.scheduler.config)
    manifest = {
        "model_path": model_path, "original_scheduler": original_config,
        "versions": {p: importlib.metadata.version(p) for p in
                     ["torch", "diffusers", "transformers", "numpy", "Pillow"]},
        "gpu": torch.cuda.get_device_name(), "python": sys.executable,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": CASES, "modes": MODES, "height": 512, "width": 512,
        "guidance_scale": 7.5, "dtype": "float16", "cache_interval": 3,
        "cache_branch_id": 0, "warmups_per_mode": 1,
        "timing": "CUDA synchronized pipeline wall time including text encoder, UNet, VAE, safety checker, CPU image conversion; excludes model load, helper setup, saving and metrics",
        "quality": "PSNR/SSIM against paired pndm50 output, not ground truth quality; no CLIP/FID",
        "scheduler_configs": {},
    }
    rows, images = [], {}

    def run(mode, steps, cached, prompt, seed):
        pipe.scheduler = (DPMSolverMultistepScheduler.from_config(
            original_config, algorithm_type="dpmsolver++", solver_order=2,
            solver_type="midpoint", timestep_spacing="linspace", use_karras_sigmas=False)
            if mode.startswith("dpm") else original_class.from_config(original_config))
        manifest["scheduler_configs"][mode] = dict(pipe.scheduler.config)
        helper = DeepCacheSDHelper(pipe)
        if cached:
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()
        calls = []
        previous = pipe.unet.forward

        def observed(*a, **kw):
            timestep = int(a[1].item())
            if cached:
                index = list(pipe.scheduler.timesteps).index(timestep)
                refresh = (index % 3 == 0)
            else:
                index, refresh = len(calls), True
            calls.append({"timestep": timestep, "helper_index": index, "refresh": refresh})
            return previous(*a, **kw)

        pipe.unet.forward = observed
        generator = torch.Generator(device="cuda").manual_seed(seed)
        latent = torch.randn((1, 4, 64, 64), generator=generator, device="cuda", dtype=torch.float16)
        latent_hash = hashlib.sha256(latent.cpu().numpy().tobytes()).hexdigest()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            result = pipe(prompt, height=512, width=512, num_inference_steps=steps,
                          guidance_scale=7.5, latents=latent, output_type="np")
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            memory = torch.cuda.max_memory_allocated() / 2**20
        finally:
            pipe.unet.forward = previous
            if cached:
                helper.disable()
        return result, seconds, memory, calls, latent_hash

    for mode, steps, cached in MODES:
        print(f"Warmup {mode}", flush=True)
        run(mode, steps, cached, CASES[0][1], 0)

    for case_index, (case, prompt, seed) in enumerate(CASES):
        # Rotate order to reduce a systematic timing advantage for one method.
        order = MODES[case_index:] + MODES[:case_index]
        for mode, steps, cached in order:
            print(f"Measured {case} {mode}", flush=True)
            result, seconds, memory, calls, latent_hash = run(mode, steps, cached, prompt, seed)
            array = (result.images[0] * 255).round().clip(0, 255).astype(np.uint8)
            images[case, mode] = array
            Image.fromarray(array).save(args.output / f"{case}_{mode}.png")
            row = {"case": case, "mode": mode, "seed": seed, "seconds": seconds,
                   "peak_allocated_mib": memory, "unet_calls": len(calls),
                   "full_calls": sum(c["refresh"] for c in calls),
                   "reuse_calls": sum(not c["refresh"] for c in calls),
                   "initial_latent_sha256": latent_hash,
                   "nsfw_flag": bool(result.nsfw_content_detected[0]) if result.nsfw_content_detected is not None else None}
            rows.append(row)
            write_json(args.output / f"{case}_{mode}_steps.json", calls)
            write_json(args.output / "metrics.json", rows)
            write_json(args.output / "manifest.json", manifest)
            print(json.dumps(row), flush=True)

    for row in rows:
        row.update(similarity(images[row["case"], "pndm50"], images[row["case"], row["mode"]]))
        if row["mode"] == "dpm20_cache":
            row.update({"vs_dpm20_"+k: v for k, v in similarity(
                images[row["case"], "dpm20"], images[row["case"], "dpm20_cache"]).items()})
    write_json(args.output / "metrics.json", rows)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (args.output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for mode, _, _ in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        summary[mode] = {"mean_seconds": statistics.mean(r["seconds"] for r in selected),
                         "min_seconds": min(r["seconds"] for r in selected),
                         "max_seconds": max(r["seconds"] for r in selected),
                         "mean_ssim": statistics.mean(r["ssim"] for r in selected),
                         "mean_psnr_db": statistics.mean(r["psnr_db"] for r in selected) if mode != "pndm50" else None,
                         "peak_allocated_mib": max(r["peak_allocated_mib"] for r in selected)}
    for value in summary.values():
        value["speedup"] = summary["pndm50"]["mean_seconds"] / value["mean_seconds"]
    write_json(args.output / "summary.json", summary)
    canvas = Image.new("RGB", (4*512, len(CASES)*550), "white")
    draw = ImageDraw.Draw(canvas)
    for y, (case, _, _) in enumerate(CASES):
        for x, (mode, _, _) in enumerate(MODES):
            canvas.paste(Image.fromarray(images[case, mode]), (x*512, y*550+38))
            draw.text((x*512+8, y*550+10), f"{case} | {mode}", fill="black")
    canvas.save(args.output / "comparison.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
