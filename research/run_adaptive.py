"""Fresh paired DPM20, fixed-cache and adaptive-v1 measurements."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DeepCache import DeepCacheSDHelper
from research.adaptive_cache import AdaptiveDeepCacheHelper, CachePolicy
from research.compare_sampling import CASES, similarity


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--risk-threshold", type=float, default=0.65)
    p.add_argument("--feature-weight", type=float, default=2)
    p.add_argument("--feature-spike", type=float, default=0.5)
    p.add_argument("--max-age", type=int, default=3)
    args = p.parse_args()
    policy = CachePolicy(args.risk_threshold, args.feature_weight, args.feature_spike, args.max_age)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model_path = snapshot_download("runwayml/stable-diffusion-v1-5", local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16,
                                                  local_files_only=True).to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    original_config = dict(pipe.scheduler.config)
    modes = ["dpm20", "fixed3", "lambda_only", "adaptive_v1"]
    manifest = dict(model_path=model_path, cases=CASES, modes=modes, original_scheduler=original_config,
                    height=512, width=512, guidance_scale=7.5, steps=20, dtype="float16",
                    cache_branch_id=0, fixed_interval=3, policy=policy.__dict__,
                    gpu=torch.cuda.get_device_name(), python=sys.executable,
                    versions={name: importlib.metadata.version(name) for name in
                              ["torch", "diffusers", "transformers", "numpy", "Pillow"]},
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                   [Path(__file__), Path(__file__).with_name("adaptive_cache.py"),
                                    Path(__file__).with_name("compare_sampling.py")]},
                    warmups_per_mode=1, scheduler_configs={},
                    quality_reference="fresh paired uncached DPM20 RGB image",
                    timing="CUDA synchronized full pipeline; excludes setup, saving and metrics; includes proxy and decision overhead",
                    note="Untuned heuristic v1. No guarantee of equal refresh counts or equal latency.")
    rows, pictures = [], {}

    def event(value):
        print(value, flush=True)
        with (args.output / "run.log").open("a") as f:
            f.write(value + "\n")

    def generate(mode, prompt, seed):
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            original_config, algorithm_type="dpmsolver++", solver_order=2,
            solver_type="midpoint", timestep_spacing="linspace", use_karras_sigmas=False)
        pipe.scheduler.set_timesteps(20, device="cuda")
        manifest["scheduler_configs"][mode] = {
            "actual_class": type(pipe.scheduler).__name__, "config": json_safe(dict(pipe.scheduler.config))}
        helper = None
        if mode == "fixed3":
            helper = DeepCacheSDHelper(pipe)
            helper.set_params(cache_interval=3, cache_branch_id=0)
        elif mode in ("lambda_only", "adaptive_v1"):
            selected = policy if mode == "adaptive_v1" else CachePolicy(
                risk_threshold=policy.risk_threshold, feature_weight=0,
                feature_spike=policy.feature_spike, max_age=policy.max_age)
            helper = AdaptiveDeepCacheHelper(pipe, selected)
        if helper is not None:
            helper.enable()
        calls = []
        old_forward = pipe.unet.forward

        def observed(*a, **kw):
            calls.append(int(a[1].item()))
            return old_forward(*a, **kw)

        pipe.unet.forward = observed
        generator = torch.Generator(device="cuda").manual_seed(seed)
        latent = torch.randn((1, 4, 64, 64), generator=generator, device="cuda", dtype=torch.float16)
        noise_hash = hashlib.sha256(latent.cpu().numpy().tobytes()).hexdigest()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            result = pipe(prompt, height=512, width=512, num_inference_steps=20,
                          guidance_scale=7.5, latents=latent, output_type="np")
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() / 2**20
            if isinstance(helper, AdaptiveDeepCacheHelper):
                trace = list(helper.trace)
            else:
                trace = [dict(index=i, timestep=t, refresh=mode == "dpm20" or i % 3 == 0,
                              reason="uncached" if mode == "dpm20" else "fixed_interval")
                         for i, t in enumerate(calls)]
        finally:
            pipe.unet.forward = old_forward
            if helper is not None:
                helper.disable()
        assert len(calls) == len(trace) == 20
        return result, seconds, peak, trace, noise_hash

    for mode in modes:
        event(f"Warmup {mode}")
        generate(mode, CASES[0][1], 0)
    save_json(args.output / "manifest.json", json_safe(manifest))
    for case_i, (case, prompt, seed) in enumerate(CASES):
        for mode in modes[case_i:] + modes[:case_i]:
            event(f"Measured {case} {mode}")
            result, seconds, peak, trace, noise_hash = generate(mode, prompt, seed)
            image = (result.images[0]*255).round().clip(0, 255).astype(np.uint8)
            pictures[case, mode] = image
            Image.fromarray(image).save(args.output / f"{case}_{mode}.png")
            row = dict(case=case, mode=mode, seed=seed, seconds=seconds, peak_allocated_mib=peak,
                       unet_calls=len(trace), full_calls=sum(r["refresh"] for r in trace),
                       reuse_calls=sum(not r["refresh"] for r in trace),
                       initial_latent_sha256=noise_hash,
                       nsfw_flag=bool(result.nsfw_content_detected[0]) if result.nsfw_content_detected is not None else None,
                       refresh_indices=[r["index"] for r in trace if r["refresh"]],
                       refresh_reasons=dict(Counter(r["reason"] for r in trace if r["refresh"])))
            rows.append(row)
            save_json(args.output / f"{case}_{mode}_steps.json", trace)
            save_json(args.output / "metrics.json", rows)
            event(json.dumps(row))

    for case, _, _ in CASES:
        paired = [r for r in rows if r["case"] == case]
        assert len({r["initial_latent_sha256"] for r in paired}) == 1
    for row in rows:
        row.update(similarity(pictures[row["case"], "dpm20"], pictures[row["case"], row["mode"]]))
    save_json(args.output / "metrics.json", rows)
    with (args.output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        summary[mode] = dict(mean_seconds=statistics.mean(r["seconds"] for r in subset),
                             mean_full_calls=statistics.mean(r["full_calls"] for r in subset),
                             mean_ssim=statistics.mean(r["ssim"] for r in subset),
                             mean_psnr_db=None if mode == "dpm20" else statistics.mean(r["psnr_db"] for r in subset),
                             peak_allocated_mib=max(r["peak_allocated_mib"] for r in subset),
                             seconds_range=[min(r["seconds"] for r in subset), max(r["seconds"] for r in subset)])
    for row in summary.values():
        row["speedup_vs_dpm20"] = summary["dpm20"]["mean_seconds"] / row["mean_seconds"]
    save_json(args.output / "summary.json", summary)
    canvas = Image.new("RGB", (512*len(modes), 550*len(CASES)), "white")
    draw = ImageDraw.Draw(canvas)
    for y, (case, _, _) in enumerate(CASES):
        for x, mode in enumerate(modes):
            row = next(r for r in rows if r["case"] == case and r["mode"] == mode)
            draw.text((x*512+8, y*550+8), f"{case} | {mode} | full={row['full_calls']}", fill="black")
            canvas.paste(Image.fromarray(pictures[case, mode]), (x*512, y*550+38))
    canvas.save(args.output / "comparison.png")
    event(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
