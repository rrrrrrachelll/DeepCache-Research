"""Run fixed3 shadow-cache diagnostics without changing the reference trajectory."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.oracle_cache import OracleCacheHelper
from research.run_adaptive import save_json, json_safe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test sample limit; zero runs all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prompts_path = Path(__file__).with_name("oracle_prompts.json")
    prompts = json.loads(prompts_path.read_text())
    cases = [(p, seed) for p in prompts["analysis"] for seed in prompts["seeds"]]
    if args.limit:
        cases = cases[:args.limit]
    torch.set_num_threads(4)
    model = snapshot_download("runwayml/stable-diffusion-v1-5", local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16,
                                                 local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    original = dict(pipe.scheduler.config)
    def scheduler():
        return DPMSolverMultistepScheduler.from_config(
            original, algorithm_type="dpmsolver++", solver_order=2,
            solver_type="midpoint", timestep_spacing="linspace", use_karras_sigmas=False)
    manifest = dict(model=model, prompts=prompts, cases=cases, steps=20, interval=3,
                    branch=0, guidance_scale=7.5, resolution=[512, 512], dtype="float16",
                    trajectory="uncached DPM20; shadow cache never advances scheduler",
                    boundary="up_blocks[-1].attentions[-2] output; helper key up/attentions/0/1",
                    feature_error="mean(abs(cached-full))/max(mean(abs(full)),1e-6)",
                    proxy="avg_pool2d(conv_in,4); previous-step and last-refresh relative L1",
                    timing="offline instrumented diagnostic, NOT inference acceleration timing",
                    gpu=torch.cuda.get_device_name(), git_commit=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True).strip(),
                    versions={n: importlib.metadata.version(n) for n in ["torch", "diffusers", "transformers", "numpy"]},
                    source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__), Path(__file__).with_name("oracle_cache.py"), prompts_path]},
                    scheduler=json_safe(dict(scheduler().config)))
    save_json(args.output / "manifest.json", manifest)
    def event(message):
        print(message, flush=True)
        with (args.output / "run.log").open("a") as f:
            f.write(message + "\n")
    rows, metrics = [], []
    for number, (case, seed) in enumerate(cases):
        case_id = f'{case["id"]}_{seed}'
        event(f"[{number+1}/{len(cases)}] {case_id}")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        latent = torch.randn((1, 4, 64, 64), generator=generator, device="cuda", dtype=torch.float16)
        latent_hash = hashlib.sha256(latent.cpu().numpy().tobytes()).hexdigest()
        kwargs = dict(prompt=case["prompt"], latents=latent.clone(), num_inference_steps=20,
                      guidance_scale=7.5, height=512, width=512, output_type="np")
        # One exact reference check detects oracle-induced trajectory changes.
        reference = None
        if number == 0:
            event("Checking paired instrumentation against a pristine uncached pipeline")
            pipe.scheduler = scheduler()
            reference = pipe(**kwargs).images.copy()
        pipe.scheduler = scheduler()
        pipe.scheduler.set_timesteps(20, device="cuda")
        helper = OracleCacheHelper(pipe)
        helper.enable()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            result = pipe(**{**kwargs, "latents": latent.clone()})
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() / 2**20
            trace = list(helper.trace)
        finally:
            helper.disable()
        assert len(trace) == 20 and sum(r["refresh"] for r in trace) == 7
        if reference is not None:
            assert np.array_equal(reference, result.images), "Oracle changed the uncached image"
            event("PASS: oracle output exactly equals pristine uncached output")
        for row in trace:
            row.update(case=case_id, prompt_id=case["id"], seed=seed)
        rows.extend(trace)
        save_json(args.output / f"{case_id}_steps.json", trace)
        Image.fromarray((result.images[0]*255).round().clip(0, 255).astype(np.uint8)).save(args.output / f"{case_id}.png")
        metrics.append(dict(case=case_id, seconds=seconds, peak_allocated_mib=peak,
                            initial_latent_sha256=latent_hash, pristine_check=reference is not None,
                            nsfw_flag=None if result.nsfw_content_detected is None else bool(result.nsfw_content_detected[0])))
        save_json(args.output / "metrics.json", metrics)
        with (args.output / "steps.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        event(f"Finished {case_id}: {seconds:.2f}s, {peak:.0f} MiB")
    save_json(args.output / "complete.json", dict(samples=len(metrics), rows=len(rows)))
    event("Complete")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
