"""Closed-loop, equal-budget comparison of uniform and stage-aware DeepCache."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.results.pilot_20260908.compare_sampling import similarity
from research.results.stage_aware_v2_dev_20260915.stage_aware_cache import (
    FIXED3_REFRESH, STAGE_AWARE9_REFRESH, UNIFORM9_REFRESH,
    ScheduledDeepCacheHelper,
)

MODES = ("dpm20", "fixed3", "uniform9", "stage_aware9")
SCHEDULES = {
    "fixed3": FIXED3_REFRESH,
    "uniform9": UNIFORM9_REFRESH,
    "stage_aware9": STAGE_AWARE9_REFRESH,
}


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


def build_cases(prompts):
    return [(f'{prompt["id"]}_{seed}', prompt["id"], prompt["prompt"], seed)
            for prompt in prompts["analysis"] for seed in prompts["seeds"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="Smoke-test case limit; zero runs all development cases")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source_dir = Path(__file__).resolve().parent
    prompts_path = source_dir.parent / "oracle_20260911" / "oracle_prompts.json"
    prompts = json.loads(prompts_path.read_text())
    cases = build_cases(prompts)
    if args.limit:
        cases = cases[:args.limit]
    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError("Resume output directory does not exist")
    else:
        args.output.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(4)
    model_path = snapshot_download("runwayml/stable-diffusion-v1-5", local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, local_files_only=True).to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    original_config = dict(pipe.scheduler.config)
    manifest = dict(
        experiment="Closed-loop stage-aware refresh development experiment",
        model_path=model_path, cases=cases, modes=MODES, prompts=prompts,
        held_out_used=False, height=512, width=512, guidance_scale=7.5,
        steps=20, dtype="float16", cache_branch_id=0,
        refresh_indices=SCHEDULES,
        stage_definition={"early": [0, 6], "middle": [7, 13], "late": [14, 19]},
        stage_aware_full_calls={"early": 2, "middle": 3, "late": 4},
        gpu=torch.cuda.get_device_name(), python=sys.executable,
        versions={name: importlib.metadata.version(name) for name in
                  ["torch", "diffusers", "transformers", "numpy", "Pillow"]},
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__), source_dir / "stage_aware_cache.py", prompts_path]},
        warmups_per_mode=1, scheduler_config=json_safe(original_config),
        timing=("CUDA-synchronized full pipeline wall time; includes text encoder, "
                "UNet, VAE, safety checker and CPU conversion; excludes setup, saving "
                "and quality metrics"),
        quality=("Saved-RGB PSNR and Gaussian SSIM against paired uncached DPM20; "
                 "reference similarity is not absolute perceptual quality"),
        note=("Development prompts only. Uniform9 and stage_aware9 have exactly nine "
              "full UNet calls; fixed3 is a seven-call speed reference."),
    )
    manifest_path = args.output / "manifest.json"
    if args.resume:
        existing = json.loads(manifest_path.read_text())
        for key in ("experiment", "cases", "modes", "prompts", "held_out_used",
                    "steps", "cache_branch_id", "refresh_indices"):
            assert existing[key] == json_safe(manifest)[key], f"Resume mismatch: {key}"
    else:
        save_json(manifest_path, json_safe(manifest))

    metrics_path = args.output / "metrics.json"
    rows = json.loads(metrics_path.read_text()) if args.resume and metrics_path.exists() else []
    completed = {(r["case"], r["mode"]) for r in rows}
    if len(completed) != len(rows):
        raise RuntimeError("Duplicate rows in partial metrics")

    def event(message):
        print(message, flush=True)
        with (args.output / "run.log").open("a") as handle:
            handle.write(message + "\n")

    def generate(mode, prompt, seed):
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            original_config, algorithm_type="dpmsolver++", solver_order=2,
            solver_type="midpoint", timestep_spacing="linspace",
            use_karras_sigmas=False)
        pipe.scheduler.set_timesteps(20, device="cuda")
        helper = None
        if mode in SCHEDULES:
            helper = ScheduledDeepCacheHelper(pipe, SCHEDULES[mode])
            helper.enable()
        calls = []
        previous = pipe.unet.forward

        def observed(*positional, **keywords):
            timestep = positional[1] if len(positional) > 1 else keywords["timestep"]
            calls.append(int(timestep.item() if hasattr(timestep, "item") else timestep))
            return previous(*positional, **keywords)

        pipe.unet.forward = observed
        generator = torch.Generator(device="cuda").manual_seed(seed)
        latent = torch.randn((1, 4, 64, 64), generator=generator,
                             device="cuda", dtype=torch.float16)
        latent_hash = hashlib.sha256(latent.cpu().numpy().tobytes()).hexdigest()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            result = pipe(prompt, height=512, width=512, num_inference_steps=20,
                          guidance_scale=7.5, latents=latent, output_type="np")
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() / 2**20
            trace = (list(helper.trace) if helper is not None else
                     [dict(index=i, timestep=t, refresh=True, reason="uncached")
                      for i, t in enumerate(calls)])
        finally:
            pipe.unet.forward = previous
            if helper is not None:
                helper.disable()
        assert len(calls) == len(trace) == 20
        expected = tuple(range(20)) if mode == "dpm20" else SCHEDULES[mode]
        assert tuple(r["index"] for r in trace if r["refresh"]) == expected
        return result, seconds, peak, trace, latent_hash

    for mode in MODES:
        event(f"Warmup {mode}")
        generate(mode, cases[0][2], 0)

    for case_index, (case, prompt_id, prompt, seed) in enumerate(cases):
        order = MODES[case_index % len(MODES):] + MODES[:case_index % len(MODES)]
        for mode in order:
            if (case, mode) in completed:
                event(f"Skip completed {case} {mode}")
                continue
            event(f"Measured {case} {mode}")
            result, seconds, peak, trace, latent_hash = generate(mode, prompt, seed)
            image = (result.images[0] * 255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(image).save(args.output / f"{case}_{mode}.png")
            save_json(args.output / f"{case}_{mode}_steps.json", trace)
            row = dict(
                case=case, prompt_id=prompt_id, mode=mode, seed=seed,
                seconds=seconds, peak_allocated_mib=peak, unet_calls=len(trace),
                full_calls=sum(r["refresh"] for r in trace),
                reuse_calls=sum(not r["refresh"] for r in trace),
                initial_latent_sha256=latent_hash,
                refresh_indices=[r["index"] for r in trace if r["refresh"]],
                refresh_reasons=dict(Counter(r["reason"] for r in trace if r["refresh"])),
                nsfw_flag=(None if result.nsfw_content_detected is None
                           else bool(result.nsfw_content_detected[0])),
            )
            rows.append(row)
            completed.add((case, mode))
            save_json(metrics_path, rows)
            event(json.dumps(row))

    assert len(rows) == len(cases) * len(MODES)
    images = {(case, mode): np.asarray(Image.open(args.output / f"{case}_{mode}.png").convert("RGB"))
              for case, _, _, _ in cases for mode in MODES}
    for case, _, _, _ in cases:
        selected = [r for r in rows if r["case"] == case]
        assert len({r["initial_latent_sha256"] for r in selected}) == 1
        for row in selected:
            row.update(similarity(images[case, "dpm20"], images[case, row["mode"]]))
            row["mean_absolute_rgb_error"] = float(
                np.abs(images[case, "dpm20"].astype(float)
                       - images[case, row["mode"]].astype(float)).mean() / 255)
    save_json(metrics_path, rows)
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for mode in MODES:
        subset = [r for r in rows if r["mode"] == mode]
        summary[mode] = dict(
            samples=len(subset),
            mean_seconds=statistics.mean(r["seconds"] for r in subset),
            seconds_range=[min(r["seconds"] for r in subset),
                           max(r["seconds"] for r in subset)],
            mean_full_calls=statistics.mean(r["full_calls"] for r in subset),
            mean_ssim=statistics.mean(r["ssim"] for r in subset),
            mean_psnr_db=(None if mode == "dpm20" else
                          statistics.mean(r["psnr_db"] for r in subset)),
            mean_absolute_rgb_error=statistics.mean(
                r["mean_absolute_rgb_error"] for r in subset),
            peak_allocated_mib=max(r["peak_allocated_mib"] for r in subset),
        )
    for item in summary.values():
        item["speedup_vs_dpm20"] = summary["dpm20"]["mean_seconds"] / item["mean_seconds"]
    save_json(args.output / "summary.json", summary)

    comparison_dir = args.output / "comparisons"
    comparison_dir.mkdir(exist_ok=True)
    for prompt_id in sorted({case[1] for case in cases}):
        prompt_cases = [case for case in cases if case[1] == prompt_id]
        canvas = Image.new("RGB", (512 * len(MODES), 550 * len(prompt_cases)), "white")
        draw = ImageDraw.Draw(canvas)
        for y, (case, _, _, _) in enumerate(prompt_cases):
            for x, mode in enumerate(MODES):
                row = next(r for r in rows if r["case"] == case and r["mode"] == mode)
                draw.text((x*512+8, y*550+8),
                          f"{case} | {mode} | full={row['full_calls']}", fill="black")
                canvas.paste(Image.fromarray(images[case, mode]), (x*512, y*550+38))
        canvas.save(comparison_dir / f"{prompt_id}.png")
    save_json(args.output / "complete.json",
              dict(cases=len(cases), generations=len(rows), modes=list(MODES)))
    event(json.dumps(summary, indent=2))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
