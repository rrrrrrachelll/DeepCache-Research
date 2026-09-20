"""Closed-loop equal-budget comparison of balanced eight-call schedules."""
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
from research.results.balanced8_dev_20260920.balanced_schedules import SCHEDULES
from research.results.pilot_20260908.compare_sampling import similarity
from research.results.stage_aware_v2_dev_20260915.stage_aware_cache import ScheduledDeepCacheHelper

MODES = tuple(SCHEDULES)
STAGE_REFERENCE = Path(__file__).resolve().parent.parent / "stage_aware_v2_dev_20260915" / "run"
DROP_REFERENCE = Path(__file__).resolve().parent.parent / "refresh_position_ablation_20260917" / "run"


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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repair-nsfw", action="store_true",
                        help="On resume, regenerate rows replaced by the safety checker")
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

    stage_complete = json.loads((STAGE_REFERENCE / "complete.json").read_text())
    drop_complete = json.loads((DROP_REFERENCE / "complete.json").read_text())
    assert stage_complete["cases"] == drop_complete["cases"] == 20
    stage_rows = json.loads((STAGE_REFERENCE / "metrics.json").read_text())
    drop_rows = json.loads((DROP_REFERENCE / "metrics.json").read_text())
    reference = {(r["case"], r["mode"]): r for r in stage_rows
                 if r["mode"] in ("dpm20", "uniform9")}
    reference.update({(r["case"], r["mode"]): r for r in drop_rows
                      if r["mode"] == "drop_middle8"})
    assert len(reference) == 60

    torch.set_num_threads(4)
    model_path = snapshot_download("runwayml/stable-diffusion-v1-5", local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, local_files_only=True).to("cuda:0")
    if args.repair_nsfw:
        if not args.resume:
            raise ValueError("--repair-nsfw requires --resume")
        pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    original_config = dict(pipe.scheduler.config)
    manifest = dict(
        experiment="Balanced eight-call refresh development comparison",
        model_path=model_path, cases=cases, modes=MODES, prompts=prompts,
        held_out_used=False, consumed_held_out_reused=False,
        height=512, width=512, guidance_scale=7.5, steps=20,
        dtype="float16", cache_branch_id=0, refresh_indices=SCHEDULES,
        design=("Both schedules keep the validated early scaffold [0,2,5], use "
                "exactly eight full calls, end at step 19, and limit every anchor "
                "gap to three steps."),
        reference_directories={"dpm20_uniform9": str(STAGE_REFERENCE),
                               "drop_middle8": str(DROP_REFERENCE)},
        gpu=torch.cuda.get_device_name(), python=sys.executable,
        versions={name: importlib.metadata.version(name) for name in
                  ["torch", "diffusers", "transformers", "numpy", "Pillow"]},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__), source_dir / "balanced_schedules.py",
                        source_dir.parent / "stage_aware_v2_dev_20260915" /
                        "stage_aware_cache.py", prompts_path]},
        warmups_per_mode=1, scheduler_config=json_safe(original_config),
        timing=("CUDA-synchronized full pipeline wall time; includes text encoder, "
                "UNet, VAE, safety checker and CPU conversion; excludes setup, saving "
                "and quality metrics"),
        quality=("Saved-RGB PSNR and Gaussian SSIM against the paired uncached DPM20 images."),
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
    if args.repair_nsfw:
        flagged = [r for r in rows if r["nsfw_flag"]]
        if not flagged:
            raise RuntimeError("No safety-checker replacements found to repair")
        rows = [r for r in rows if not r["nsfw_flag"]]
        save_json(metrics_path, rows)
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
            solver_type="midpoint", timestep_spacing="linspace", use_karras_sigmas=False)
        pipe.scheduler.set_timesteps(20, device="cuda")
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
            seconds = time.perf_counter()-start
            peak = torch.cuda.max_memory_allocated()/2**20
            trace = list(helper.trace)
        finally:
            pipe.unet.forward = previous
            helper.disable()
        assert len(calls) == len(trace) == 20
        assert tuple(r["index"] for r in trace if r["refresh"]) == SCHEDULES[mode]
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
            assert latent_hash == reference[case, "dpm20"]["initial_latent_sha256"]
            image = (result.images[0]*255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(image).save(args.output / f"{case}_{mode}.png")
            save_json(args.output / f"{case}_{mode}_steps.json", trace)
            dpm_image = np.asarray(Image.open(
                STAGE_REFERENCE / f"{case}_dpm20.png").convert("RGB"))
            quality = similarity(dpm_image, image)
            row = dict(case=case, prompt_id=prompt_id, mode=mode, seed=seed,
                       seconds=seconds, peak_allocated_mib=peak, unet_calls=len(trace),
                       full_calls=sum(r["refresh"] for r in trace),
                       reuse_calls=sum(not r["refresh"] for r in trace),
                       initial_latent_sha256=latent_hash,
                       refresh_indices=[r["index"] for r in trace if r["refresh"]],
                       max_anchor_gap=max(b-a for a,b in zip(SCHEDULES[mode], SCHEDULES[mode][1:])),
                       refresh_reasons=dict(Counter(r["reason"] for r in trace if r["refresh"])),
                       nsfw_flag=(None if result.nsfw_content_detected is None else
                                  bool(result.nsfw_content_detected[0])),
                       ssim=quality["ssim"], psnr_db=quality["psnr_db"],
                       mean_absolute_rgb_error=float(
                           np.abs(dpm_image.astype(float)-image.astype(float)).mean()/255))
            rows.append(row)
            completed.add((case, mode))
            save_json(metrics_path, rows)
            event(json.dumps(row))

    assert len(rows) == len(cases)*len(MODES)
    for case, _, _, _ in cases:
        selected = [r for r in rows if r["case"] == case]
        assert len(selected) == len(MODES)
        assert len({r["initial_latent_sha256"] for r in selected}) == 1
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for mode in MODES:
        subset = [r for r in rows if r["mode"] == mode]
        summary[mode] = dict(samples=len(subset),
            mean_seconds=statistics.mean(r["seconds"] for r in subset),
            seconds_range=[min(r["seconds"] for r in subset), max(r["seconds"] for r in subset)],
            mean_full_calls=statistics.mean(r["full_calls"] for r in subset),
            mean_ssim=statistics.mean(r["ssim"] for r in subset),
            mean_psnr_db=statistics.mean(r["psnr_db"] for r in subset),
            mean_absolute_rgb_error=statistics.mean(r["mean_absolute_rgb_error"] for r in subset),
            peak_allocated_mib=max(r["peak_allocated_mib"] for r in subset))
    save_json(args.output / "summary.json", summary)

    comparison_dir = args.output / "comparisons"; comparison_dir.mkdir(exist_ok=True)
    display_modes = ("dpm20", "uniform9", "drop_middle8") + MODES
    for prompt_id in sorted({case[1] for case in cases}):
        prompt_cases = [case for case in cases if case[1] == prompt_id]
        canvas = Image.new("RGB", (512*len(display_modes), 550*len(prompt_cases)), "white")
        draw = ImageDraw.Draw(canvas)
        for y, (case, _, _, _) in enumerate(prompt_cases):
            for x, mode in enumerate(display_modes):
                if mode in ("dpm20", "uniform9"):
                    path = STAGE_REFERENCE / f"{case}_{mode}.png"
                elif mode == "drop_middle8":
                    path = DROP_REFERENCE / f"{case}_{mode}.png"
                else:
                    path = args.output / f"{case}_{mode}.png"
                calls = 20 if mode == "dpm20" else (9 if mode == "uniform9" else 8)
                canvas.paste(Image.open(path).convert("RGB"), (x*512, y*550+38))
                draw.text((x*512+8, y*550+8), f"{case} | {mode} | full={calls}", fill="black")
        canvas.save(comparison_dir / f"{prompt_id}.png")
    save_json(args.output / "complete.json",
              dict(cases=len(cases), generations=len(rows), modes=list(MODES)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
