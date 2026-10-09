"""Experiment 14B: close the loop on static DiT whole-block residual reuse."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time
from types import MethodType

import numpy as np
from PIL import Image
import torch
from diffusers import DiTPipeline
from skimage.metrics import structural_similarity

from profile_dit_redundancy import DEFAULT_CLASS_IDS, MODEL_ID, make_generator


TOTAL_STEPS_WITH_PREVIOUS = 19
DEFAULT_TIERS = {
    "conservative_25": 0.25,
    "target_35": 0.35,
    "aggressive_50": 0.50,
}


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_profile_rows(profile_dir: Path, class_ids: list[int]) -> dict[int, list[dict]]:
    rows = {}
    for class_id in class_ids:
        path = profile_dir / f"class_{class_id:04d}_profile.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Experiment 14A profile: {path}")
        rows[class_id] = json.loads(path.read_text())
    return rows


def rank_residual_coordinates(
    profile_rows: dict[int, list[dict]], block_count: int, steps: int
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], dict]]:
    """Rank step/block coordinates by worst-case relative L2, then cosine."""
    by_class: dict[int, dict[tuple[int, int], dict]] = {}
    for class_id, rows in profile_rows.items():
        by_class[class_id] = {
            (int(row["step"]), int(row["block"])): row
            for row in rows
            if row["component"] == "residual" and "relative_l2_change" in row
        }

    universe = [(step, block) for step in range(1, steps) for block in range(block_count)]
    expected = set(universe)
    for class_id, values in by_class.items():
        if set(values) != expected:
            raise ValueError(f"Incomplete residual profile for class {class_id}")

    aggregate = {}
    for coordinate in universe:
        class_values = [values[coordinate] for values in by_class.values()]
        aggregate[coordinate] = {
            "max_relative_l2": max(float(row["relative_l2_change"]) for row in class_values),
            "min_cosine": min(float(row["cosine"]) for row in class_values),
        }
    ranked = sorted(
        universe,
        key=lambda item: (
            aggregate[item]["max_relative_l2"],
            -aggregate[item]["min_cosine"],
            item[0],
            item[1],
        ),
    )
    return ranked, aggregate


def build_policies(
    ranked: list[tuple[int, int]], tiers: dict[str, float], seed: int
) -> tuple[dict[str, set[tuple[int, int]]], dict[str, dict]]:
    policies: dict[str, set[tuple[int, int]]] = {}
    metadata: dict[str, dict] = {}
    rng = np.random.default_rng(seed)
    shuffled = [ranked[index] for index in rng.permutation(len(ranked))]
    for name, fraction in tiers.items():
        count = int(math.ceil(fraction * len(ranked)))
        selected = set(ranked[:count])
        control = set(shuffled[:count])
        policies[name] = selected
        policies[f"{name}_shuffled"] = control
        metadata[name] = {
            "requested_fraction": fraction,
            "selected_coordinates": count,
            "actual_fraction": count / len(ranked),
        }
        metadata[f"{name}_shuffled"] = {
            **metadata[name],
            "control": "globally shuffled step/block coordinates with matched count",
        }
    return policies, metadata


class BlockResidualReuse:
    """Temporarily replace selected block calls with the last computed residual."""

    def __init__(self, transformer, coordinates: set[tuple[int, int]]):
        self.blocks = list(transformer.transformer_blocks)
        self.coordinates = coordinates
        self.original_forwards = []
        self.calls = [0] * len(self.blocks)
        self.cached_residuals: list[torch.Tensor | None] = [None] * len(self.blocks)
        self.skipped = 0
        self.computed = 0

    def _replacement(self, block_index: int, original_forward):
        controller = self

        def forward(_block, hidden_states, *args, **kwargs):
            step = controller.calls[block_index]
            cached = controller.cached_residuals[block_index]
            if (step, block_index) in controller.coordinates and cached is not None:
                output = hidden_states + cached
                controller.skipped += 1
            else:
                output = original_forward(hidden_states, *args, **kwargs)
                controller.cached_residuals[block_index] = (output - hidden_states).detach().clone()
                controller.computed += 1
            controller.calls[block_index] += 1
            return output

        return forward

    def __enter__(self):
        for index, block in enumerate(self.blocks):
            original = block.forward
            self.original_forwards.append(original)
            block.forward = MethodType(self._replacement(index, original), block)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for block, original in zip(self.blocks, self.original_forwards):
            block.forward = original

    def validate(self, expected_steps: int) -> None:
        if self.calls != [expected_steps] * len(self.blocks):
            raise RuntimeError(f"Incomplete block calls: {self.calls}")
        expected_skips = len(self.coordinates)
        if self.skipped != expected_skips:
            raise RuntimeError(f"Expected {expected_skips} skips, observed {self.skipped}")


def run_pipeline(pipe, class_id: int, seed: int, steps: int, guidance_scale: float,
                 coordinates: set[tuple[int, int]] | None = None):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if coordinates is None:
        with torch.inference_mode():
            images = pipe(
                class_labels=[class_id], generator=make_generator(seed),
                num_inference_steps=steps, guidance_scale=guidance_scale,
                output_type="np",
            ).images
        stats = {"skipped_block_calls": 0, "computed_block_calls": steps * len(pipe.transformer.transformer_blocks)}
    else:
        controller = BlockResidualReuse(pipe.transformer, coordinates)
        with torch.inference_mode(), controller:
            images = pipe(
                class_labels=[class_id], generator=make_generator(seed),
                num_inference_steps=steps, guidance_scale=guidance_scale,
                output_type="np",
            ).images
        controller.validate(steps)
        stats = {"skipped_block_calls": controller.skipped, "computed_block_calls": controller.computed}
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    stats.update({
        "seconds": seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    })
    return images, stats


class PerceptualMetrics:
    def __init__(self, device: str = "cuda"):
        import lpips
        from DISTS_pytorch import DISTS

        self.device = device
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.dists = DISTS().to(device).eval()

    def __call__(self, reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
        ref = np.clip(reference[0], 0.0, 1.0)
        cand = np.clip(candidate[0], 0.0, 1.0)
        ssim = structural_similarity(ref, cand, channel_axis=-1, data_range=1.0)
        ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        cand_tensor = torch.from_numpy(cand).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.inference_mode():
            lpips_value = self.lpips(ref_tensor * 2 - 1, cand_tensor * 2 - 1)
            dists_value = self.dists(ref_tensor, cand_tensor)
        return {
            "ssim": float(ssim),
            "lpips": float(lpips_value.item()),
            "dists": float(dists_value.item()),
            "pixel_mae": float(np.mean(np.abs(ref.astype(np.float64) - cand.astype(np.float64)))),
        }


def save_image(path: Path, image: np.ndarray) -> None:
    pixels = (np.clip(image[0], 0.0, 1.0) * 255).round().astype(np.uint8)
    Image.fromarray(pixels).save(path)


def label_for(pipe, class_id: int) -> str:
    matches = sorted(name for name, value in pipe.labels.items() if value == class_id)
    return matches[0] if matches else f"class_{class_id}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-ids", type=int, nargs="+", default=list(DEFAULT_CLASS_IDS))
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--policy-seed", type=int, default=1402)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = DiTPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16, local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    block_count = len(pipe.transformer.transformer_blocks)
    profile_rows = load_profile_rows(args.profile_dir, args.class_ids)
    ranked, aggregate = rank_residual_coordinates(profile_rows, block_count, args.steps)
    policies, policy_metadata = build_policies(ranked, DEFAULT_TIERS, args.policy_seed)
    for name, coordinates in policies.items():
        policy_metadata[name]["coordinates"] = sorted([list(item) for item in coordinates])
    save_json(args.output_dir / "policies.json", policy_metadata)
    save_json(args.output_dir / "coordinate_ranking.json", [
        {"step": step, "block": block, **aggregate[(step, block)]}
        for step, block in ranked
    ])

    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "experiment": "Experiment 14B static DiT whole-block residual reuse",
        "model": MODEL_ID,
        "model_path": str(args.model_path.resolve()),
        "profile_dir": str(args.profile_dir.resolve()),
        "class_ids": args.class_ids,
        "seed": args.seed,
        "policy_seed": args.policy_seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "dtype": "float16",
        "resolution": 256,
        "gpu": torch.cuda.get_device_name(),
        "block_count": block_count,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "diffusers", "transformers", "numpy", "Pillow", "scikit-image", "lpips")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": source_hash,
    }
    save_json(args.output_dir / "manifest.json", manifest)

    # Warm kernels before timing.
    with torch.inference_mode():
        pipe(class_labels=[args.class_ids[0]], generator=make_generator(args.seed),
             num_inference_steps=2, guidance_scale=args.guidance_scale, output_type="np")
    torch.cuda.synchronize()

    results = []
    strategy_order = list(policies)
    for ordinal, class_id in enumerate(args.class_ids, start=1):
        label = label_for(pipe, class_id)
        print(f"baseline [{ordinal}/{len(args.class_ids)}] class={class_id} label={label}", flush=True)
        baseline, baseline_stats = run_pipeline(
            pipe, class_id, args.seed, args.steps, args.guidance_scale)
        save_image(args.output_dir / f"class_{class_id:04d}_baseline.png", baseline)
        baseline_row = {
            "class_id": class_id, "label": label, "strategy": "baseline",
            **baseline_stats, "ssim": 1.0, "lpips": 0.0, "dists": 0.0, "pixel_mae": 0.0,
        }
        results.append(baseline_row)
        save_json(args.output_dir / "results.partial.json", results)

        candidates = []
        for strategy in strategy_order:
            print(f"  strategy={strategy}", flush=True)
            image, stats = run_pipeline(
                pipe, class_id, args.seed, args.steps, args.guidance_scale, policies[strategy])
            save_image(args.output_dir / f"class_{class_id:04d}_{strategy}.png", image)
            candidates.append((strategy, image, stats))

        # Release most of the pipeline before allocating perceptual backbones.
        metrics = PerceptualMetrics()
        for strategy, image, stats in candidates:
            row = {
                "class_id": class_id, "label": label, "strategy": strategy,
                **stats, **metrics(baseline, image),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            save_json(args.output_dir / "results.partial.json", results)
        del metrics
        torch.cuda.empty_cache()

    save_json(args.output_dir / "results.json", results)
    save_json(args.output_dir / "complete.json", {
        "complete": True,
        "cases": len(args.class_ids),
        "strategies": 1 + len(policies),
        "rows": len(results),
    })
    (args.output_dir / "results.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
