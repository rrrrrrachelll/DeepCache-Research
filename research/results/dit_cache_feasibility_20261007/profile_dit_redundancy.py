"""Experiment 14A: read-only adjacent-step redundancy profiling for DiT."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiTPipeline


MODEL_ID = "facebook/DiT-XL-2-256"
DEFAULT_CLASS_IDS = (207, 360, 417, 980)


def tensor_similarity(current: torch.Tensor, previous: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return stable token cosine and normalized L2 change as GPU scalars."""
    current32 = current.float()
    previous32 = previous.float()
    cosine = F.cosine_similarity(current32, previous32, dim=-1, eps=1e-8).mean()
    change = torch.linalg.vector_norm(current32 - previous32) / torch.linalg.vector_norm(current32).clamp_min(1e-8)
    return {"cosine": cosine, "relative_l2_change": change}


def relative_norm(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value.float()) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-8)


class DiTRedundancyProfiler:
    """Read-only hooks that compare each component with its previous diffusion call."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.blocks = list(transformer.transformer_blocks)
        self.handles = []
        self.calls = [0] * len(self.blocks)
        self.current_inputs: dict[int, torch.Tensor] = {}
        self.previous: dict[tuple[int, str], torch.Tensor] = {}
        self.records: list[dict] = []

    def _component_hook(self, block_index: int, component: str):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise TypeError(f"Expected tensor output from {component}, got {type(output)}")
            step = self.calls[block_index]
            block_input = self.current_inputs[block_index]
            record = {
                "step": step,
                "block": block_index,
                "component": component,
                "relative_input_norm": relative_norm(output, block_input),
            }
            key = (block_index, component)
            if key in self.previous:
                record.update(tensor_similarity(output, self.previous[key]))
            self.records.append(record)
            self.previous[key] = output.detach().clone()
        return hook

    def _block_pre_hook(self, block_index: int):
        def hook(_module, inputs):
            hidden_states = inputs[0]
            self.current_inputs[block_index] = hidden_states
            key = (block_index, "input")
            record = {
                "step": self.calls[block_index],
                "block": block_index,
                "component": "input",
            }
            if key in self.previous:
                record.update(tensor_similarity(hidden_states, self.previous[key]))
            self.records.append(record)
            self.previous[key] = hidden_states.detach().clone()
        return hook

    def _block_hook(self, block_index: int):
        def hook(_module, _inputs, output):
            step = self.calls[block_index]
            block_input = self.current_inputs.pop(block_index)
            residual = output - block_input
            for component, value in (("output", output), ("residual", residual)):
                record = {
                    "step": step,
                    "block": block_index,
                    "component": component,
                    "relative_input_norm": relative_norm(value, block_input),
                }
                key = (block_index, component)
                if key in self.previous:
                    record.update(tensor_similarity(value, self.previous[key]))
                self.records.append(record)
                self.previous[key] = value.detach().clone()
            self.calls[block_index] += 1
        return hook

    def __enter__(self):
        for index, block in enumerate(self.blocks):
            self.handles.append(block.register_forward_pre_hook(self._block_pre_hook(index)))
            self.handles.append(block.attn1.register_forward_hook(self._component_hook(index, "attention")))
            self.handles.append(block.ff.register_forward_hook(self._component_hook(index, "mlp")))
            self.handles.append(block.register_forward_hook(self._block_hook(index)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finalize(self, expected_steps: int) -> list[dict]:
        if self.calls != [expected_steps] * len(self.blocks):
            raise RuntimeError(f"Incomplete block calls: {self.calls}")
        rows = []
        for record in self.records:
            row = {}
            for key, value in record.items():
                row[key] = float(value.detach().cpu()) if torch.is_tensor(value) else value
            rows.append(row)
        return rows


def make_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def run_baseline(pipe, class_id: int, seed: int, steps: int, guidance_scale: float):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = pipe(
            class_labels=[class_id],
            generator=make_generator(seed),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            output_type="np",
        ).images
    torch.cuda.synchronize()
    return output, time.perf_counter() - started, torch.cuda.max_memory_allocated() / 2**20


def run_profile(pipe, class_id: int, seed: int, steps: int, guidance_scale: float):
    profiler = DiTRedundancyProfiler(pipe.transformer)
    with torch.inference_mode(), profiler:
        output = pipe(
            class_labels=[class_id],
            generator=make_generator(seed),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            output_type="np",
        ).images
    return output, profiler.finalize(steps)


def label_for(pipe, class_id: int) -> str:
    matches = sorted(name for name, value in pipe.labels.items() if value == class_id)
    return matches[0] if matches else f"class_{class_id}"


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-ids", type=int, nargs="+", default=list(DEFAULT_CLASS_IDS))
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = DiTPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if not hasattr(pipe.transformer, "transformer_blocks"):
        raise RuntimeError("Expected a Transformer2DModel with transformer_blocks")

    # Warm kernels without contaminating measured runs.
    with torch.inference_mode():
        pipe(class_labels=[args.class_ids[0]], generator=make_generator(args.seed),
             num_inference_steps=2, guidance_scale=args.guidance_scale, output_type="np")
    torch.cuda.synchronize()

    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "experiment": "Experiment 14A DiT adjacent-step redundancy profile",
        "model": MODEL_ID,
        "model_path": str(args.model_path.resolve()),
        "class_ids": args.class_ids,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "dtype": "float16",
        "resolution": 256,
        "gpu": torch.cuda.get_device_name(),
        "block_count": len(pipe.transformer.transformer_blocks),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "diffusers", "transformers", "numpy", "Pillow")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": source_hash,
    }
    save_json(args.output_dir / "manifest.json", manifest)

    summary = []
    for ordinal, class_id in enumerate(args.class_ids, start=1):
        label = label_for(pipe, class_id)
        print(f"profile [{ordinal}/{len(args.class_ids)}] class={class_id} label={label}", flush=True)
        baseline, seconds, peak_mib = run_baseline(
            pipe, class_id, args.seed, args.steps, args.guidance_scale)
        profiled, records = run_profile(
            pipe, class_id, args.seed, args.steps, args.guidance_scale)
        max_abs = float(np.max(np.abs(baseline.astype(np.float64) - profiled.astype(np.float64))))
        exact = bool(np.array_equal(baseline, profiled))
        if not exact:
            raise RuntimeError(f"Read-only hooks changed output for class {class_id}: {max_abs}")
        save_json(args.output_dir / f"class_{class_id:04d}_profile.json", records)
        np.save(args.output_dir / f"class_{class_id:04d}_baseline.npy", baseline)
        row = {
            "class_id": class_id,
            "label": label,
            "seed": args.seed,
            "baseline_seconds": seconds,
            "peak_allocated_mib": peak_mib,
            "profile_records": len(records),
            "read_only_exact_match": exact,
            "max_abs_pixel_difference": max_abs,
        }
        summary.append(row)
        save_json(args.output_dir / "summary.json", summary)
        print(json.dumps(row), flush=True)

    save_json(args.output_dir / "complete.json", {
        "complete": True,
        "cases": len(summary),
        "all_read_only_exact": all(row["read_only_exact_match"] for row in summary),
    })


if __name__ == "__main__":
    main()
