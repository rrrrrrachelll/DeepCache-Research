"""Encode the frozen SD 1.5 CLIP prompt representation for experiment 11."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer

from prompt_set import build_prompts, prompt_hash


MODEL_ID = "runwayml/stable-diffusion-v1-5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(MODEL_ID, local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(
        model_path, subfolder="tokenizer", local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=torch.float16,
        local_files_only=True).to("cuda").eval()
    prompts = build_prompts()
    embeddings = {}
    with torch.inference_mode():
        for start in range(0, len(prompts), args.batch_size):
            rows = prompts[start:start + args.batch_size]
            tokens = tokenizer(
                [row["prompt"] for row in rows], padding="max_length",
                max_length=tokenizer.model_max_length, truncation=True,
                return_tensors="pt")
            outputs = encoder(tokens.input_ids.to("cuda"))
            pooled = outputs.pooler_output.float().cpu()
            for row, value in zip(rows, pooled):
                embeddings[row["id"]] = value

    artifact = {
        "model": MODEL_ID,
        "model_path": model_path,
        "encoder": "CLIPTextModel pooler_output (frozen)",
        "dimension": next(iter(embeddings.values())).numel(),
        "prompt_hash": prompt_hash(),
        "count": len(embeddings),
        "transformers_version": importlib.metadata.version("transformers"),
        "embeddings": embeddings,
    }
    torch.save(artifact, args.output)
    print(json.dumps({key: value for key, value in artifact.items()
                      if key != "embeddings"}, indent=2))


if __name__ == "__main__":
    main()
