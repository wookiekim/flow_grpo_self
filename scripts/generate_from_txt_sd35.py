#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
import json
from typing import List, Tuple, Optional

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model, PeftModel

import torch.distributed as dist

# --------------------------- Distributed helpers ---------------------------

def setup_distributed_if_needed() -> Tuple[int, int, int, Optional[torch.device]]:
    """Initialize torch.distributed only when WORLD_SIZE>1. Returns (rank, world_size, local_rank, device)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12356")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0

# ------------------------------- I/O helpers -------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_subdir(name: str) -> str:
    # Keep it filesystem-friendly
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name.strip()) or "uncategorized"


def chunked(seq: List, n: int):
    """Yield successive chunks of size n from seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def load_prompts_txt(txt_path: str) -> List[str]:
    """Load prompts from a .txt file; 1 prompt per line. Blank lines ignored."""
    prompts: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                prompts.append(p)
    return prompts


def sanitize_filename_from_prompt(prompt: str, max_len: int = 120) -> str:
    """
    Convert prompt to a safe filename:
      - spaces -> underscores (requested behavior)
      - remove/replace unsafe characters
      - truncate to max_len (without extension)
    """
    # replace whitespace with underscores first (explicit requirement)
    name = re.sub(r"\s+", "_", prompt)
    # then sanitize other unsafe characters
    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)
    # collapse multiple underscores
    name = re.sub(r"_+", "_", name).strip("._")
    if not name:
        name = "image"
    if len(name) > max_len:
        name = name[:max_len].rstrip("._-")
    return name


def unique_path_with_suffix(base_dir: str, base_name: str, ext: str = ".jpg") -> str:
    """
    Ensure filename uniqueness by appending _1, _2, ... if needed.
    """
    candidate = os.path.join(base_dir, base_name + ext)
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        cand = os.path.join(base_dir, f"{base_name}_{i}{ext}")
        if not os.path.exists(cand):
            return cand
        i += 1

# ------------------------------- LoRA helpers ------------------------------

def attach_lora_to_sd3(
    pipeline: StableDiffusion3Pipeline,
    lora_hf_path: str = "",
    local_checkpoint_dir: str = "",
    init_strategy: str = "gaussian",
):
    """
    Two options:
    1) lora_hf_path: HuggingFace repo/path with PEFT weights → merge into transformer.
    2) local_checkpoint_dir: folder containing 'lora' subdir with PEFT adapter → load (no merge).
    """
    if lora_hf_path:
        # Merge-and-unload for fastest inference
        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, lora_hf_path)
        pipeline.transformer = pipeline.transformer.merge_and_unload()
        return

    if local_checkpoint_dir:
        lora_dir = os.path.join(local_checkpoint_dir, "lora")
        if not os.path.isdir(lora_dir):
            raise FileNotFoundError(
                f"Local LoRA directory not found: {lora_dir}. Expected '{local_checkpoint_dir}/lora'."
            )
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        lora_cfg = LoraConfig(r=32, lora_alpha=64, init_lora_weights=init_strategy, target_modules=target_modules)
        pipeline.transformer = get_peft_model(pipeline.transformer, lora_cfg)
        pipeline.transformer.load_adapter(lora_dir, adapter_name="default", is_trainable=False)
        try:
            pipeline.transformer.set_adapter("default")
        except Exception:
            pass  # Optional in many PEFT versions

# ------------------------------- Main logic --------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate SD3.5 images for prompts listed in a .txt file (one per line).")
    # Model / LoRA
    parser.add_argument("--sd35_repo", type=str, default="stabilityai/stable-diffusion-3.5-medium",
                        help="Base SD3.5 model repo or local path.")
    parser.add_argument("--lora_hf_path", type=str, default="",
                        help="Optional HuggingFace LoRA path to merge for inference.")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Optional local checkpoint path containing a 'lora' subdir.")
    # Data / output
    parser.add_argument("--prompts_txt", type=str, required=True, help="Path to .txt file containing prompts (1 per line).")
    parser.add_argument("--output_dir", type=str, default="./sd35_outputs", help="Output directory.")
    parser.add_argument("--manifest_name", type=str, default="manifest.jsonl",
                        help="Filename for metadata/paths written to output_dir.")
    # Sampling
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345, help="-1 for random.")
    # Precision
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--disable_safety_checker", action="store_true")
    args = parser.parse_args()

    # Distributed
    rank, world_size, local_rank, device = setup_distributed_if_needed()
    main_proc = is_main_process(rank)

    # Mixed precision
    mp_dtype = None
    if args.mixed_precision == "fp16":
        mp_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        mp_dtype = torch.bfloat16

    # Env tweaks
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if main_proc:
        print(f"Device: {device}, World size: {world_size}")
        print(f"Loading model: {args.sd35_repo}")

    # Load pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(args.sd35_repo)

    # Attach LoRA if requested
    if args.lora_hf_path or args.checkpoint_path:
        if main_proc:
            print("Attaching LoRA...")
        attach_lora_to_sd3(
            pipeline,
            lora_hf_path=args.lora_hf_path,
            local_checkpoint_dir=args.checkpoint_path,
            init_strategy="gaussian",
        )

    # Move to device + dtypes
    txt_dtype = mp_dtype if mp_dtype is not None else torch.float32
    pipeline.transformer.to(device, dtype=txt_dtype)
    pipeline.vae.to(device, dtype=torch.float32)  # VAE often prefers fp32 for stability
    pipeline.text_encoder.to(device, dtype=txt_dtype)
    pipeline.text_encoder_2.to(device, dtype=txt_dtype)
    pipeline.text_encoder_3.to(device, dtype=txt_dtype)
    if args.disable_safety_checker:
        pipeline.safety_checker = None
    pipeline.set_progress_bar_config(disable=not main_proc)

    # Output dirs
    images_root = os.path.join(args.output_dir, "images")
    ensure_dir(args.output_dir)
    ensure_dir(images_root)

    # Load prompts
    all_prompts = load_prompts_txt(args.prompts_txt)
    total_items = len(all_prompts)
    if main_proc:
        print(f"Loaded {total_items} prompts from {args.prompts_txt}")

    # Shard across ranks
    if world_size > 1:
        all_prompts = all_prompts[rank::world_size]
        if main_proc:
            print(f"Rank {rank} will process {len(all_prompts)} items.")

    # RNG for reproducibility
    if args.seed >= 0:
        base_seed = args.seed + rank
        generator = torch.Generator(device=device)
        generator.manual_seed(base_seed)
    else:
        generator = None

    # Generation loop
    manifest_path = os.path.join(args.output_dir, args.manifest_name)
    # Each rank writes its own partial manifest; rank0 will keep its own file.
    manifest_file = open(manifest_path if main_proc else f"{manifest_path}.rank{rank}", "w", encoding="utf-8")

    height = width = args.resolution
    pipe_kwargs = dict(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=height,
        width=width,
        output_type="pil",
    )
    if generator is not None:
        pipe_kwargs["generator"] = generator

    with torch.no_grad():
        # mixed precision autocast for speed/memory
        autocast_enabled = mp_dtype is not None and device.type == "cuda"
        autocast_dtype = mp_dtype if autocast_enabled else torch.float32
        progress = tqdm(total=len(all_prompts), disable=not main_proc, desc="Generating")

        for idx_chunk in chunked(list(range(len(all_prompts))), args.batch_size):
            batch_prompts = [all_prompts[i] for i in idx_chunk]

            with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
                images: List[Image.Image] = pipeline(batch_prompts, **pipe_kwargs).images  # type: ignore

            # Save
            for prompt_text, img in zip(batch_prompts, images):
                base_name = sanitize_filename_from_prompt(prompt_text)
                save_path = unique_path_with_suffix(images_root, base_name, ext=".jpg")
                img.save(save_path, format="JPEG", quality=95)

                record = {"prompt": prompt_text, "path": save_path}
                manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Free GPU memory between batches
            del images
            if device.type == "cuda":
                torch.cuda.empty_cache()

            progress.update(len(idx_chunk))

        progress.close()

    manifest_file.close()

    if main_proc:
        print(f"\nDone. Images saved under: {images_root}")
        print(f"Manifest: {manifest_path}")
        if dist.is_available() and dist.is_initialized():
            extra = [f"{manifest_path}.rank{r}" for r in range(1, int(os.environ.get('WORLD_SIZE', '1')))]
            if extra:
                print("Additional rank manifests:", ", ".join(extra))

    cleanup_distributed()


if __name__ == "__main__":
    main()
