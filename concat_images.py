#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def concat_pair(path_a: Path, path_b: Path, margin: int = 16, resize_height: bool = True) -> Image.Image:
    """Concatenate two images side-by-side with an optional center margin.
       If resize_height=True (default), both images are resized to the same height (the smaller one) preserving aspect ratio.
       If resize_height=False, images are padded to the same height (no scaling)."""

    # Open and fix EXIF orientation, keep alpha if present
    img_a = ImageOps.exif_transpose(Image.open(path_a).convert("RGBA"))
    img_b = ImageOps.exif_transpose(Image.open(path_b).convert("RGBA"))

    if resize_height:
        target_h = min(img_a.height, img_b.height)  # avoid upscaling
        if img_a.height != target_h:
            new_w = int(round(img_a.width * target_h / img_a.height))
            img_a = img_a.resize((new_w, target_h), Image.Resampling.LANCZOS)
        if img_b.height != target_h:
            new_w = int(round(img_b.width * target_h / img_b.height))
            img_b = img_b.resize((new_w, target_h), Image.Resampling.LANCZOS)
        canvas_h = target_h
    else:
        # pad to same height (no scaling)
        canvas_h = max(img_a.height, img_b.height)
        def pad_to_h(im: Image.Image) -> Image.Image:
            if im.height == canvas_h:
                return im
            out = Image.new("RGBA", (im.width, canvas_h), (0, 0, 0, 0))
            top = (canvas_h - im.height) // 2
            out.paste(im, (0, top))
            return out
        img_a = pad_to_h(img_a)
        img_b = pad_to_h(img_b)

    out_w = img_a.width + margin + img_b.width
    out = Image.new("RGBA", (out_w, canvas_h), (255, 255, 255, 0))  # transparent background
    out.paste(img_a, (0, (canvas_h - img_a.height) // 2))
    out.paste(img_b, (img_a.width + margin, (canvas_h - img_b.height) // 2))
    return out

def main():
    ap = argparse.ArgumentParser(description="Concatenate same-named images from two folders side-by-side.")
    ap.add_argument("--dirA", required=True, type=Path, help="Directory A (left image).")
    ap.add_argument("--dirB", required=True, type=Path, help="Directory B (right image).")
    ap.add_argument("--out", required=True, type=Path, help="Output directory.")
    ap.add_argument("--margin", type=int, default=16, help="Center margin in pixels (default: 16).")
    ap.add_argument("--no-resize", action="store_true",
                    help="Do NOT resize; pad to same height instead (default is to resize both to min height).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    files_a = {p.name: p for p in args.dirA.iterdir() if is_image(p)}
    files_b = {p.name: p for p in args.dirB.iterdir() if is_image(p)}
    common = sorted(set(files_a.keys()) & set(files_b.keys()))

    if not common:
        raise SystemExit("No common image filenames found between the two directories.")

    for name in tqdm(common, desc="Concatenating"):
        out_img = concat_pair(files_a[name], files_b[name],
                              margin=args.margin, resize_height=not args.no_resize)

        # Save using the same filename; convert to RGB if saving as JPEG
        out_path = args.out / name
        ext = out_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            out_img = out_img.convert("RGB")
            out_img.save(out_path, quality=95, subsampling=1, optimize=True)
        else:
            out_img.save(out_path)

if __name__ == "__main__":
    main()
