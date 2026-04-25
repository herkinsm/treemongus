"""Run SAM 3 text-prompted segmentation over the All2023 day folders.

Expected layout:
    <ROOT>/2023 day <N>/2023 day <N>/<category>/<session>/RGB/*.bmp|jpg|png

Outputs:
    <OUT>/results.csv                  one row per (image, prompt)
    <OUT>/masks/<prompt>/*.npz         per-image masks (--save-masks)
    <OUT>/overlays/<prompt>/*.jpg      per-image overlays (--save-overlays)
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

DEFAULT_PROMPTS = [
    "apple",
    "branch",
    "trunk",
    "flower in full bloom",
    "flower in partial bloom",
    "leaf",
    "fruitlet",
]


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def sample_indices(n: int, k: int | None) -> list[int]:
    """Pick k evenly-spaced indices from range(n), or all if k is None or k>=n."""
    if k is None or k >= n:
        return list(range(n))
    if k <= 0:
        return []
    # evenly spaced; np.linspace gives endpoints, rounding produces even coverage
    return sorted(set(np.linspace(0, n - 1, num=k).round().astype(int).tolist()))


def find_images(root: Path, only_rgb_folders: bool = True, sample_per_session: int | None = None):
    """Yield (day, category, session, image_path) tuples, evenly sampled per session."""
    root = Path(root)
    for day_dir in sorted(root.glob("2023 day *")):
        inner = day_dir / day_dir.name
        day_root = inner if inner.is_dir() else day_dir
        for category_dir in sorted(p for p in day_root.iterdir() if p.is_dir()):
            for session_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                if only_rgb_folders:
                    rgb_dir = session_dir / "RGB"
                    if not rgb_dir.is_dir():
                        continue
                    search_dirs = [rgb_dir]
                else:
                    search_dirs = [session_dir]
                for sd in search_dirs:
                    imgs = sorted(p for p in sd.iterdir() if p.suffix.lower() in IMAGE_EXTS)
                    if not imgs:
                        continue
                    idxs = sample_indices(len(imgs), sample_per_session)
                    for i in idxs:
                        yield day_dir.name, category_dir.name, session_dir.name, imgs[i]


def make_overlay(img: Image.Image, masks: np.ndarray, boxes: np.ndarray | None,
                 title: str | None = None) -> Image.Image:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)
    ax.imshow(img)
    if masks is not None and len(masks) > 0:
        rng = np.random.default_rng(0)
        for m in masks:
            color = np.concatenate([rng.random(3), [0.45]])
            h, w = m.shape[-2:]
            rgba = np.zeros((h, w, 4))
            rgba[m.astype(bool)] = color
            ax.imshow(rgba)
    if boxes is not None:
        for b in boxes:
            x1, y1, x2, y2 = b
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=1.5))
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.canvas.draw()
    # matplotlib >=3.10 removed tostring_rgb(); use buffer_rgba() and drop A.
    rgba = np.asarray(fig.canvas.buffer_rgba())
    buf = rgba[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(buf)


def to_np(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        t = x.detach().cpu()
        # numpy has no bfloat16/float16 — upcast to float32 before .numpy().
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        return t.numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Depth handling. Thresholds mirror sprayer_pipeline/config.py:
#   CANOPY_DEPTH_MIN_MM = 600, CANOPY_DEPTH_MAX_MM = 3000
# Depth .txt files are ASCII uint16 mm, stored 480 rows x 50 cols for a
# 640x480 RGB frame (horizontally decimated ~12.8x). 0 means invalid.
# ---------------------------------------------------------------------------
def depth_path_for(img_path: Path) -> Path:
    """Given .../<session>/RGB/<stem>-RGB[-BP].<ext>, return the matching
    .../<session>/depth/<stem>-Depth.txt."""
    session_dir = img_path.parent.parent  # drop RGB/
    stem = img_path.stem
    for suffix in ("-RGB-BP", "-RGB-bp", "-RGB", "-rgb"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return session_dir / "depth" / f"{stem}-Depth.txt"


def load_depth_mm(depth_path: Path, target_hw: tuple[int, int]) -> np.ndarray | None:
    """Load an ASCII depth .txt and return uint16 mm upsampled to target (H, W).
    Returns None if the file is missing or malformed."""
    if not depth_path.is_file():
        return None
    try:
        d = np.loadtxt(depth_path, dtype=np.int32)
    except Exception:
        return None
    if d.ndim != 2 or d.size == 0:
        return None
    H, W = target_hw
    if d.shape != (H, W):
        # Bilinear upsample via PIL in float32 mode.
        pim = Image.fromarray(d.astype(np.float32), mode="F").resize((W, H), Image.BILINEAR)
        d = np.asarray(pim)
    d = np.clip(d, 0, 65535).astype(np.uint16)
    return d


def near_frac_per_mask(masks_np: np.ndarray, depth_mm: np.ndarray,
                        min_mm: int, max_mm: int) -> list[float]:
    """For each mask, fraction of its pixels whose depth is in [min_mm, max_mm].
    NaN for empty masks."""
    near = (depth_mm >= min_mm) & (depth_mm <= max_mm)
    out: list[float] = []
    for m in masks_np:
        mb = m.astype(bool)
        if mb.ndim == 3:
            mb = mb.any(axis=0)
        total = int(mb.sum())
        if total == 0:
            out.append(float("nan"))
            continue
        out.append(float((mb & near).sum()) / float(total))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\matth\OneDrive\Desktop\Postdoc\Image\All2023")
    ap.add_argument("--out", default=r"C:\Users\matth\OneDrive\Desktop\Postdoc\Image\sam3_out")
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS,
                    help="One or more text prompts. Default: apple, branch, trunk, "
                         "flower in full bloom, flower in partial bloom, leaf, fruitlet.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sample-per-session", type=int, default=20,
                    help="How many evenly-spaced frames to analyze per session "
                         "(captures start/middle/end). Use 0 for all frames.")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--all-folders", action="store_true",
                    help="Search every leaf folder, not just RGB/.")
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Depth-based background filter (matches sprayer_pipeline/config.py).
    ap.add_argument("--depth", action="store_true",
                    help="Load matched depth .txt files and annotate each detection "
                         "with its near-field fraction in the [min,max] mm band.")
    ap.add_argument("--depth-min-mm", type=int, default=600,
                    help="Lower bound of canopy depth band (default 600; sprayer_pipeline).")
    ap.add_argument("--depth-max-mm", type=int, default=3000,
                    help="Upper bound of canopy depth band (default 3000; sprayer_pipeline).")
    ap.add_argument("--depth-near-frac", type=float, default=0.5,
                    help="Count a detection as near-field if >= this fraction of its "
                         "mask pixels lie inside the depth band (default 0.5).")
    args = ap.parse_args()

    sample = None if args.sample_per_session == 0 else args.sample_per_session
    prompts = list(args.prompts)
    prompt_slugs = {p: slugify(p) for p in prompts}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    if args.save_masks:
        masks_dir.mkdir(exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(exist_ok=True)

    print(f"[init] device={args.device} threshold={args.threshold} sample/session={sample}")
    print(f"[init] prompts: {prompts}")
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # Let SAM 3 resolve its own BPE vocab via pkg_resources — the file ships
    # inside the package at sam3/assets/, not the repo's top-level assets/.
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.threshold)

    csv_path = out_dir / "results.csv"
    fieldnames = ["day", "category", "session", "image", "prompt",
                  "n_detections", "mean_score", "max_score", "elapsed_s",
                  "n_near", "near_frac_mean", "near_frac_max"]
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    total_imgs = 0
    start = time.time()
    try:
        for day, category, session, img_path in find_images(
            Path(args.root),
            only_rgb_folders=not args.all_folders,
            sample_per_session=sample,
        ):
            if args.max_images is not None and total_imgs >= args.max_images:
                break
            t_img = time.time()
            try:
                img = Image.open(img_path).convert("RGB")
                state = processor.set_image(img)  # encode once

                # Optional depth load (once per image, reused across prompts).
                depth_mm = None
                if args.depth:
                    depth_mm = load_depth_mm(depth_path_for(img_path), (img.height, img.width))

                for prompt in prompts:
                    t0 = time.time()
                    processor.reset_all_prompts(state)
                    out = processor.set_text_prompt(state=state, prompt=prompt)
                    masks_np = to_np(out.get("masks"))
                    boxes_np = to_np(out.get("boxes"))
                    scores_np = to_np(out.get("scores"))

                    # SAM 3 returns masks as (N, 1, H, W); squeeze the channel
                    # dim so every consumer below sees a clean (N, H, W).
                    if masks_np is not None and masks_np.ndim == 4 and masks_np.shape[1] == 1:
                        masks_np = masks_np.squeeze(1)

                    n = 0 if masks_np is None else len(masks_np)
                    mean_s = float(np.mean(scores_np)) if scores_np is not None and len(scores_np) else 0.0
                    max_s = float(np.max(scores_np)) if scores_np is not None and len(scores_np) else 0.0

                    # Depth-based near-field stats (skipped when --depth off or file missing).
                    n_near: int | str = ""
                    near_mean: float | str = ""
                    near_max: float | str = ""
                    if args.depth and depth_mm is not None and n > 0:
                        fracs = near_frac_per_mask(
                            masks_np, depth_mm, args.depth_min_mm, args.depth_max_mm
                        )
                        fracs_arr = np.asarray(fracs, dtype=float)
                        valid = fracs_arr[~np.isnan(fracs_arr)]
                        if valid.size:
                            n_near = int((valid >= args.depth_near_frac).sum())
                            near_mean = round(float(valid.mean()), 4)
                            near_max = round(float(valid.max()), 4)

                    slug = prompt_slugs[prompt]
                    if args.save_masks and n > 0:
                        rel = img_path.relative_to(args.root).with_suffix(".npz")
                        mp = masks_dir / slug / rel
                        mp.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(mp, masks=masks_np.astype(bool),
                                            scores=scores_np, boxes=boxes_np)
                    if args.save_overlays and n > 0:
                        rel = img_path.relative_to(args.root).with_suffix(".jpg")
                        op = overlays_dir / slug / rel
                        op.parent.mkdir(parents=True, exist_ok=True)
                        overlay = make_overlay(img, masks_np, boxes_np, title=f"{prompt} (n={n})")
                        overlay.save(op, quality=85)

                    writer.writerow({
                        "day": day, "category": category, "session": session,
                        "image": str(img_path), "prompt": prompt,
                        "n_detections": n,
                        "mean_score": round(mean_s, 4),
                        "max_score": round(max_s, 4),
                        "elapsed_s": round(time.time() - t0, 3),
                        "n_near": n_near,
                        "near_frac_mean": near_mean,
                        "near_frac_max": near_max,
                    })
                total_imgs += 1
                if total_imgs % 10 == 0:
                    f.flush()
                    rate = total_imgs / (time.time() - start)
                    print(f"[{total_imgs}] {day}/{category}/{session}/{img_path.name} "
                          f"({time.time()-t_img:.2f}s/img, {rate:.2f} img/s overall)")
            except Exception as e:
                print(f"[ERR] {img_path}: {e}", file=sys.stderr)
    finally:
        f.close()

    dt = time.time() - start
    print(f"[done] {total_imgs} images × {len(prompts)} prompts in {dt:.1f}s -> {csv_path}")


if __name__ == "__main__":
    main()
