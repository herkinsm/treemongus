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
    "flower",
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


# ---------------------------------------------------------------------------
# Cross-frame instance tracker (greedy IoU). One per (session, prompt). Each
# persistent track == one physical object (flower / apple / etc.). A track
# that is observed for >= min_frames frames is counted as a "unique" instance.
# ---------------------------------------------------------------------------
class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: dict[int, dict] = {}
        self.next_id = 0
        self.frame = 0

    @staticmethod
    def _iou(a, b) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return float(inter) / float(union) if union > 0 else 0.0

    def step(self, boxes, scores) -> list[int]:
        """Update with one frame's detections. Returns track_id per detection."""
        self.frame += 1
        if boxes is None or len(boxes) == 0:
            return []

        # Active tracks = those last seen within max_age frames.
        active = [(tid, t) for tid, t in self.tracks.items()
                  if self.frame - t["last_frame"] <= self.max_age + 1]
        # Match high-confidence tracks first (greedy, but priority-ordered).
        active.sort(key=lambda x: -x[1]["max_score"])

        n = len(boxes)
        det_to_tid = [-1] * n
        used = [False] * n

        for tid, t in active:
            best_iou = self.iou_threshold
            best_d = -1
            for d in range(n):
                if used[d]:
                    continue
                iou = self._iou(t["bbox"], boxes[d])
                if iou > best_iou:
                    best_iou = iou
                    best_d = d
            if best_d >= 0:
                used[best_d] = True
                det_to_tid[best_d] = tid
                t["bbox"] = boxes[best_d]
                t["last_frame"] = self.frame
                t["n_frames"] += 1
                s = float(scores[best_d]) if scores is not None else 0.0
                if s > t["max_score"]:
                    t["max_score"] = s

        for d in range(n):
            if used[d]:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "bbox": tuple(float(x) for x in boxes[d]),
                "first_frame": self.frame,
                "last_frame": self.frame,
                "n_frames": 1,
                "max_score": float(scores[d]) if scores is not None else 0.0,
            }
            det_to_tid[d] = tid

        return det_to_tid

    def n_unique(self, min_frames: int = 1) -> int:
        return sum(1 for t in self.tracks.values() if t["n_frames"] >= min_frames)

    def summary(self) -> list[dict]:
        return [{"track_id": tid, **t} for tid, t in self.tracks.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\matth\OneDrive\Desktop\Postdoc\Image\All2023")
    ap.add_argument("--out", default=r"C:\Users\matth\OneDrive\Desktop\Postdoc\Image\sam3_out")
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS,
                    help="One or more text prompts. Default: apple, branch, trunk, "
                         "flower, leaf, fruitlet.")
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
    # Cross-frame instance tracking (per-session IoU tracker).
    ap.add_argument("--track", action="store_true",
                    help="Track detections across frames within each session via IoU "
                         "matching, so a single physical flower/apple is counted once. "
                         "REQUIRES dense temporal sampling — set --sample-per-session 0 "
                         "(or a large value) so consecutive frames are processed.")
    ap.add_argument("--track-iou", type=float, default=0.3,
                    help="IoU threshold to associate a detection with an existing track.")
    ap.add_argument("--track-max-age", type=int, default=3,
                    help="A track lost for more than this many frames is closed.")
    ap.add_argument("--track-min-frames", type=int, default=2,
                    help="Only count a track as a 'unique instance' if seen in >= this "
                         "many frames (filters one-frame false positives).")
    ap.add_argument("--track-prompts", nargs="+", default=None,
                    help="Subset of --prompts to actually track. Defaults to all "
                         "flower-named prompts (so apples/leaves/etc. are detected per "
                         "frame but not deduped across frames).")
    args = ap.parse_args()

    sample = None if args.sample_per_session == 0 else args.sample_per_session
    prompts = list(args.prompts)
    prompt_slugs = {p: slugify(p) for p in prompts}

    # Resolve which prompts the IoU tracker should run on. Default: any prompt
    # whose text contains "flower". User can override with --track-prompts.
    if args.track_prompts is not None:
        tracked_prompts = [p for p in args.track_prompts if p in prompts]
        unknown = [p for p in args.track_prompts if p not in prompts]
        if unknown:
            print(f"[warn] --track-prompts entries not in --prompts (ignored): {unknown}",
                  file=sys.stderr)
    else:
        tracked_prompts = [p for p in prompts if "flower" in p.lower()]
    tracked_set = set(tracked_prompts)
    if args.track:
        print(f"[init] tracking prompts: {tracked_prompts}")

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
    # n_detections is the post-filter count (after --depth), n_raw is always
    # the SAM 3 raw count. near_frac_* describe the kept detections.
    fieldnames = ["day", "category", "session", "image", "prompt",
                  "n_detections", "n_raw", "mean_score", "max_score", "elapsed_s",
                  "near_frac_mean", "near_frac_max", "track_ids"]
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # Accumulator for the per-image wide-format CSV: image_path -> {
    #   "day": str, "category": str, "session": str,
    #   "counts": {prompt: n_detections, ...}}
    image_aggregates: dict[str, dict] = {}

    # Per-(session, prompt) IoU trackers. Keyed by session_key; each value is
    # a dict {prompt: IoUTracker}. Trackers are flushed to track_summaries at
    # session boundaries so memory stays bounded.
    track_summaries: list[dict] = []   # one per (session, prompt)
    track_details: list[dict] = []     # one per individual track

    def flush_session_trackers(session_key, trackers):
        # trackers may not include every prompt (only tracked_set was created),
        # which is fine — we just emit summaries for the prompts present.
        d, c, s = session_key
        for p, tr in trackers.items():
            track_summaries.append({
                "day": d, "category": c, "session": s, "prompt": p,
                "n_unique": tr.n_unique(min_frames=args.track_min_frames),
                "n_unique_any": tr.n_unique(min_frames=1),
                "n_tracks_total": len(tr.tracks),
            })
            for trk in tr.summary():
                track_details.append({
                    "day": d, "category": c, "session": s, "prompt": p,
                    "track_id": trk["track_id"],
                    "first_frame": trk["first_frame"],
                    "last_frame": trk["last_frame"],
                    "n_frames": trk["n_frames"],
                    "max_score": round(trk["max_score"], 4),
                })

    current_session_key = None
    current_trackers: dict[str, IoUTracker] = {}

    if args.track and (sample is not None and sample < 50):
        print(f"[warn] --track with --sample-per-session {sample} produces "
              f"too-sparse frames for IoU association; pass "
              f"--sample-per-session 0 for dense tracking.", file=sys.stderr)

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
            # Detect session change → flush trackers for the previous session
            # and start fresh ones for the new session.
            session_key = (day, category, session)
            if args.track and session_key != current_session_key:
                if current_session_key is not None:
                    flush_session_trackers(current_session_key, current_trackers)
                current_trackers = {p: IoUTracker(args.track_iou, args.track_max_age)
                                    for p in tracked_prompts}
                current_session_key = session_key

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

                    n_raw = 0 if masks_np is None else len(masks_np)
                    n = n_raw
                    mean_s = float(np.mean(scores_np)) if scores_np is not None and len(scores_np) else 0.0
                    max_s = float(np.max(scores_np)) if scores_np is not None and len(scores_np) else 0.0

                    # Depth-based near-field filter. When --depth is on, we
                    # actively drop detections whose mask doesn't sit inside
                    # the canopy band (background trees, far ground, etc.).
                    near_mean: float | str = ""
                    near_max: float | str = ""
                    if args.depth and depth_mm is not None and n_raw > 0:
                        fracs = near_frac_per_mask(
                            masks_np, depth_mm, args.depth_min_mm, args.depth_max_mm
                        )
                        fracs_arr = np.asarray(fracs, dtype=float)
                        keep = ~np.isnan(fracs_arr) & (fracs_arr >= args.depth_near_frac)
                        if not keep.all():
                            masks_np = masks_np[keep]
                            if scores_np is not None:
                                scores_np = scores_np[keep]
                            if boxes_np is not None:
                                boxes_np = boxes_np[keep]
                            n = int(keep.sum())
                            mean_s = (float(np.mean(scores_np))
                                      if scores_np is not None and len(scores_np) else 0.0)
                            max_s = (float(np.max(scores_np))
                                     if scores_np is not None and len(scores_np) else 0.0)
                        kept_fracs = fracs_arr[keep]
                        if kept_fracs.size:
                            near_mean = round(float(kept_fracs.mean()), 4)
                            near_max = round(float(kept_fracs.max()), 4)

                    # Tracker step (after depth filter, so we only track real
                    # near-field objects). Only runs for prompts in tracked_set.
                    track_ids: list[int] = []
                    if (args.track and prompt in tracked_set
                            and current_trackers and n > 0 and boxes_np is not None):
                        track_ids = current_trackers[prompt].step(boxes_np, scores_np)

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
                        "n_raw": n_raw,
                        "mean_score": round(mean_s, 4),
                        "max_score": round(max_s, 4),
                        "elapsed_s": round(time.time() - t0, 3),
                        "near_frac_mean": near_mean,
                        "near_frac_max": near_max,
                        "track_ids": ";".join(str(t) for t in track_ids) if track_ids else "",
                    })

                    # Per-image aggregator for wide CSV.
                    img_key = str(img_path)
                    agg = image_aggregates.setdefault(img_key, {
                        "day": day, "category": category, "session": session,
                        "counts": {},
                    })
                    agg["counts"][prompt] = n
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

    # Flush the final session's trackers.
    if args.track and current_session_key is not None:
        flush_session_trackers(current_session_key, current_trackers)

    dt = time.time() - start
    print(f"[done] {total_imgs} images × {len(prompts)} prompts in {dt:.1f}s -> {csv_path}")

    # ---------- wide CSV: one row per image, columns per prompt ----------
    flower_prompts = [p for p in prompts if "flower" in p.lower()]
    wide_path = out_dir / "results_wide.csv"
    wide_fields = ["day", "category", "session", "image"]
    wide_fields += [f"n_{prompt_slugs[p]}" for p in prompts]
    if flower_prompts:
        wide_fields.append("n_flowers_total")
    with open(wide_path, "w", newline="", encoding="utf-8") as wf:
        ww = csv.DictWriter(wf, fieldnames=wide_fields)
        ww.writeheader()
        for img_key, agg in image_aggregates.items():
            row = {
                "day": agg["day"],
                "category": agg["category"],
                "session": agg["session"],
                "image": img_key,
            }
            for p in prompts:
                row[f"n_{prompt_slugs[p]}"] = agg["counts"].get(p, 0)
            if flower_prompts:
                row["n_flowers_total"] = sum(
                    agg["counts"].get(p, 0) for p in flower_prompts
                )
            ww.writerow(row)
    print(f"[done] wide summary -> {wide_path}")

    # ---------- per-session unique tracks (instance dedup) ----------
    if args.track and track_summaries:
        # Only the prompts we actually tracked appear here.
        prompt_to_slug = prompt_slugs
        tracked_flower_prompts = [p for p in tracked_prompts if "flower" in p.lower()]
        ts_path = out_dir / "tracks_summary.csv"
        ts_fields = ["day", "category", "session"]
        ts_fields += [f"n_unique_{prompt_to_slug[p]}" for p in tracked_prompts]
        if tracked_flower_prompts:
            ts_fields.append("n_unique_flowers_total")
        # Group track_summaries by session.
        by_session: dict[tuple, dict] = {}
        for r in track_summaries:
            key = (r["day"], r["category"], r["session"])
            row = by_session.setdefault(key, {
                "day": r["day"], "category": r["category"], "session": r["session"],
            })
            row[f"n_unique_{prompt_to_slug[r['prompt']]}"] = r["n_unique"]
        with open(ts_path, "w", newline="", encoding="utf-8") as tf:
            tw = csv.DictWriter(tf, fieldnames=ts_fields)
            tw.writeheader()
            for key, row in by_session.items():
                # Default to 0 for any missing prompt column (only tracked prompts).
                for p in tracked_prompts:
                    row.setdefault(f"n_unique_{prompt_to_slug[p]}", 0)
                if tracked_flower_prompts:
                    row["n_unique_flowers_total"] = sum(
                        row.get(f"n_unique_{prompt_to_slug[p]}", 0)
                        for p in tracked_flower_prompts
                    )
                tw.writerow(row)
        print(f"[done] per-session unique tracks -> {ts_path}")

        # Per-track diagnostics CSV.
        td_path = out_dir / "tracks_detail.csv"
        td_fields = ["day", "category", "session", "prompt", "track_id",
                     "first_frame", "last_frame", "n_frames", "max_score"]
        with open(td_path, "w", newline="", encoding="utf-8") as df_:
            dw = csv.DictWriter(df_, fieldnames=td_fields)
            dw.writeheader()
            for r in track_details:
                dw.writerow(r)
        print(f"[done] per-track detail -> {td_path}")


if __name__ == "__main__":
    main()
