"""Assemble SAM-3 detections into a YOLO-format training dataset.

Reads:
    <in>/results.csv               per-frame counts and (when --track is on)
                                   semicolon-joined per-detection track IDs
    <in>/tracks_detail.csv         per-track stats (n_frames, max_score, ...)
                                   used to filter to STABLE tracks only
    <in>/masks/flower/.../*.npz    per-frame masks + boxes from --save-masks

Writes:
    <out>/images/{train,val}/<flat_name>.jpg     RGB frame copies
    <out>/labels/{train,val}/<flat_name>.txt     YOLO labels:
                                                   class cx cy w h  (normalized)
    <out>/dataset.yaml             Ultralytics YOLO config

Design choices:
  - Single class, "flower" (class 0). YOLO treats the multi-prompt
    flower / blossom / apple-blossom variations as one class.
  - Filenames are flattened (day__category__session__stem.jpg) so YOLO
    doesn't trip on whitespace and slashes. The original path is recoverable
    from the slug.
  - Train/val split is per-FRAME, not per-session. With --val-frac 0.2,
    20% of all frames go to val. Stratified by session if --stratify is on.
  - Only tracks with n_frames >= --min-track-frames are exported. A
    1-frame "track" is almost always a false positive; >=3 frames means
    the IoU tracker saw the flower in three consecutive frames at least.
  - Boxes are read straight from the .npz that --save-masks wrote --
    those have already passed every quality gate in analyze_days.py.

Usage:
    python make_yolo_dataset.py --in $HOME/sam3_yolo_labels \\
        --out $HOME/yolo_dataset --val-frac 0.2 --min-track-frames 3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert SAM-3 detection output into a YOLO-format dataset.",
    )
    ap.add_argument("--in", dest="in_dir", required=True,
                    help="Input directory: where analyze_days.py wrote "
                         "results.csv, tracks_detail.csv, and masks/.")
    ap.add_argument("--out", dest="out_dir", required=True,
                    help="Output directory for the YOLO dataset.")
    ap.add_argument("--root", default=None,
                    help="Original RGB image root (e.g. /fs/scratch/PAS0228). "
                         "If omitted, parsed from the first results.csv row.")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="Fraction of frames assigned to validation split "
                         "(default 0.2 = 80/20 train/val).")
    ap.add_argument("--min-track-frames", type=int, default=3,
                    help="Drop boxes whose track was seen in fewer than N "
                         "frames. 1-frame tracks are usually noise. Default 3.")
    ap.add_argument("--stratify", action="store_true",
                    help="Split train/val per-session (each session "
                         "contributes the same val-frac). Without this, "
                         "the split is uniform random across all frames.")
    ap.add_argument("--min-box-px", type=int, default=8,
                    help="Drop boxes smaller than this many pixels on a "
                         "side (degenerate / sub-pixel detections).")
    ap.add_argument("--copy-images", action="store_true",
                    help="Copy the source BMPs into <out>/images/. "
                         "Default is to symlink (saves disk).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for the train/val split.")
    return ap.parse_args()


def read_kept_track_ids(
    tracks_detail_csv: Path, min_frames: int,
) -> set[tuple[str, str, str, str, int]]:
    """Return the set of (day, category, session, prompt, track_id) tuples
    whose track was seen in >= min_frames consecutive frames. A track that
    appears in only 1-2 frames is almost always a false positive (glint,
    motion artifact, transient lighting); a stable apple blossom appears
    in many consecutive frames as the camera moves past it."""
    kept: set[tuple[str, str, str, str, int]] = set()
    if not tracks_detail_csv.is_file():
        print(
            f"[warn] {tracks_detail_csv} not found -- including ALL "
            f"detections (no track-stability filter).",
            file=sys.stderr,
        )
        return kept  # empty -> caller treats as "no filter"
    with open(tracks_detail_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n_frames = int(row["n_frames"])
            except (KeyError, ValueError):
                continue
            if n_frames < min_frames:
                continue
            kept.add((
                row["day"], row["category"], row["session"],
                row["prompt"], int(row["track_id"]),
            ))
    return kept


def slugify(s: str) -> str:
    """Filesystem-safe single-token version of a path component. Replaces
    whitespace with underscores and strips characters that confuse YOLO's
    file walker."""
    return "".join(
        ch if ch.isalnum() or ch in "-_." else "_"
        for ch in s.strip()
    )


def find_npz_for_frame(
    masks_root: Path, day: str, category: str, session: str,
    img_stem: str,
) -> Path | None:
    """Locate the .npz mask file for a given frame. analyze_days.py writes
    them under <out>/masks/<prompt_slug>/<rel_path>.npz where rel_path is
    img_path.relative_to(args.root). For the YOLO labeler we only need the
    'flower' slug -- it's the only prompt the run targets."""
    candidates = list(
        masks_root.rglob(f"{img_stem}.npz")
    )
    if not candidates:
        return None
    # Prefer one that has the session name in its path -- there could be
    # collisions if two sessions have a frame with the same stem.
    for c in candidates:
        if session in str(c):
            return c
    return candidates[0]


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    results_csv = in_dir / "results.csv"
    tracks_detail_csv = in_dir / "tracks_detail.csv"
    masks_root = in_dir / "masks" / "flower"
    if not results_csv.is_file():
        print(f"[fatal] {results_csv} not found", file=sys.stderr)
        sys.exit(1)
    if not masks_root.is_dir():
        print(f"[fatal] {masks_root} not found -- did you run with "
              f"--save-masks?", file=sys.stderr)
        sys.exit(1)

    kept_tracks = read_kept_track_ids(
        tracks_detail_csv, args.min_track_frames,
    )
    if kept_tracks:
        print(
            f"[info] track filter: keeping {len(kept_tracks)} "
            f"tracks with >= {args.min_track_frames} frames",
        )
    else:
        print(
            f"[info] no track filter (all detections kept regardless "
            f"of n_frames)",
        )

    # 1. Walk results.csv to enumerate frames and pair them with their
    #    track-id lists. Also use it to discover the data root if the
    #    user didn't pass --root.
    frames: list[dict] = []  # {day, category, session, img_path, track_ids: list[int]}
    discovered_root: Path | None = None
    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("prompt") != "flower":
                continue
            n = int(row.get("n_detections") or 0)
            if n == 0:
                continue
            img = Path(row["image"])
            if discovered_root is None:
                # Probe upward until we hit a stable common ancestor.
                # All paths share /fs/scratch/PAS0228 (or whatever
                # --root was). Use the first three parents as a guess.
                discovered_root = img.parents[3] if len(img.parents) > 3 else img.parent
            tids_raw = (row.get("track_ids") or "").strip()
            tids = (
                [int(t) for t in tids_raw.split(";") if t]
                if tids_raw else [-1] * n
            )
            frames.append({
                "day": row["day"],
                "category": row["category"],
                "session": row["session"],
                "img_path": img,
                "track_ids": tids,
            })

    root = Path(args.root).expanduser() if args.root else discovered_root
    if root is None:
        print("[fatal] could not infer --root", file=sys.stderr)
        sys.exit(1)
    print(f"[info] root = {root}")
    print(f"[info] {len(frames)} frames have at least one flower detection")

    # 2. Train/val split.
    rng = random.Random(args.seed)
    if args.stratify:
        # Per-session split: each session contributes a val-frac slice
        # of its frames to val.
        by_session: dict[tuple, list[int]] = {}
        for i, fr in enumerate(frames):
            key = (fr["day"], fr["category"], fr["session"])
            by_session.setdefault(key, []).append(i)
        val_idx: set[int] = set()
        for key, idxs in by_session.items():
            rng.shuffle(idxs)
            n_val = int(round(len(idxs) * args.val_frac))
            val_idx.update(idxs[:n_val])
    else:
        all_idx = list(range(len(frames)))
        rng.shuffle(all_idx)
        n_val = int(round(len(all_idx) * args.val_frac))
        val_idx = set(all_idx[:n_val])

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    # 3. For each frame, find the corresponding .npz, read its boxes,
    #    drop boxes whose track id wasn't kept, and write the YOLO label
    #    + image.
    n_train = n_val_w = 0
    n_boxes_written = 0
    n_boxes_dropped_track = 0
    n_boxes_dropped_size = 0
    n_frames_no_npz = 0
    for i, fr in enumerate(frames):
        img_path = fr["img_path"]
        npz_path = find_npz_for_frame(
            masks_root,
            fr["day"], fr["category"], fr["session"],
            img_path.stem,
        )
        if npz_path is None or not npz_path.is_file():
            n_frames_no_npz += 1
            continue
        try:
            data = np.load(npz_path, allow_pickle=False)
            masks = data["masks"]
            boxes = data["boxes"]
        except Exception as e:
            print(f"[warn] could not read {npz_path}: {e}", file=sys.stderr)
            continue
        if boxes is None or len(boxes) == 0:
            continue
        if masks.ndim < 2:
            continue
        H, W = masks.shape[-2:]
        track_ids = fr["track_ids"]
        # Truncate / pad if results.csv and .npz disagree on count.
        n_dets = min(len(boxes), len(track_ids))

        keep_box: list[bool] = []
        for d in range(n_dets):
            x1, y1, x2, y2 = boxes[d]
            w = float(x2) - float(x1)
            h = float(y2) - float(y1)
            if w < args.min_box_px or h < args.min_box_px:
                keep_box.append(False)
                n_boxes_dropped_size += 1
                continue
            if kept_tracks:
                tid = track_ids[d]
                key = (fr["day"], fr["category"], fr["session"], "flower", tid)
                if key not in kept_tracks:
                    keep_box.append(False)
                    n_boxes_dropped_track += 1
                    continue
            keep_box.append(True)
        if not any(keep_box):
            continue

        split = "val" if i in val_idx else "train"
        # Flatten the path so YOLO doesn't choke on whitespace.
        flat = "__".join([
            slugify(fr["day"]), slugify(fr["category"]),
            slugify(fr["session"]), slugify(img_path.stem),
        ])
        out_img = images_dir / split / f"{flat}.jpg"
        out_lbl = labels_dir / split / f"{flat}.txt"

        # YOLO label.
        with open(out_lbl, "w") as lf:
            for d in range(n_dets):
                if not keep_box[d]:
                    continue
                x1, y1, x2, y2 = boxes[d]
                cx = ((float(x1) + float(x2)) / 2) / W
                cy = ((float(y1) + float(y2)) / 2) / H
                w = (float(x2) - float(x1)) / W
                h = (float(y2) - float(y1)) / H
                # YOLO requires normalized coords in [0, 1] and class
                # index. Single-class: class = 0.
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)
                lf.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                n_boxes_written += 1

        # Image: copy or symlink. JPG conversion would require Pillow;
        # YOLO accepts .bmp natively if we keep the extension. Decision:
        # copy/symlink the .bmp as-is and rename .jpg only if needed.
        if not img_path.is_file():
            print(f"[warn] source image missing: {img_path}", file=sys.stderr)
            try:
                out_lbl.unlink()
            except Exception:
                pass
            continue
        # Save with .bmp extension to avoid silent format conversion.
        out_img_bmp = out_img.with_suffix(".bmp")
        if out_img_bmp.exists() or out_img.exists():
            pass  # already done
        elif args.copy_images:
            shutil.copy2(img_path, out_img_bmp)
        else:
            try:
                out_img_bmp.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, out_img_bmp)

        if split == "val":
            n_val_w += 1
        else:
            n_train += 1

    # 4. dataset.yaml for Ultralytics.
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"# YOLO dataset config generated by make_yolo_dataset.py\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['flower']\n",
        encoding="utf-8",
    )

    print()
    print(f"[done] dataset written to: {out_dir}")
    print(f"  train images: {n_train}")
    print(f"  val images:   {n_val_w}")
    print(f"  total boxes written: {n_boxes_written}")
    if n_boxes_dropped_track:
        print(f"  boxes dropped by track filter: {n_boxes_dropped_track}")
    if n_boxes_dropped_size:
        print(f"  boxes dropped by --min-box-px: {n_boxes_dropped_size}")
    if n_frames_no_npz:
        print(f"  frames with no matching .npz: {n_frames_no_npz}")
    print(f"  dataset config: {yaml_path}")
    print()
    print(f"[next] To train YOLOv8/v11 (Ultralytics):")
    print(f"  pip install ultralytics")
    print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=640")


if __name__ == "__main__":
    main()
