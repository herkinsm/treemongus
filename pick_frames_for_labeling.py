"""Select N frames worth manually labeling, based on SAM's own uncertainty.

For supplementing SAM-generated YOLO labels with human corrections, we want
to focus labeling effort where SAM is most likely wrong. The signals SAM
already gives us per detection (in rejections_per_mask.jsonl):

  - soft_score:      a borderline value (0.20-0.40) means the score
                     barely cleared / barely missed the gate. These are
                     the high-leverage cases.
  - sam_score:       a high SAM raw score paired with a low soft_score
                     means the model SAW a flower but our gates rejected
                     it. Could be wrong rejection.
  - n_detections:    frames with many detections are usually high-density
                     canopy where mistakes compound.

Selection strategy (mix of three buckets, equal-weighted by default):

  1) BORDERLINE-KEPT: frames whose kept flowers had soft_scores clustered
                       near the threshold (0.20-0.35). These are the
                       'maybe-real' SAM detections that most need a human
                       check.
  2) BORDERLINE-REJECTED: frames where the highest-SAM rejected mask had
                       soft_score in 0.15-0.25 range. SAM was confident
                       but the gate dropped them. Could be missed real
                       flowers.
  3) HIGH-DENSITY: frames with the most kept flowers (n_detections >=
                       p90 across the corpus). High-density canopy is
                       where labeler accuracy matters most.

Output: a list of image paths, one per line, suitable for piping into
Label Studio or copying to a labeling staging directory.

Usage:
    python pick_frames_for_labeling.py \\
        --in $HOME/sam3_yolo_labels \\
        --n 200 \\
        --out $HOME/yolo_label_candidates.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True,
                    help="The SAM run output directory (with results.csv "
                         "and rejections_per_mask.jsonl).")
    ap.add_argument("--n", type=int, default=200,
                    help="Total number of frames to select.")
    ap.add_argument("--out", required=True,
                    help="Output file -- one image path per line.")
    ap.add_argument("--copy-to", default=None,
                    help="If set, copy the selected images into this "
                         "directory (preserves session folder structure).")
    ap.add_argument("--borderline-kept-frac", type=float, default=0.4,
                    help="Fraction of N to draw from the borderline-kept "
                         "bucket (soft scores 0.20-0.35).")
    ap.add_argument("--borderline-rejected-frac", type=float, default=0.4,
                    help="Fraction of N to draw from the borderline-"
                         "rejected bucket (high SAM, low soft score).")
    ap.add_argument("--high-density-frac", type=float, default=0.2,
                    help="Fraction of N for high-density (most kept "
                         "flowers per frame) bucket.")
    ap.add_argument("--one-per-session", action="store_true",
                    help="Override: ignore --n and the bucket fractions. "
                         "Pick exactly one representative frame per "
                         "session (the median-density one). Maximum "
                         "diversity, minimum redundancy. Good for a "
                         "small but well-distributed labeling set.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir).expanduser()
    results_csv = in_dir / "results.csv"
    rej_jsonl = in_dir / "rejections_per_mask.jsonl"
    if not results_csv.is_file():
        raise SystemExit(f"missing: {results_csv}")
    have_jsonl = rej_jsonl.is_file()
    if not have_jsonl:
        print(f"[warn] {rej_jsonl} not present -- only the high-density "
              f"bucket will fire (need --debug-rejection-log on the run).")

    # Build per-frame stats from results.csv (n_detections per flower frame).
    frame_density: dict[str, int] = {}
    frame_session: dict[str, tuple[str, str, str]] = {}
    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("prompt") != "flower":
                continue
            try:
                n = int(row.get("n_detections") or 0)
            except ValueError:
                n = 0
            frame_density[row["image"]] = n
            frame_session[row["image"]] = (
                row.get("day", ""), row.get("category", ""),
                row.get("session", ""),
            )

    # Short-circuit: --one-per-session takes the median-density frame
    # in each session. Returns max-diversity, min-redundancy set.
    if args.one_per_session:
        by_session_frames: dict[tuple[str, str, str], list[tuple[str, int]]] = {}
        for img, n in frame_density.items():
            key = frame_session[img]
            by_session_frames.setdefault(key, []).append((img, n))
        selected: list[str] = []
        for key, lst in sorted(by_session_frames.items()):
            lst.sort(key=lambda x: x[1])
            mid = lst[len(lst) // 2]
            selected.append(mid[0])
        print(f"[picks] one-per-session: {len(selected)} frames "
              f"(one median-density frame per session)")
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
        print(f"\n[done] wrote {len(selected)} paths to {out_path}")
        if args.copy_to:
            copy_dst = Path(args.copy_to).expanduser()
            copy_dst.mkdir(parents=True, exist_ok=True)
            n_copied = 0
            for src_str in selected:
                src = Path(src_str)
                if not src.is_file():
                    continue
                try:
                    rel = src.relative_to(Path("/fs/scratch/PAS0228"))
                except ValueError:
                    rel = Path(src.name)
                flat = "__".join(p.replace(" ", "_") for p in rel.parts)
                dst = copy_dst / flat
                if not dst.exists():
                    shutil.copy2(src, dst)
                    n_copied += 1
            print(f"[done] copied {n_copied} images to {copy_dst}")
        return

    # Per-frame soft-score signals from the JSONL.
    # min_kept_soft_per_frame: lowest kept soft score (most borderline-real)
    # max_rejected_high_sam_per_frame: best (sam_score, -soft_score) pair
    #   for any rejected detection (best candidate for "missed real flower")
    min_kept_soft: dict[str, float] = {}
    best_rej_evidence: dict[str, tuple[float, float]] = {}
    if have_jsonl:
        with open(rej_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("prompt") != "flower":
                    continue
                soft = r.get("soft_score")
                sam = r.get("sam_score")
                img = r.get("image")
                if img is None:
                    continue
                if r.get("kept") and isinstance(soft, (int, float)):
                    if 0.20 <= float(soft) <= 0.35:
                        cur = min_kept_soft.get(img)
                        if cur is None or float(soft) < cur:
                            min_kept_soft[img] = float(soft)
                elif (not r.get("kept")
                      and isinstance(sam, (int, float))
                      and isinstance(soft, (int, float))
                      and r.get("rejected_by") == "soft_score"
                      and 0.15 <= float(soft) <= 0.25
                      and float(sam) >= 0.30):
                    cur = best_rej_evidence.get(img, (0.0, 1.0))
                    # Higher SAM, lower soft = stronger candidate for
                    # "real flower SAM saw but soft-score barely missed"
                    score = (float(sam), -float(soft))
                    if score > cur:
                        best_rej_evidence[img] = score

    # Bucket selection.
    n_kept = int(round(args.n * args.borderline_kept_frac))
    n_rej = int(round(args.n * args.borderline_rejected_frac))
    n_dense = max(0, args.n - n_kept - n_rej)

    # Bucket 1: borderline-kept frames, sorted by min kept soft score
    # (closest to threshold = highest priority).
    bk_sorted = sorted(min_kept_soft.items(), key=lambda kv: kv[1])
    bucket_kept = [img for img, _ in bk_sorted[:n_kept]]

    # Bucket 2: borderline-rejected. Pick frames with the best
    # rejected-evidence score (high SAM + low soft).
    br_sorted = sorted(
        best_rej_evidence.items(), key=lambda kv: -kv[1][0],
    )
    bucket_rej = [img for img, _ in br_sorted[:n_rej]
                  if img not in bucket_kept]

    # Bucket 3: high-density frames not already picked.
    used = set(bucket_kept) | set(bucket_rej)
    fd_sorted = sorted(frame_density.items(), key=lambda kv: -kv[1])
    bucket_dense: list[str] = []
    for img, n in fd_sorted:
        if img in used:
            continue
        bucket_dense.append(img)
        if len(bucket_dense) >= n_dense:
            break

    selected = bucket_kept + bucket_rej + bucket_dense
    print(f"[picks]")
    print(f"  borderline-kept     : {len(bucket_kept):4d}")
    print(f"  borderline-rejected : {len(bucket_rej):4d}")
    print(f"  high-density        : {len(bucket_dense):4d}")
    print(f"  total               : {len(selected):4d}")

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"\n[done] wrote {len(selected)} paths to {out_path}")

    if args.copy_to:
        copy_dst = Path(args.copy_to).expanduser()
        copy_dst.mkdir(parents=True, exist_ok=True)
        n_copied = 0
        for src_str in selected:
            src = Path(src_str)
            if not src.is_file():
                continue
            # Flatten path so the labeling tool sees unique filenames:
            #   /fs/scratch/PAS0228/2023 day 4/Dynamic/2023-5-11.../<stem>.bmp
            # -> 2023_day_4__Dynamic__2023-5-11..._<stem>.bmp
            try:
                rel = src.relative_to(Path("/fs/scratch/PAS0228"))
            except ValueError:
                rel = Path(src.name)
            flat = "__".join(p.replace(" ", "_") for p in rel.parts)
            dst = copy_dst / flat
            if not dst.exists():
                shutil.copy2(src, dst)
                n_copied += 1
        print(f"[done] copied {n_copied} images to {copy_dst}")


if __name__ == "__main__":
    main()
