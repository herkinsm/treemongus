"""Aggregate the original RGB BMPs that the SAM run analyzed into a
single flat folder, ready for Label Studio upload.

Reads `results.csv` from a SAM analysis output, deduplicates the
'image' column to get the unique source RGB paths, and copies (or
symlinks, default) each into a single flat directory with a
filesystem-safe flattened filename:

    /fs/scratch/PAS0228/2023 day 4/Dynamic/2023-5-11-9-34-3/RGB/<stem>.bmp
    -> 2023_day_4__Dynamic__2023-5-11-9-34-3__<stem>.bmp

Use this OUTPUT FOLDER as the Label Studio `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`
or as the input to make_labelstudio_tasks.py:

    python extract_rgb_for_labeling.py \\
        --in $HOME/sam3_yolo_labels \\
        --out $HOME/sam3_yolo_labels/rgb_for_labeling \\
        --prompt flower
    python make_labelstudio_tasks.py \\
        --images $HOME/sam3_yolo_labels/rgb_for_labeling \\
        --yolo-dataset $HOME/yolo_dataset \\
        --out $HOME/labelstudio_import.json

Filters by --prompt so you only get frames that actually had flower
detections (matching what the YOLO labels and the LS predictions cover).
Pass --prompt-any to instead pull every analyzed frame regardless of
prompt -- useful for labeling no-detection frames as negatives.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True,
                    help="Input directory: where analyze_days.py wrote "
                         "results.csv.")
    ap.add_argument("--out", required=True,
                    help="Output flat directory for the RGB files.")
    ap.add_argument("--prompt", default="flower",
                    help="Only include frames with detections for this "
                         "prompt (default 'flower'). Use --prompt-any to "
                         "include every analyzed frame regardless.")
    ap.add_argument("--prompt-any", action="store_true",
                    help="Include every analyzed frame, even ones with "
                         "0 detections or non-flower prompts.")
    ap.add_argument("--copy", action="store_true",
                    help="COPY the BMPs (default: symlink to save disk). "
                         "Use this if you need to move the folder to a "
                         "different machine.")
    ap.add_argument("--convert-jpg", action="store_true",
                    help="Convert BMPs to JPG (q=92) instead of "
                         "copying/symlinking. Smaller files, faster "
                         "Label Studio loading. Requires Pillow.")
    ap.add_argument("--root-prefix", default="/fs/scratch/PAS0228",
                    help="Path prefix to strip when flattening filenames. "
                         "Default '/fs/scratch/PAS0228'. Falls back to "
                         "using just the basename if no match.")
    return ap.parse_args()


def slugify_path(p: Path, prefix: Path | None) -> str:
    """Convert a full path into a flat filesystem-safe filename.

    Skips the per-modality folder name (RGB / IR / Depth / PRGB)
    so the resulting stem matches what make_yolo_dataset.py
    produces -- this keeps make_labelstudio_tasks.py's stem-based
    YOLO-label lookup working without slug-rewriting.
    """
    try:
        rel = p.relative_to(prefix) if prefix is not None else p
    except (ValueError, TypeError):
        rel = Path(p.name)
    skip_modality = {"RGB", "IR", "Depth", "PRGB", "Info",
                     "rgb", "ir", "depth", "prgb", "info"}
    parts = [
        part.replace(" ", "_")
        for part in rel.parts
        if part not in skip_modality
    ]
    flat = "__".join(parts)
    return flat


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir).expanduser()
    out_dir = Path(args.out).expanduser()
    results_csv = in_dir / "results.csv"
    if not results_csv.is_file():
        raise SystemExit(f"missing: {results_csv}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(args.root_prefix) if args.root_prefix else None

    seen: set[str] = set()
    n_in = n_copied = n_already = n_missing = 0

    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not args.prompt_any:
                if row.get("prompt") != args.prompt:
                    continue
                # Only frames that had at least one detection of the
                # target prompt -- LS upload set should match what the
                # YOLO labels cover.
                try:
                    if int(row.get("n_detections") or 0) <= 0:
                        continue
                except ValueError:
                    continue
            src_str = row.get("image") or ""
            if not src_str or src_str in seen:
                continue
            seen.add(src_str)
            n_in += 1
            src = Path(src_str)
            if not src.is_file():
                n_missing += 1
                continue
            flat = slugify_path(src, prefix)
            if args.convert_jpg:
                flat = Path(flat).with_suffix(".jpg").name
            dst = out_dir / flat
            if dst.exists() or dst.is_symlink():
                n_already += 1
                continue
            try:
                if args.convert_jpg:
                    from PIL import Image
                    img = Image.open(src).convert("RGB")
                    img.save(dst, "JPEG", quality=92)
                elif args.copy:
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src.resolve())
                n_copied += 1
            except OSError as e:
                # symlink can fail on filesystems that don't support
                # them (some HPC scratch is mounted with nosymlinks);
                # fall back to copy.
                if not args.copy and not args.convert_jpg:
                    try:
                        shutil.copy2(src, dst)
                        n_copied += 1
                    except Exception as e2:
                        print(
                            f"[warn] could not copy {src}: {e2}",
                            file=sys.stderr,
                        )
                else:
                    print(
                        f"[warn] could not write {dst}: {e}",
                        file=sys.stderr,
                    )

    print()
    print(f"[done] flat RGB folder -> {out_dir}")
    print(f"  unique frames matched : {n_in}")
    print(f"  written / linked      : {n_copied}")
    print(f"  already present       : {n_already}")
    if n_missing:
        print(f"  source missing        : {n_missing}")
    print()
    print("[next] Build the Label Studio import JSON:")
    print(f"  python make_labelstudio_tasks.py \\")
    print(f"    --images {out_dir} \\")
    print(f"    --yolo-dataset $HOME/yolo_dataset \\")
    print(f"    --out $HOME/labelstudio_import.json")


if __name__ == "__main__":
    main()
