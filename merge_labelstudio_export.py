"""Merge a Label Studio YOLO export back into the SAM-generated dataset.

Label Studio's 'Export -> YOLO' produces:
    <ls_export>/images/                  the images you labeled
    <ls_export>/labels/                  one .txt per image (YOLO fmt)
    <ls_export>/notes.json               LS metadata (ignored)
    <ls_export>/classes.txt              one class per line

Strategy:
    For every label file in the LS export, OVERWRITE the matching
    label file in the SAM dataset (the manual labels are
    authoritative for these frames). Frames that LS labeled but
    weren't in the SAM dataset are added to the train split.

Usage:
    python merge_labelstudio_export.py \\
        --ls-export ~/labelstudio_export/ \\
        --sam-dataset ~/yolo_dataset/ \\
        --out ~/yolo_dataset_combined/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls-export", required=True,
                    help="Directory where LS YOLO export was unzipped.")
    ap.add_argument("--sam-dataset", required=True,
                    help="The SAM-generated YOLO dataset (output of "
                         "make_yolo_dataset.py).")
    ap.add_argument("--out", required=True,
                    help="New combined dataset path. Will be created.")
    ap.add_argument("--copy", action="store_true",
                    help="Copy the SAM dataset into --out (default is "
                         "to symlink the unchanged files for speed).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ls_dir = Path(args.ls_export).expanduser()
    sam_dir = Path(args.sam_dataset).expanduser()
    out_dir = Path(args.out).expanduser()

    if not (ls_dir / "labels").is_dir():
        raise SystemExit(
            f"{ls_dir / 'labels'} not found -- is --ls-export pointing "
            f"at the unzipped Label Studio YOLO export?"
        )
    if not (sam_dir / "labels").is_dir():
        raise SystemExit(f"{sam_dir / 'labels'} not found")

    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # 1. Carry over the entire SAM dataset (image + label pairs).
    n_sam_carried = 0
    for split in ("train", "val"):
        for src_lbl in (sam_dir / "labels" / split).glob("*.txt"):
            dst_lbl = out_dir / "labels" / split / src_lbl.name
            shutil.copy2(src_lbl, dst_lbl)
            # Find matching image (any extension):
            for ext in (".bmp", ".jpg", ".jpeg", ".png"):
                src_img = sam_dir / "images" / split / f"{src_lbl.stem}{ext}"
                if src_img.exists() or src_img.is_symlink():
                    dst_img = out_dir / "images" / split / src_img.name
                    if dst_img.exists():
                        break
                    if args.copy:
                        # Resolve symlinks if --copy:
                        if src_img.is_symlink():
                            shutil.copy2(src_img.resolve(), dst_img)
                        else:
                            shutil.copy2(src_img, dst_img)
                    else:
                        try:
                            target = src_img.resolve() if src_img.is_symlink() else src_img.resolve()
                            dst_img.symlink_to(target)
                        except OSError:
                            shutil.copy2(src_img, dst_img)
                    break
            n_sam_carried += 1

    # 2. Apply Label Studio overrides.
    n_overridden = 0
    n_added = 0
    ls_labels = ls_dir / "labels"
    ls_images = ls_dir / "images"
    for ls_lbl in ls_labels.glob("*.txt"):
        stem = ls_lbl.stem
        target_split: str | None = None
        # Find the same stem in the SAM dataset:
        for split in ("train", "val"):
            if (out_dir / "labels" / split / f"{stem}.txt").exists():
                target_split = split
                break
        if target_split is None:
            target_split = "train"
            n_added += 1
        else:
            n_overridden += 1
        # Overwrite label.
        shutil.copy2(ls_lbl, out_dir / "labels" / target_split / ls_lbl.name)
        # Find the matching image in the LS export and place it in the
        # same split so train/val pairing stays intact.
        ls_img: Path | None = None
        for ext in (".bmp", ".jpg", ".jpeg", ".png"):
            cand = ls_images / f"{stem}{ext}"
            if cand.is_file():
                ls_img = cand
                break
        if ls_img is None:
            continue
        dst_img = out_dir / "images" / target_split / ls_img.name
        if not dst_img.exists():
            if args.copy:
                shutil.copy2(ls_img, dst_img)
            else:
                try:
                    dst_img.symlink_to(ls_img.resolve())
                except OSError:
                    shutil.copy2(ls_img, dst_img)

    # 3. dataset.yaml.
    sam_yaml = sam_dir / "dataset.yaml"
    out_yaml = out_dir / "dataset.yaml"
    if sam_yaml.is_file():
        text = sam_yaml.read_text()
        text = text.replace(
            f"path: {sam_dir.resolve()}",
            f"path: {out_dir.resolve()}",
        )
        out_yaml.write_text(text)
    else:
        out_yaml.write_text(
            f"path: {out_dir.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: 1\n"
            f"names: ['flower']\n",
        )

    print()
    print(f"[done] combined dataset -> {out_dir}")
    print(f"  SAM frames carried over     : {n_sam_carried}")
    print(f"  LS labels overrode SAM      : {n_overridden}")
    print(f"  LS labels added new frames  : {n_added}")
    print(f"  config: {out_yaml}")
    print()
    print(f"[next] yolo detect train data={out_yaml} model=yolov8n.pt epochs=50 imgsz=640")


if __name__ == "__main__":
    main()
