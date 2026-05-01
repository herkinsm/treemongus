"""Build a Label Studio import file with SAM-3 pre-annotations.

Each task = one image. The image's existing YOLO label (from
make_yolo_dataset.py) is converted into Label Studio's
'predictions' format so the boxes appear pre-drawn when you open
the task. You correct/add/delete instead of starting from scratch
-- ~10x faster than blank-canvas labeling.

Inputs:
    --images       Directory containing the BMPs to be labeled
                   (typically the --copy-to output of
                   pick_frames_for_labeling.py).
    --yolo-dataset The YOLO dataset folder produced by
                   make_yolo_dataset.py. Optional -- without it,
                   tasks are emitted with no pre-annotations
                   (you'd label from scratch).
    --out          The Label Studio JSON to save.

Output: a JSON array of {data, predictions} dicts. In Label
Studio, "Import" this file as 'Tasks With Predictions'.

Notes:
    Label Studio coordinates are in PERCENT of image size, not
    pixels. YOLO is normalized [0,1] with center-x, center-y,
    width, height. We multiply by 100 and convert center -> top-
    left.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True,
                    help="Directory of images to label.")
    ap.add_argument("--yolo-dataset", default=None,
                    help="Optional: path to yolo_dataset/ folder so "
                         "SAM's existing labels are pre-loaded.")
    ap.add_argument("--out", required=True,
                    help="Output JSON path for Label Studio import.")
    ap.add_argument("--label-name", default="flower",
                    help="Single class name to write into the labels.")
    ap.add_argument("--url-prefix", default="/data/local-files/?d=",
                    help="Prefix for image URLs as Label Studio "
                         "expects them when using local-files storage. "
                         "Leave default unless you know you need to "
                         "change it.")
    return ap.parse_args()


def find_yolo_label(stem: str, yolo_dir: Path | None) -> Path | None:
    """Look up the YOLO label for an image stem in train/ then val/.

    Tries multiple stem variants to absorb the slug-format mismatch
    between extract_rgb_for_labeling.py (which includes "RGB" as a
    path part: <day>__<category>__<session>__RGB__<stem>) and
    make_yolo_dataset.py (which uses only 4 parts:
    <day>__<category>__<session>__<stem>). Without these variants,
    every task imports with zero predictions because the image
    filename's stem never matches the YOLO label filename's stem.
    """
    if yolo_dir is None:
        return None
    variants = [
        stem,
        # Strip the RGB folder marker if present anywhere:
        stem.replace("__RGB__", "__"),
        stem.replace("__rgb__", "__"),
    ]
    seen = set()
    for cand_stem in variants:
        if cand_stem in seen:
            continue
        seen.add(cand_stem)
        for split in ("train", "val"):
            cand = yolo_dir / "labels" / split / f"{cand_stem}.txt"
            if cand.is_file():
                return cand
    return None


def yolo_to_labelstudio_boxes(
    yolo_lbl_path: Path, label_name: str,
) -> list[dict]:
    """Convert a YOLO label file (one bbox per line) into Label
    Studio's 'rectanglelabels' result format. Drops malformed lines
    silently."""
    out: list[dict] = []
    for line in yolo_lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            _cls = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            continue
        # YOLO is normalized [0,1] center+size; LS is percent top-left+size.
        x_pct = (cx - w / 2) * 100
        y_pct = (cy - h / 2) * 100
        w_pct = w * 100
        h_pct = h * 100
        out.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": 640,
            "original_height": 480,
            "value": {
                "x": max(0.0, x_pct),
                "y": max(0.0, y_pct),
                "width": min(100.0, w_pct),
                "height": min(100.0, h_pct),
                "rotation": 0,
                "rectanglelabels": [label_name],
            },
        })
    return out


def main() -> None:
    args = parse_args()
    img_dir = Path(args.images).expanduser()
    yolo_dir = (
        Path(args.yolo_dataset).expanduser()
        if args.yolo_dataset else None
    )
    out_path = Path(args.out).expanduser()

    images: list[Path] = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {
            ".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff",
        }
    )
    if not images:
        raise SystemExit(f"no images in {img_dir}")

    tasks: list[dict] = []
    n_pre = 0
    for img in images:
        url = f"{args.url_prefix}{img}"
        item: dict = {"data": {"image": url}}
        yolo_lbl = find_yolo_label(img.stem, yolo_dir)
        if yolo_lbl is not None:
            boxes = yolo_to_labelstudio_boxes(yolo_lbl, args.label_name)
            if boxes:
                item["predictions"] = [{
                    "model_version": "sam3-prelabel",
                    "result": boxes,
                }]
                n_pre += 1
        tasks.append(item)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(tasks)} tasks ({n_pre} with pre-annotations) "
          f"to {out_path}")
    print(f"[next] In Label Studio: Import -> Upload Files -> "
          f"select {out_path}")


if __name__ == "__main__":
    main()
