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
    "apple blossom",
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


def find_images(root: Path, only_rgb_folders: bool = True,
                sample_per_session: int | None = None,
                frame_range: tuple[int, int] | None = None,
                require_all_modalities: bool = False):
    """Yield (day, category, session, image_path) tuples.

    When require_all_modalities is True, the per-session image list is FIRST
    filtered down to frames that have matching depth / IR / PRGB sibling
    files, and THEN frame_range / sample_per_session is applied. This way
    --frame-range 50 80 means "the 50th-80th *complete* frame" rather than
    "the 50th-80th raw frame, of which some may get skipped later"."""
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
                    if require_all_modalities:
                        n_total = len(imgs)
                        miss_d = miss_i = miss_p = 0
                        kept: list[Path] = []
                        for p in imgs:
                            ok_d = depth_path_for(p).is_file()
                            ok_i = ir_path_for(p).is_file()
                            ok_p = prgb_path_for(p).is_file()
                            if ok_d and ok_i and ok_p:
                                kept.append(p)
                            else:
                                if not ok_d:
                                    miss_d += 1
                                if not ok_i:
                                    miss_i += 1
                                if not ok_p:
                                    miss_p += 1
                        imgs = kept
                        if len(imgs) < n_total:
                            print(f"[scan] {session_dir.name}: "
                                  f"{len(imgs)}/{n_total} complete frames "
                                  f"(missing depth={miss_d} IR={miss_i} PRGB={miss_p})",
                                  file=sys.stderr)
                        if not imgs:
                            continue
                    if frame_range is not None:
                        a, b = frame_range
                        a = max(0, a)
                        b = min(len(imgs), b)
                        idxs = list(range(a, b))
                    else:
                        idxs = sample_indices(len(imgs), sample_per_session)
                    for i in idxs:
                        yield day_dir.name, category_dir.name, session_dir.name, imgs[i]


def make_overlay(img: Image.Image, masks: np.ndarray, boxes: np.ndarray | None,
                 title: str | None = None,
                 track_ids: list[int] | None = None) -> Image.Image:
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
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                    edgecolor="lime", linewidth=1.5))
            # Label each box with its track_id when available so the same
            # physical flower keeps the same number across frames.
            if track_ids is not None and i < len(track_ids) and track_ids[i] >= 0:
                ax.text(x1 + 2, max(y1 - 4, 8), f"#{track_ids[i]}",
                        color="lime", fontsize=9, fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.5,
                                  edgecolor="none", pad=1))
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
    .../<session>/depth/<stem>-Depth.<bmp|txt>. Prefers .bmp (16-bit grayscale
    uint16 mm — the format tree_mask.py expects) over the legacy ASCII .txt
    decimation. Falls back to .txt if .bmp doesn't exist."""
    session_dir = img_path.parent.parent  # drop RGB/
    stem = img_path.stem
    base = stem
    for suffix in ("-RGB-BP", "-RGB-bp", "-RGB", "-rgb"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    bmp = session_dir / "depth" / f"{base}-Depth.bmp"
    if bmp.is_file():
        return bmp
    return session_dir / "depth" / f"{base}-Depth.txt"


def prgb_path_for(img_path: Path) -> Path:
    """Given .../<session>/RGB/<stem>-RGB-BP.<ext>, return
    .../<session>/PRGB/<stem>-RGB-PP.bmp (PRGB folder is a sibling of RGB,
    files share the timestamp stem with the suffix swapped to -RGB-PP)."""
    session_dir = img_path.parent.parent  # drop RGB/
    stem = img_path.stem
    base = stem
    for suffix in ("-RGB-BP", "-RGB-bp", "-RGB", "-rgb"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return session_dir / "PRGB" / f"{base}-RGB-PP.bmp"


def ir_path_for(img_path: Path) -> Path:
    """Given .../<session>/RGB/<stem>-RGB-BP.<ext>, return
    .../<session>/IR/<stem>-IR.bmp."""
    session_dir = img_path.parent.parent
    stem = img_path.stem
    base = stem
    for suffix in ("-RGB-BP", "-RGB-bp", "-RGB", "-rgb"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return session_dir / "IR" / f"{base}-IR.bmp"


def extract_roi_mask(prgb_path: Path, target_hw: tuple[int, int],
                      red_r_min: int = 180, red_gb_max: int = 80,
                      min_box_area_px: int = 200,
                      min_box_side_px: int = 25) -> np.ndarray | None:
    """Build a binary ROI mask from a PRGB image whose red rectangles outline
    the per-tree zones drawn by the sprayer pipeline.

    Returns (H, W) bool mask: True INSIDE any detected red rectangle,
    False outside. Returns None if the PRGB file is missing/unreadable or
    if no rectangles were detected.

    Algorithm:
      1. Threshold pixels where R >= red_r_min AND G,B <= red_gb_max
         (the bright red used for the bounding boxes).
      2. Connected-component analysis on the red-edge mask.
      3. For each CC of sufficient pixel area and bbox side, fill that CC's
         axis-aligned bounding rectangle into the ROI mask.
    """
    if not prgb_path.is_file():
        return None
    try:
        prgb = np.asarray(Image.open(prgb_path).convert("RGB"))
    except Exception:
        return None
    if prgb.shape[:2] != target_hw:
        pim = Image.fromarray(prgb).resize((target_hw[1], target_hw[0]), Image.NEAREST)
        prgb = np.asarray(pim)
    r, g, b = prgb[..., 0], prgb[..., 1], prgb[..., 2]
    red_edge = (r >= red_r_min) & (g <= red_gb_max) & (b <= red_gb_max)
    if int(red_edge.sum()) < 20:
        return None

    import cv2
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(
        red_edge.astype(np.uint8), connectivity=8
    )
    roi = np.zeros(target_hw, dtype=bool)
    for cc_id in range(1, n_cc):
        x, y, w, h, area = stats[cc_id]
        if area < min_box_area_px:
            continue
        if w < min_box_side_px or h < min_box_side_px:
            continue
        roi[y:y + h, x:x + w] = True
    return roi if roi.any() else None


def roi_overlap_per_mask(masks_np: np.ndarray, roi: np.ndarray) -> list[float]:
    """Fraction of each mask's pixels that lie inside the ROI mask. NaN for empty."""
    out: list[float] = []
    for m in masks_np:
        mb = m.astype(bool)
        if mb.ndim == 3:
            mb = mb.any(axis=0)
        total = int(mb.sum())
        if total == 0:
            out.append(float("nan"))
            continue
        out.append(float((mb & roi).sum()) / float(total))
    return out


def load_depth_mm(depth_path: Path, target_hw: tuple[int, int]) -> np.ndarray | None:
    """Load a depth file and return uint16 mm upsampled to target (H, W).
    Supports two formats:
      - .bmp / .png : 16-bit (or 8-bit) single-channel image, values = depth mm
      - .txt        : ASCII space-separated ints, values = depth mm
    Returns None if the file is missing or malformed."""
    if not depth_path.is_file():
        return None
    suffix = depth_path.suffix.lower()
    try:
        if suffix == ".txt":
            d = np.loadtxt(depth_path, dtype=np.int32)
        else:
            import cv2
            # IMREAD_UNCHANGED preserves the original bit depth (uint16 for
            # the modern depth captures; uint8 if it's an 8-bit normalized
            # visualization, which we'll clip-cast to uint16 below).
            d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if d is None:
                return None
            if d.ndim == 3:  # 3-channel encoding: take the first channel
                d = d[..., 0]
            d = d.astype(np.int32)
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
# Tight port of sprayer_pipeline/tree_mask.py canopy mask. Mirrors:
#   - 3x3 median blur on depth (line 153)
#   - foreground depth band [CANOPY_DEPTH_MIN_MM, CANOPY_DEPTH_MAX_MM]
#   - row-banded ground filter:
#       above ROW_GROUND_START (=280)            : full [min, max] band
#       row-3 band [280, ROW_LOWER_BAND_START=385]: cap at centre + 500 mm
#       row-4 band [385, H)                      : cap at centre + 100 mm
#     where centre = median canopy depth in the upper centre strip [100, 300].
# Skips depth-spread / row-depth correlation / sky filters (option 2).
# ---------------------------------------------------------------------------
def compute_canopy_mask(depth_mm: np.ndarray,
                         min_mm: int = 600, max_mm: int = 3000,
                         row_ground_start: int = 280,
                         row_lower_band_start: int = 385,
                         ground_band_mm: int = 100,
                         ground_band_mm_row3: int = 500,
                         min_cc_area_px: int = 2000,
                         max_row_width_frac: float = 0.0) -> np.ndarray:
    """Return a (H, W) bool canopy mask. True == canopy, False == ground/sky/bg.

    Pipeline:
      1. 3x3 median blur on depth.
      2. Foreground depth band [min_mm, max_mm].
      3. Row-banded ground filter (above row 280: full band; row 3 [280,385):
         centre + ground_band_mm_row3; row 4 [385,H): centre + ground_band_mm).
      4. Morphological close (5x5) to merge gappy canopy fragments.
      5. Connected-component size filter — drop CCs smaller than min_cc_area_px.
         Trees are typically thousands of pixels; dandelion / ground-flower
         patches are typically <1000 px. Default 2000 px keeps trees, removes
         isolated ground islands.
      6. Optional row-width filter — clear rows whose canopy spans > some
         fraction of frame width (long horizontal canopy bands are usually
         ground/grass at a tilt rather than a tree). Off by default
         (max_row_width_frac=0).
    """
    import cv2
    H, W = depth_mm.shape
    d = cv2.medianBlur(depth_mm.astype(np.uint16), 3).astype(np.int32)

    # 1-2. Median depth of the upper centre strip — frame's canopy reference depth.
    upper_band = (d >= min_mm) & (d <= max_mm)
    cs_top, cs_bot = min(100, H), min(300, H)
    valid = d[cs_top:cs_bot][upper_band[cs_top:cs_bot]]
    centre = int(np.median(valid)) if valid.size else (min_mm + max_mm) // 2

    # 3. Per-row max depth allowed for canopy.
    row_max = np.full(H, max_mm, dtype=np.int32)
    if row_ground_start < H:
        r3_end = min(row_lower_band_start, H)
        row_max[row_ground_start:r3_end] = min(centre + ground_band_mm_row3, max_mm)
    if row_lower_band_start < H:
        row_max[row_lower_band_start:H] = min(centre + ground_band_mm, max_mm)

    canopy = (d >= min_mm) & (d <= row_max[:, None])

    # 4-5. Connected-component size filter (drop isolated ground-flower patches).
    if min_cc_area_px > 0:
        u8 = canopy.astype(np.uint8)
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
        keep_label = np.zeros(n_cc, dtype=bool)
        for cc_id in range(1, n_cc):
            if stats[cc_id, cv2.CC_STAT_AREA] >= min_cc_area_px:
                keep_label[cc_id] = True
        canopy = keep_label[labels] & canopy

    # 6. Optional: clear rows that are mostly canopy width-wise (horizontal grass).
    if max_row_width_frac > 0.0:
        row_widths = canopy.sum(axis=1) / max(W, 1)
        wide_rows = row_widths > max_row_width_frac
        canopy[wide_rows, :] = False

    return canopy.astype(bool)


def canopy_overlap_per_mask(masks_np: np.ndarray, canopy: np.ndarray) -> list[float]:
    """Fraction of each mask's pixels that lie inside the canopy mask. NaN for empty."""
    out: list[float] = []
    for m in masks_np:
        mb = m.astype(bool)
        if mb.ndim == 3:
            mb = mb.any(axis=0)
        total = int(mb.sum())
        if total == 0:
            out.append(float("nan"))
            continue
        out.append(float((mb & canopy).sum()) / float(total))
    return out


def mask_depth_geom(mask, depth_mm: np.ndarray) -> tuple[float, float, float]:
    """For a single mask, return (depth_spread_mm, row_depth_corr, mean_depth_mm)
    using only valid (depth > 0) pixels.

    spread = max - min depth within mask; ports CANOPY_MIN_DEPTH_SPREAD_MM.
    row_depth_corr = Pearson r between image-y and depth — high positive
        correlation (depth increases with y) is the signature of orchard
        floor at a typical camera tilt; ports CANOPY_MAX_DEPTH_ROW_CORRELATION.
    Returns (0.0, 0.0, 0.0) when too few valid pixels to compute statistics."""
    mb = np.asarray(mask).astype(bool)
    if mb.ndim == 3:
        mb = mb.any(axis=0)
    if mb.sum() < 10:
        return 0.0, 0.0, 0.0
    ys, _ = np.nonzero(mb)
    d_in = depth_mm[mb]
    valid = d_in > 0
    if valid.sum() < 10:
        return 0.0, 0.0, 0.0
    ys = ys[valid].astype(np.float64)
    d = d_in[valid].astype(np.float64)
    spread = float(d.max() - d.min())
    yvar, dvar = ys.var(), d.var()
    if yvar < 1e-9 or dvar < 1e-9:
        return spread, 0.0, float(d.mean())
    corr = float(((ys - ys.mean()) * (d - d.mean())).mean()
                  / np.sqrt(yvar * dvar))
    return spread, corr, float(d.mean())


def split_cluster_mask(mask: np.ndarray, rgb_arr: np.ndarray,
                        min_blossom_area_px: int = 30,
                        min_marker_distance_px: int = 5,
                        intensity_weight: float = 5.0,
                        edge_weight: float = 0.5,
                        use_otsu: bool = True) -> list[np.ndarray]:
    """Split a cluster mask into per-blossom sub-masks using marker-controlled
    watershed segmentation, fusing four cues:

      - Distance transform of the mask              (geometric centers)
      - V-channel intensity within the mask          (optical petal centers)
      - OTSU on V to gate weak intensity peaks       (only bright pixels qualify)
      - Sobel gradient as watershed barriers         (petal-petal boundaries)

    Returns a list of (H, W) bool masks (one per blossom). If the cluster only
    yields one valid marker, returns the input mask unchanged in a 1-element
    list."""
    import cv2

    mb = np.asarray(mask).astype(bool)
    if mb.ndim == 3:
        mb = mb.any(axis=0)
    if mb.sum() < 2 * min_blossom_area_px:
        return [mb]

    # Crop to bbox for speed.
    ys, xs = np.nonzero(mb)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub = mb[y0:y1, x0:x1]
    sub_rgb = rgb_arr[y0:y1, x0:x1]
    H_s, W_s = sub.shape

    # Distance transform — peaks are geometric blossom centers.
    dist = cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 3)
    if dist.max() < min_marker_distance_px / 2:
        return [mb]

    # V channel inside the mask.
    hsv = cv2.cvtColor(sub_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32) / 255.0
    v_in = np.where(sub, v, 0.0)

    # Optional OTSU gate: only allow markers where V passes the OTSU threshold.
    bright_gate = np.ones_like(sub, dtype=bool)
    if use_otsu and sub.sum() > 50:
        v_u8 = (v_in * 255).astype(np.uint8)
        # OTSU only on the masked region — pixels outside mask should not bias it.
        masked_vals = v_u8[sub]
        if masked_vals.size > 0:
            otsu_val, _ = cv2.threshold(masked_vals, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bright_gate = v_u8 >= int(otsu_val)

    # Marker score = distance * (1 + intensity_weight * V), gated by bright_gate.
    score = dist * (1.0 + intensity_weight * v_in) * bright_gate.astype(np.float32)

    # Local maxima inside the mask, separated by min_marker_distance_px.
    k = 2 * min_marker_distance_px + 1
    dilated = cv2.dilate(score, np.ones((k, k), np.float32))
    local_max = (score == dilated) & (score > 0) & sub

    # Convert peaks to integer markers.
    n_labels, markers = cv2.connectedComponents(local_max.astype(np.uint8))
    if n_labels <= 2:  # 0 = background, 1 = single peak → no split needed
        return [mb]

    # Sobel edges → watershed barriers.
    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)

    # Topographic input: peaks become valleys (negate score), edges raise barriers.
    topo = -score + edge_weight * edge
    if topo.max() > topo.min():
        topo = (topo - topo.min()) / (topo.max() - topo.min())
    topo_u8 = (topo * 255).astype(np.uint8)
    topo_bgr = cv2.cvtColor(topo_u8, cv2.COLOR_GRAY2BGR)

    # OpenCV watershed needs int32 markers, mask area gets non-zero seeds; outside mask stays 0.
    ws_markers = markers.astype(np.int32).copy()
    ws_markers[~sub] = 0  # only flood within the cluster mask
    cv2.watershed(topo_bgr, ws_markers)

    # Extract per-marker sub-masks back into full-frame coords.
    out: list[np.ndarray] = []
    for label_id in range(1, n_labels):
        sub_label = (ws_markers == label_id) & sub
        if int(sub_label.sum()) < min_blossom_area_px:
            continue
        full = np.zeros_like(mb, dtype=bool)
        full[y0:y1, x0:x1] = sub_label
        out.append(full)

    return out if out else [mb]


def _mask_mean_hsv(rgb_arr: np.ndarray, m) -> tuple[float, float, float]:
    """Mean (H, S, V) over masked pixels of an RGB image. OpenCV convention:
    H is 0-179, S and V are 0-255. Returns (0,0,0) for empty masks."""
    import cv2
    mb = np.asarray(m).astype(bool)
    if mb.ndim == 3:
        mb = mb.any(axis=0)
    if mb.sum() == 0:
        return 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
    h = float(hsv[..., 0][mb].mean())
    s = float(hsv[..., 1][mb].mean())
    v = float(hsv[..., 2][mb].mean())
    return h, s, v


def _mask_shape_stats(m) -> tuple[float, float, int, int, int, int, int, float]:
    """Return (circularity, solidity, area, bx, by, bw, bh, centroid_y)
    for a 2D-or-3D bool mask, using cv2 contour ops to mirror
    sprayer_pipeline/flower_detector.py's circularity / solidity / bbox checks.
    circ = 4*pi*area / perim^2 ; sol = area / hull_area."""
    import cv2
    mb = np.asarray(m).astype(bool)
    if mb.ndim == 3:
        mb = mb.any(axis=0)
    area = int(mb.sum())
    if area == 0:
        return 0.0, 0.0, 0, 0, 0, 0, 0, 0.0
    u8 = (mb.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, 0.0, area, 0, 0, 0, 0, 0.0
    cnt = max(contours, key=cv2.contourArea)
    perim = float(cv2.arcLength(cnt, True))
    circ = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    sol = (area / hull_area) if hull_area > 0 else 0.0
    bx, by, bw, bh = (int(v) for v in cv2.boundingRect(cnt))
    ys, _ = np.nonzero(mb)
    cy = float(ys.mean()) if ys.size else 0.0
    return circ, sol, area, bx, by, bw, bh, cy


def flower_quality_keep(masks_np: np.ndarray,
                         min_area: int, max_area: int,
                         y_min: int, y_max: int,
                         min_circ: float, min_sol: float,
                         edge_margin: int,
                         img_h: int, img_w: int,
                         rgb_arr: np.ndarray | None = None,
                         reject_yellow: bool = False,
                         yellow_h_lo: int = 15, yellow_h_hi: int = 45,
                         yellow_s_min: int = 80) -> tuple[np.ndarray, dict, list[int]]:
    """Boolean keep-array + per-rejection diagnostics + per-mask area, mirroring
    the gates in sprayer_pipeline/flower_detector.py:
      - cluster area in [min_area, max_area]                  (min_cluster_px / max_cluster_px)
      - centroid y in [y_min, y_max]                          (top_row / ground_row)
      - bbox 15 px clear of every frame edge                  (edge_margin)
      - circularity = 4*pi*A/P^2 >= 0.25                      (circularity)
      - solidity   = A / hull_area >= 0.50                    (solidity)
      - mean H not in yellow range AND mean S < threshold     (yellow_color)
        (rejects dandelions / yellow ground flowers; off unless reject_yellow)
    Diagnostic dict counts how many masks fell to each gate.
    Returns (keep, diag, areas_kept_in_order_of_input)."""
    keep = np.ones(len(masks_np), dtype=bool)
    diag = {"min_cluster_px": 0, "max_cluster_px": 0,
            "top_row": 0, "ground_row": 0,
            "edge_margin": 0, "circularity": 0, "solidity": 0,
            "yellow_color": 0}
    areas: list[int] = []
    for i, m in enumerate(masks_np):
        circ, sol, area, bx, by, bw, bh, cy = _mask_shape_stats(m)
        areas.append(area)
        if area < min_area:
            keep[i] = False; diag["min_cluster_px"] += 1; continue
        if area > max_area:
            keep[i] = False; diag["max_cluster_px"] += 1; continue
        if cy < y_min:
            keep[i] = False; diag["top_row"] += 1; continue
        if cy > y_max:
            keep[i] = False; diag["ground_row"] += 1; continue
        if edge_margin > 0:
            if (bx < edge_margin or by < edge_margin
                    or (bx + bw) > (img_w - 1 - edge_margin)
                    or (by + bh) > (img_h - 1 - edge_margin)):
                keep[i] = False; diag["edge_margin"] += 1; continue
        if circ < min_circ:
            keep[i] = False; diag["circularity"] += 1; continue
        if sol < min_sol:
            keep[i] = False; diag["solidity"] += 1; continue
        if reject_yellow and rgb_arr is not None:
            h, s, _ = _mask_mean_hsv(rgb_arr, m)
            if yellow_h_lo <= h <= yellow_h_hi and s >= yellow_s_min:
                keep[i] = False; diag["yellow_color"] += 1; continue
    return keep, diag, areas


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

    def step(self, boxes, scores, areas=None) -> list[int]:
        """Update with one frame's detections. Returns track_id per detection.
        Optional `areas` updates each track's max_area, used downstream for the
        density-based individual-flower count."""
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
                if areas is not None and best_d < len(areas):
                    a = int(areas[best_d])
                    if a > t.get("max_area", 0):
                        t["max_area"] = a

        for d in range(n):
            if used[d]:
                continue
            tid = self.next_id
            self.next_id += 1
            a0 = int(areas[d]) if (areas is not None and d < len(areas)) else 0
            self.tracks[tid] = {
                "bbox": tuple(float(x) for x in boxes[d]),
                "first_frame": self.frame,
                "last_frame": self.frame,
                "n_frames": 1,
                "max_score": float(scores[d]) if scores is not None else 0.0,
                "max_area": a0,
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
    ap.add_argument("--frame-range", nargs=2, type=int, metavar=("START", "END"),
                    help="Process consecutive frames imgs[START:END] from each "
                         "session instead of even sampling. Tracker-friendly "
                         "subset. Overrides --sample-per-session when set.")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--all-folders", action="store_true",
                    help="Search every leaf folder, not just RGB/.")
    ap.add_argument("--require-all-modalities", action="store_true",
                    help="Skip any frame whose timestamp doesn't have matching "
                         "files in all four folders (RGB, depth, IR, PRGB). "
                         "Guarantees strict cross-modality alignment for the run.")
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
    # Tree-mask canopy filter (port of sprayer_pipeline/tree_mask.py).
    ap.add_argument("--tree-mask", action="store_true",
                    help="Apply a row-banded canopy mask derived from depth and reject "
                         "SAM 3 detections that fall on ground/sky/background instead "
                         "of canopy. Tight port of sprayer_pipeline/tree_mask.py: "
                         "3x3 median blur on depth + depth band + row-banded ground "
                         "filter (row 3 cap = centre + 500 mm, row 4 cap = centre + "
                         "100 mm). Implies --depth.")
    ap.add_argument("--tree-mask-min-overlap", type=float, default=0.5,
                    help="Keep a detection only if >= this fraction of its mask pixels "
                         "lie inside the canopy mask (default 0.5).")
    ap.add_argument("--canopy-min-cc-area-px", type=int, default=2000,
                    help="Connected-component size filter inside compute_canopy_mask. "
                         "Canopy regions smaller than this many pixels are dropped "
                         "(default 2000). This is the gate that kills isolated "
                         "ground-flower patches (dandelions / clover / etc.) — they "
                         "form small in-band islands while real trees form large "
                         "connected canopy regions. Set to 0 to disable.")
    ap.add_argument("--canopy-max-row-width-frac", type=float, default=0.0,
                    help="Clear any image row whose canopy mask covers more than this "
                         "fraction of the frame width — long horizontal canopy bands "
                         "are usually ground/grass at a tilt (default 0 = disabled; "
                         "try 0.5 if grass is bleeding through).")
    # Per-detection ground-rejection gates from tree_mask.py: depth-spread
    # rejects flat-depth detections (artifacts / distant fragments) and the
    # row-depth correlation gate kills tilted ground (depth ramping with image-y).
    ap.add_argument("--mask-min-depth-spread-mm", type=int, default=30,
                    help="Reject a SAM 3 mask whose internal depth-spread "
                         "(max-min of valid depths inside the mask) is less than "
                         "this many mm — flat-depth detections are usually ground "
                         "patches or stereo artifacts (default 30; "
                         "matches CANOPY_MIN_DEPTH_SPREAD_MM). Set 0 to disable.")
    ap.add_argument("--mask-max-depth-row-corr", type=float, default=0.80,
                    help="Reject a mask whose internal pearson_r(image_y, depth) "
                         "is above this threshold — ground tilted away from the "
                         "camera shows depth increasing linearly with image-y, "
                         "giving |r| close to 1 (default 0.80; matches "
                         "CANOPY_MAX_DEPTH_ROW_CORRELATION). Set >=1.0 to disable.")
    # Per-tree ROI restriction via PRGB images (red bounding boxes).
    ap.add_argument("--prgb", action="store_true",
                    help="Restrict detections to within the per-tree ROIs drawn as "
                         "red rectangles on PRGB images. Each session's PRGB folder "
                         "(sibling of RGB, files named <stem>-RGB-PP.bmp) is parsed "
                         "for red boxes per frame. Detections whose mask is mostly "
                         "outside the boxes are rejected. Useful when SAM 3 catches "
                         "background-row flowers that aren't the spray target.")
    ap.add_argument("--prgb-min-overlap", type=float, default=0.5,
                    help="Keep a detection only if >= this fraction of its mask "
                         "pixels lie inside any red ROI (default 0.5).")
    ap.add_argument("--prgb-red-r-min", type=int, default=180,
                    help="Lower bound on the R channel for red-box detection "
                         "(0-255; default 180).")
    ap.add_argument("--prgb-red-gb-max", type=int, default=80,
                    help="Upper bound on G and B channels for red-box detection "
                         "(0-255; default 80).")
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
    # Domain-specific quality filter for flower/blossom prompts (mirrors
    # sprayer_pipeline/flower_detector.py constants: clusters 10-400 px, top
    # 100 rows = leaf glint, rows > 400 = grass/ground).
    ap.add_argument("--flower-min-area-px", type=int, default=10,
                    help="Drop flower detections smaller than this many pixels (default 10).")
    ap.add_argument("--flower-max-area-px", type=int, default=2000,
                    help="Drop flower detections larger than this many pixels "
                         "(default 2000; classical detector uses 400 for clusters but "
                         "SAM 3 may merge multiple clusters into one detection).")
    ap.add_argument("--flower-y-min", type=int, default=100,
                    help="Drop flower detections whose mask centroid is above row Y "
                         "(default 100; rejects waxy leaf-glint band at the top of the "
                         "frame). Set to 0 to disable.")
    ap.add_argument("--flower-y-max", type=int, default=400,
                    help="Drop flower detections whose mask centroid is below row Y "
                         "(default 400; rejects ground/grass band at the bottom of the "
                         "frame). Set to image height (e.g. 480) to disable.")
    ap.add_argument("--flower-min-circularity", type=float, default=0.25,
                    help="4*pi*A/P^2 floor; rejects ragged blobs (default 0.25, "
                         "matches flower_detector.py).")
    ap.add_argument("--flower-min-solidity", type=float, default=0.50,
                    help="A/hull_area floor; rejects elongated/concave shapes (default "
                         "0.50, matches flower_detector.py).")
    ap.add_argument("--flower-edge-margin-px", type=int, default=15,
                    help="Reject flower bboxes that come within this many pixels of "
                         "any frame edge (default 15; D455 stereo edge artefacts).")
    ap.add_argument("--flower-area-per-flower-px", type=int, default=200,
                    help="Average pixels per individual blossom; used to estimate "
                         "n_individual_flowers from cluster area as "
                         "max(1, round(area / N)). Default 200 matches "
                         "flower_detector.py's area_per_flower constant.")
    # Yellow-flower rejection (kills dandelions on the ground, which are bright
    # yellow vs. apple blossoms which are white/pink). Mirrors the H/S logic
    # from flower_detector.py.
    ap.add_argument("--flower-reject-yellow", action="store_true", default=True,
                    help="Reject flower detections whose mean HSV color is bright "
                         "yellow (dandelions / ground-cover flowers). On by default.")
    ap.add_argument("--no-flower-reject-yellow", action="store_false",
                    dest="flower_reject_yellow",
                    help="Disable the yellow-color rejection.")
    ap.add_argument("--flower-yellow-hue-min", type=int, default=15,
                    help="Lower bound of yellow hue in OpenCV HSV (0-179; default 15).")
    ap.add_argument("--flower-yellow-hue-max", type=int, default=45,
                    help="Upper bound of yellow hue in OpenCV HSV (0-179; default 45).")
    ap.add_argument("--flower-yellow-sat-min", type=int, default=80,
                    help="Minimum mean saturation for a detection to be classified "
                         "yellow (0-255; default 80; below this it's near-white "
                         "and treated as a possible apple blossom).")
    # Cluster splitting via marker-controlled watershed (true per-blossom
    # segmentation, not the area/200 density estimate).
    ap.add_argument("--split-clusters", action="store_true",
                    help="For each surviving flower mask larger than 2x "
                         "area_per_flower, run marker-controlled watershed "
                         "segmentation (distance transform + intensity peaks + "
                         "OTSU gate + Sobel edge barriers) to split clusters into "
                         "individual blossom masks. Each sub-mask becomes its own "
                         "detection (own bbox, own track).")
    ap.add_argument("--split-min-blossom-area-px", type=int, default=30,
                    help="Drop a watershed sub-mask smaller than this (default 30 px; "
                         "rejects watershed slivers).")
    ap.add_argument("--split-min-marker-distance-px", type=int, default=5,
                    help="Minimum pixel separation between two blossom-center "
                         "markers before they get merged (default 5).")
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
        tracked_prompts = [p for p in prompts if ("flower" in p.lower() or "blossom" in p.lower())]
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
    # n_detections is the post-filter count (after --depth + tree-mask + flower
    # quality filter). n_raw is always SAM 3's raw count.
    # est_flowers is the density-based individual blossom estimate
    # (sum over kept detections of max(1, round(area / area_per_flower))).
    fieldnames = ["day", "category", "session", "image", "prompt",
                  "n_detections", "n_raw", "est_flowers",
                  "mean_score", "max_score", "elapsed_s",
                  "near_frac_mean", "near_frac_max",
                  "canopy_overlap_mean", "roi_overlap_mean", "track_ids"]
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

    # Cumulative per-session rejection diagnostics for flower prompts.
    rejection_totals: dict[tuple, dict] = {}

    def flush_session_trackers(session_key, trackers):
        # trackers may not include every prompt (only tracked_set was created),
        # which is fine — we just emit summaries for the prompts present.
        d, c, s = session_key
        per = args.flower_area_per_flower_px
        for p, tr in trackers.items():
            min_f = args.track_min_frames
            kept = [t for t in tr.tracks.values() if t["n_frames"] >= min_f]
            est_unique_flowers = sum(
                max(1, round(t.get("max_area", 0) / per)) for t in kept
            ) if (("flower" in p.lower()) or ("blossom" in p.lower())) else len(kept)
            track_summaries.append({
                "day": d, "category": c, "session": s, "prompt": p,
                "n_unique": tr.n_unique(min_frames=min_f),
                "n_unique_any": tr.n_unique(min_frames=1),
                "n_tracks_total": len(tr.tracks),
                "est_unique_flowers": est_unique_flowers,
            })
            for trk in tr.summary():
                area = int(trk.get("max_area", 0))
                est_flw = (max(1, round(area / per))
                           if (("flower" in p.lower()) or ("blossom" in p.lower()))
                           else 1)
                track_details.append({
                    "day": d, "category": c, "session": s, "prompt": p,
                    "track_id": trk["track_id"],
                    "first_frame": trk["first_frame"],
                    "last_frame": trk["last_frame"],
                    "n_frames": trk["n_frames"],
                    "max_score": round(trk["max_score"], 4),
                    "max_area_px": area,
                    "est_flowers": est_flw,
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
            frame_range=tuple(args.frame_range) if args.frame_range else None,
            require_all_modalities=args.require_all_modalities,
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
                canopy_mask_img = None
                if args.depth or args.tree_mask:
                    depth_mm = load_depth_mm(depth_path_for(img_path), (img.height, img.width))
                if args.tree_mask and depth_mm is not None:
                    canopy_mask_img = compute_canopy_mask(
                        depth_mm,
                        min_mm=args.depth_min_mm, max_mm=args.depth_max_mm,
                        min_cc_area_px=args.canopy_min_cc_area_px,
                        max_row_width_frac=args.canopy_max_row_width_frac,
                    )

                # Optional PRGB ROI mask — red rectangles per tree.
                roi_mask_img = None
                if args.prgb:
                    roi_mask_img = extract_roi_mask(
                        prgb_path_for(img_path),
                        (img.height, img.width),
                        red_r_min=args.prgb_red_r_min,
                        red_gb_max=args.prgb_red_gb_max,
                    )

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

                    # Per-detection ground rejection: depth-spread + row-depth
                    # correlation. Applies to ALL prompts. Catches ground/floor
                    # patches that survived the canopy mask (because their depth
                    # was inside the band).
                    if (depth_mm is not None and n > 0
                            and (args.mask_min_depth_spread_mm > 0
                                 or args.mask_max_depth_row_corr < 1.0)):
                        keep = np.ones(n, dtype=bool)
                        diag_g = {"depth_spread": 0, "depth_row_corr": 0}
                        for i, m in enumerate(masks_np):
                            spread, corr, _ = mask_depth_geom(m, depth_mm)
                            if (args.mask_min_depth_spread_mm > 0
                                    and spread < args.mask_min_depth_spread_mm):
                                keep[i] = False
                                diag_g["depth_spread"] += 1
                                continue
                            if (args.mask_max_depth_row_corr < 1.0
                                    and corr > args.mask_max_depth_row_corr):
                                keep[i] = False
                                diag_g["depth_row_corr"] += 1
                        rt_key = (day, category, session, prompt)
                        rt = rejection_totals.setdefault(rt_key, {})
                        for k, v in diag_g.items():
                            rt[k] = rt.get(k, 0) + v
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

                    # PRGB ROI filter — restrict detections to inside the
                    # per-tree red boxes. Applies to ALL prompts.
                    roi_overlap_mean: float | str = ""
                    if args.prgb and roi_mask_img is not None and n > 0:
                        ovs = roi_overlap_per_mask(masks_np, roi_mask_img)
                        ov_arr = np.asarray(ovs, dtype=float)
                        keep = ~np.isnan(ov_arr) & (ov_arr >= args.prgb_min_overlap)
                        rt_key = (day, category, session, prompt)
                        rt = rejection_totals.setdefault(rt_key, {})
                        rt["prgb_roi"] = rt.get("prgb_roi", 0) + int((~keep).sum())
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
                        kept_ov = ov_arr[keep]
                        if kept_ov.size and not np.all(np.isnan(kept_ov)):
                            roi_overlap_mean = round(float(np.nanmean(kept_ov)), 4)

                    # Tree-mask canopy filter (applies to ALL prompts — every
                    # category benefits from rejecting ground/sky/bg).
                    canopy_overlap_mean: float | str = ""
                    if args.tree_mask and canopy_mask_img is not None and n > 0:
                        overlaps = canopy_overlap_per_mask(masks_np, canopy_mask_img)
                        ov_arr = np.asarray(overlaps, dtype=float)
                        keep = ~np.isnan(ov_arr) & (ov_arr >= args.tree_mask_min_overlap)
                        # Roll up tree-mask rejections per (session, prompt).
                        rt_key = (day, category, session, prompt)
                        rt = rejection_totals.setdefault(rt_key, {})
                        rt["tree_mask"] = rt.get("tree_mask", 0) + int((~keep).sum())
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
                        kept_ov = ov_arr[keep]
                        if kept_ov.size and not np.all(np.isnan(kept_ov)):
                            canopy_overlap_mean = round(
                                float(np.nanmean(kept_ov)), 4
                            )

                    # Domain-specific flower quality filter: cluster size +
                    # centroid y + bbox edge margin + circularity + solidity,
                    # all matching constants from flower_detector.py.
                    is_flower_prompt = ("flower" in prompt.lower()
                                        or "blossom" in prompt.lower())
                    kept_areas: list[int] = []
                    if is_flower_prompt and n > 0:
                        rgb_arr = np.asarray(img)
                        keep, diag, areas_in = flower_quality_keep(
                            masks_np,
                            args.flower_min_area_px, args.flower_max_area_px,
                            args.flower_y_min, args.flower_y_max,
                            args.flower_min_circularity, args.flower_min_solidity,
                            args.flower_edge_margin_px,
                            img.height, img.width,
                            rgb_arr=rgb_arr,
                            reject_yellow=args.flower_reject_yellow,
                            yellow_h_lo=args.flower_yellow_hue_min,
                            yellow_h_hi=args.flower_yellow_hue_max,
                            yellow_s_min=args.flower_yellow_sat_min,
                        )
                        # Roll up rejections per (session, prompt).
                        rt_key = (day, category, session, prompt)
                        rt = rejection_totals.setdefault(rt_key, {k: 0 for k in diag})
                        for k, v in diag.items():
                            rt[k] = rt.get(k, 0) + v
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
                        kept_areas = [a for a, k in zip(areas_in, keep) if k]

                    # Optional: split each cluster mask into per-blossom masks
                    # via marker-controlled watershed. Each sub-mask becomes a
                    # standalone detection so the tracker can ID individual
                    # blossoms across frames.
                    if args.split_clusters and is_flower_prompt and n > 0:
                        rgb_arr_for_split = (rgb_arr if 'rgb_arr' in dir()
                                             else np.asarray(img))
                        thr = 2 * args.flower_area_per_flower_px
                        new_masks: list[np.ndarray] = []
                        new_scores: list[float] = []
                        new_boxes: list[list[float]] = []
                        new_areas: list[int] = []
                        for idx, m in enumerate(masks_np):
                            mb = m.astype(bool)
                            if mb.ndim == 3:
                                mb = mb.any(axis=0)
                            cluster_area = int(mb.sum())
                            if cluster_area < thr:
                                new_masks.append(m)
                                if scores_np is not None:
                                    new_scores.append(float(scores_np[idx]))
                                if boxes_np is not None:
                                    new_boxes.append(list(boxes_np[idx]))
                                new_areas.append(cluster_area)
                                continue
                            subs = split_cluster_mask(
                                m, rgb_arr_for_split,
                                min_blossom_area_px=args.split_min_blossom_area_px,
                                min_marker_distance_px=args.split_min_marker_distance_px,
                            )
                            for sub in subs:
                                new_masks.append(sub.astype(masks_np.dtype))
                                if scores_np is not None:
                                    new_scores.append(float(scores_np[idx]))
                                ys2, xs2 = np.nonzero(sub)
                                if ys2.size:
                                    new_boxes.append([float(xs2.min()), float(ys2.min()),
                                                       float(xs2.max()), float(ys2.max())])
                                else:
                                    new_boxes.append([0.0, 0.0, 0.0, 0.0])
                                new_areas.append(int(sub.sum()))
                        if new_masks:
                            masks_np = np.stack(new_masks, axis=0)
                            scores_np = (np.asarray(new_scores, dtype=np.float32)
                                         if scores_np is not None else None)
                            boxes_np = (np.asarray(new_boxes, dtype=np.float32)
                                        if boxes_np is not None else None)
                            kept_areas = new_areas
                            n = len(new_masks)
                            mean_s = (float(np.mean(scores_np))
                                      if scores_np is not None and len(scores_np) else 0.0)
                            max_s = (float(np.max(scores_np))
                                     if scores_np is not None and len(scores_np) else 0.0)

                    # Density-based individual flower estimate per cluster.
                    if is_flower_prompt and kept_areas:
                        per = args.flower_area_per_flower_px
                        est_flowers = int(sum(max(1, round(a / per)) for a in kept_areas))
                    else:
                        est_flowers = n if not is_flower_prompt else 0

                    # Tracker step (after depth + flower-quality filters, so we
                    # only track real near-field, properly-sized flowers).
                    track_ids: list[int] = []
                    if (args.track and prompt in tracked_set
                            and current_trackers and n > 0 and boxes_np is not None):
                        track_ids = current_trackers[prompt].step(
                            boxes_np, scores_np,
                            areas=kept_areas if is_flower_prompt else None,
                        )

                    slug = prompt_slugs[prompt]
                    if args.save_masks and n > 0:
                        rel = img_path.relative_to(args.root).with_suffix(".npz")
                        mp = masks_dir / slug / rel
                        mp.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(mp, masks=masks_np.astype(bool),
                                            scores=scores_np, boxes=boxes_np)
                    # Save overlays only for tracked prompts when --track is on
                    # (so we get just flower JPGs instead of one per category).
                    save_this_overlay = args.save_overlays and n > 0 and (
                        not args.track or prompt in tracked_set
                    )
                    if save_this_overlay:
                        rel = img_path.relative_to(args.root).with_suffix(".jpg")
                        op = overlays_dir / slug / rel
                        op.parent.mkdir(parents=True, exist_ok=True)
                        overlay = make_overlay(
                            img, masks_np, boxes_np,
                            title=f"{prompt} (n={n})",
                            track_ids=track_ids if track_ids else None,
                        )
                        overlay.save(op, quality=85)

                    writer.writerow({
                        "day": day, "category": category, "session": session,
                        "image": str(img_path), "prompt": prompt,
                        "n_detections": n,
                        "n_raw": n_raw,
                        "est_flowers": est_flowers,
                        "mean_score": round(mean_s, 4),
                        "max_score": round(max_s, 4),
                        "elapsed_s": round(time.time() - t0, 3),
                        "near_frac_mean": near_mean,
                        "near_frac_max": near_max,
                        "canopy_overlap_mean": canopy_overlap_mean,
                        "roi_overlap_mean": roi_overlap_mean,
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
    flower_prompts = [p for p in prompts if ("flower" in p.lower() or "blossom" in p.lower())]
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
        tracked_flower_prompts = [p for p in tracked_prompts if ("flower" in p.lower() or "blossom" in p.lower())]
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
                     "first_frame", "last_frame", "n_frames", "max_score",
                     "max_area_px", "est_flowers"]
        with open(td_path, "w", newline="", encoding="utf-8") as df_:
            dw = csv.DictWriter(df_, fieldnames=td_fields)
            dw.writeheader()
            for r in track_details:
                dw.writerow(r)
        print(f"[done] per-track detail -> {td_path}")

    # ---------- per-(session, prompt) rejection diagnostics ----------
    if rejection_totals:
        rej_path = out_dir / "rejections.csv"
        rej_fields = ["day", "category", "session", "prompt",
                      "prgb_roi",
                      "tree_mask",
                      "depth_spread", "depth_row_corr",
                      "min_cluster_px", "max_cluster_px",
                      "top_row", "ground_row",
                      "edge_margin", "circularity", "solidity",
                      "yellow_color"]
        with open(rej_path, "w", newline="", encoding="utf-8") as rf:
            rw = csv.DictWriter(rf, fieldnames=rej_fields)
            rw.writeheader()
            for (d, c, s, p), counts in rejection_totals.items():
                row = {"day": d, "category": c, "session": s, "prompt": p}
                for k in rej_fields[4:]:
                    row[k] = counts.get(k, 0)
                rw.writerow(row)
        print(f"[done] flower-quality rejections -> {rej_path}")


if __name__ == "__main__":
    main()
