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
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Compatibility shim for CPU-only nodes (login nodes, salloc / sbatch
# without --gpus). SAM 3's source has hardcoded CUDA references
# scattered across model build AND inference paths, e.g.
#   sam3/model/position_encoding.py:55:
#       tensors = torch.zeros((1, 1) + size, device="cuda")
#   per-frame inference: tensor.cuda() and .to("cuda") method calls.
# On a CPU node these raise `RuntimeError: No CUDA GPUs are available`.
# Patching SAM 3's installed source is fragile; instead, when no GPU
# is visible, intercept torch's tensor creators AND the .cuda() / .to()
# methods on Tensor and nn.Module so explicit cuda placements transparently
# map to cpu. No-op on a real GPU node (the wrapper is never installed).
def _is_cuda_device_arg(d):
    if d is None:
        return False
    if isinstance(d, str) and d.startswith("cuda"):
        return True
    if hasattr(d, "type") and getattr(d, "type", "") == "cuda":
        return True
    return False


if not torch.cuda.is_available():
    # 1. Tensor *creators*: torch.zeros, torch.ones, etc. that take a
    #    `device=` kwarg. Remap cuda -> cpu.
    def _make_cpu_fallback(orig):
        def _patched(*args, **kwargs):
            if _is_cuda_device_arg(kwargs.get("device")):
                kwargs["device"] = "cpu"
            return orig(*args, **kwargs)
        return _patched
    for _fn_name in (
        "zeros", "ones", "empty", "rand", "randn",
        "tensor", "full", "arange", "eye", "linspace", "logspace",
        "zeros_like", "ones_like", "empty_like",
    ):
        if hasattr(torch, _fn_name):
            setattr(
                torch, _fn_name,
                _make_cpu_fallback(getattr(torch, _fn_name)),
            )

    # 2. .cuda() method on Tensor / nn.Module: make it a no-op that
    #    returns self. Inference paths often do `tensor.cuda()` to push
    #    inputs to GPU; on CPU we want the tensor to stay where it is.
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self

    # 3. .to() method: when called with a cuda device (positional or
    #    kw), substitute cpu. Preserve dtype-only calls (.to(torch.float32))
    #    and other forms unchanged.
    _orig_tensor_to = torch.Tensor.to

    def _tensor_to_patched(self, *args, **kwargs):
        if args and _is_cuda_device_arg(args[0]):
            args = ("cpu",) + tuple(args[1:])
        if _is_cuda_device_arg(kwargs.get("device")):
            kwargs["device"] = "cpu"
        return _orig_tensor_to(self, *args, **kwargs)
    torch.Tensor.to = _tensor_to_patched

    _orig_module_to = torch.nn.Module.to

    def _module_to_patched(self, *args, **kwargs):
        if args and _is_cuda_device_arg(args[0]):
            args = ("cpu",) + tuple(args[1:])
        if _is_cuda_device_arg(kwargs.get("device")):
            kwargs["device"] = "cpu"
        return _orig_module_to(self, *args, **kwargs)
    torch.nn.Module.to = _module_to_patched

    # 4. torch.cuda.* utility functions that some libraries call
    #    unconditionally for memory mgmt / sync / device queries.
    #    On a CPU-only node `torch.cuda.current_device()` and
    #    `synchronize()` raise the same "No CUDA GPUs are available"
    #    error mid-inference. Stub them to harmless no-ops or sane
    #    fallback values.
    #
    #    IMPORTANT: each stub MUST be a distinct function object,
    #    not the same `_noop` reused across attributes. torch._dynamo
    #    iterates torch's namespace at import time and asserts when
    #    the same function object is registered against multiple
    #    torch attributes with different trace rules
    #    (`AssertionError: Duplicate torch object ... with different
    #    rules`). Building each stub via a factory closure guarantees
    #    distinct objects.
    def _make_noop():
        def _stub_noop(*args, **kwargs):
            return None
        return _stub_noop

    def _make_zero():
        def _stub_zero(*args, **kwargs):
            return 0
        return _stub_zero

    for _cuda_fn, _stub_factory in (
        ("synchronize", _make_noop),
        ("empty_cache", _make_noop),
        ("set_device", _make_noop),
        ("current_device", _make_zero),
        ("device_count", _make_zero),
        ("ipc_collect", _make_noop),
        ("reset_peak_memory_stats", _make_noop),
        ("reset_max_memory_allocated", _make_noop),
    ):
        if hasattr(torch.cuda, _cuda_fn):
            try:
                # Fresh closure per attribute -> distinct function id.
                setattr(torch.cuda, _cuda_fn, _stub_factory())
            except Exception:
                pass

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
                 track_ids: list[int] | None = None,
                 tile_rects: list[tuple[int, int, int, int]] | None = None,
                 roi_mask: np.ndarray | None = None,
                 mask_color: tuple[float, float, float, float] | None = None,
                 count_label: str | None = None) -> Image.Image:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)
    ax.imshow(img)
    if roi_mask is not None and roi_mask.any():
        roi_rgba = np.zeros((*roi_mask.shape, 4), dtype=np.float32)
        roi_rgba[roi_mask] = [1.0, 1.0, 0.0, 0.10]  # faint yellow wash
        ax.imshow(roi_rgba)
    if masks is not None and len(masks) > 0:
        if mask_color is not None:
            color = np.asarray(mask_color, dtype=np.float32)
            for m in masks:
                h, w = m.shape[-2:]
                rgba = np.zeros((h, w, 4))
                rgba[m.astype(bool)] = color
                ax.imshow(rgba)
        else:
            rng = np.random.default_rng(0)
            for m in masks:
                rcol = np.concatenate([rng.random(3), [0.45]])
                h, w = m.shape[-2:]
                rgba = np.zeros((h, w, 4))
                rgba[m.astype(bool)] = rcol
                ax.imshow(rgba)
    if tile_rects:
        for (tx, ty, tw, th) in tile_rects:
            ax.add_patch(Rectangle((tx, ty), tw, th, fill=False,
                                    edgecolor="cyan", linewidth=1.0,
                                    linestyle="--", alpha=0.7))
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
    if count_label:
        # Big count badge in the top-left of the IMAGE (not the figure).
        ax.text(8, 22, count_label,
                color="white", fontsize=18, fontweight="bold",
                family="sans-serif",
                bbox=dict(facecolor="deeppink", alpha=0.9,
                          edgecolor="white", linewidth=1.5,
                          boxstyle="round,pad=0.4"))
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
# Tile-based inference: chop the frame into overlapping tiles, run SAM 3 on
# each tile, then NMS-merge the per-tile detections back to full-frame.
# Helps recall on small objects (apple blossoms) because each blossom
# occupies a larger fraction of what SAM 3 sees per pass.
# ---------------------------------------------------------------------------
def tile_coords_for(img_w: int, img_h: int, rows: int, cols: int,
                     overlap_frac: float,
                     region: tuple[int, int, int, int] | None = None
                     ) -> list[tuple[int, int, int, int]]:
    """Return a list of (x0, y0, w, h) tile rectangles in full-frame
    coordinates. If `region` is given (rx, ry, rw, rh), tile only inside
    that rectangle instead of the full frame — useful with --tile-within-roi
    to put SAM 3's perceptual budget on the actual tree. (1, 1) returns a
    single tile equal to the region (or full frame)."""
    if region is None:
        rx, ry, rw, rh = 0, 0, img_w, img_h
    else:
        rx, ry, rw, rh = region
    if rows <= 1 and cols <= 1:
        return [(rx, ry, rw, rh)]
    rows = max(rows, 1); cols = max(cols, 1)
    base_w = rw / cols
    base_h = rh / rows
    pad_w = int(base_w * overlap_frac) // 2
    pad_h = int(base_h * overlap_frac) // 2
    tiles: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(rx, int(rx + c * base_w) - pad_w)
            y0 = max(ry, int(ry + r * base_h) - pad_h)
            x1 = min(rx + rw, int(rx + (c + 1) * base_w) + pad_w)
            y1 = min(ry + rh, int(ry + (r + 1) * base_h) + pad_h)
            if x1 - x0 > 0 and y1 - y0 > 0:
                tiles.append((x0, y0, x1 - x0, y1 - y0))
    return tiles


def roi_bounding_box(roi_mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    """Tightest enclosing rectangle (x, y, w, h) around all True pixels of
    a binary mask. Returns None if the mask is empty / None."""
    if roi_mask is None or not roi_mask.any():
        return None
    ys, xs = np.nonzero(roi_mask)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def nms_indices(boxes: np.ndarray, scores: np.ndarray,
                 iou_threshold: float) -> list[int]:
    """Greedy NMS on bounding boxes. Returns kept indices, highest score first."""
    if boxes is None or len(boxes) == 0:
        return []
    if iou_threshold >= 1.0:
        return list(range(len(boxes)))
    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-s)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xa = np.maximum(b[i, 0], b[rest, 0])
        ya = np.maximum(b[i, 1], b[rest, 1])
        xb = np.minimum(b[i, 2], b[rest, 2])
        yb = np.minimum(b[i, 3], b[rest, 3])
        inter = np.maximum(0.0, xb - xa) * np.maximum(0.0, yb - ya)
        ai = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        ar = (b[rest, 2] - b[rest, 0]) * (b[rest, 3] - b[rest, 1])
        union = ai + ar - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou < iou_threshold]
    return keep


def infer_per_prompt(processor, img: Image.Image, prompts: list[str],
                      tile_rows: int, tile_cols: int,
                      tile_overlap: float, tile_nms_iou: float,
                      region: tuple[int, int, int, int] | None = None,
                      ) -> dict[str, dict]:
    """Run SAM 3 over the image. If tile_rows*tile_cols > 1, run on each
    overlapping tile and NMS-merge results in full-frame coordinates.
    If `region` is given, tiles are restricted to that rectangle so SAM 3
    only sees the area of interest (e.g. the PRGB ROI bounding box).

    Returns {prompt: {"masks": (N, H, W) bool|None,
                       "boxes": (N, 4) float|None,
                       "scores": (N,) float|None,
                       "elapsed_s": float}}"""
    img_w, img_h = img.width, img.height
    tiles = tile_coords_for(img_w, img_h, tile_rows, tile_cols, tile_overlap,
                             region=region)
    use_tiling = len(tiles) > 1 or region is not None

    # accum keeps tile-local masks (smaller arrays) until NMS picks survivors;
    # only the kept masks get embedded into full-frame buffers, saving memory.
    accum: dict[str, dict] = {p: {"masks": [], "boxes": [], "scores": [],
                                    "tile": [], "elapsed_s": 0.0}
                              for p in prompts}

    for (tx, ty, tw, th) in tiles:
        if use_tiling:
            tile_img = img.crop((tx, ty, tx + tw, ty + th))
        else:
            tile_img = img
        state = processor.set_image(tile_img)
        for prompt in prompts:
            t0 = time.time()
            processor.reset_all_prompts(state)
            o = processor.set_text_prompt(state=state, prompt=prompt)
            m = to_np(o.get("masks"))
            b = to_np(o.get("boxes"))
            s = to_np(o.get("scores"))
            accum[prompt]["elapsed_s"] += time.time() - t0
            if m is None or len(m) == 0:
                continue
            if m.ndim == 4 and m.shape[1] == 1:
                m = m.squeeze(1)
            for i in range(len(m)):
                accum[prompt]["masks"].append(m[i])  # tile-local
                if b is not None:
                    bx1, by1, bx2, by2 = b[i]
                    accum[prompt]["boxes"].append(
                        [float(bx1) + tx, float(by1) + ty,
                         float(bx2) + tx, float(by2) + ty])
                if s is not None:
                    accum[prompt]["scores"].append(float(s[i]))
                accum[prompt]["tile"].append((tx, ty, tw, th))

    result: dict[str, dict] = {}
    for prompt in prompts:
        e = accum[prompt]["elapsed_s"]
        if not accum[prompt]["masks"]:
            result[prompt] = {"masks": None, "boxes": None,
                              "scores": None, "elapsed_s": e}
            continue
        boxes_np = np.asarray(accum[prompt]["boxes"], dtype=np.float32)
        scores_np = np.asarray(accum[prompt]["scores"], dtype=np.float32)
        keep = (nms_indices(boxes_np, scores_np, tile_nms_iou)
                if use_tiling else list(range(len(boxes_np))))
        kept_masks: list[np.ndarray] = []
        for k in keep:
            tx, ty, tw, th = accum[prompt]["tile"][k]
            tm = accum[prompt]["masks"][k]
            full = np.zeros((img_h, img_w), dtype=tm.dtype)
            h_a = min(th, tm.shape[0]); w_a = min(tw, tm.shape[1])
            full[ty:ty + h_a, tx:tx + w_a] = tm[:h_a, :w_a]
            kept_masks.append(full)
        masks_np = (np.stack(kept_masks, axis=0)
                    if kept_masks
                    else np.zeros((0, img_h, img_w), dtype=bool))
        result[prompt] = {
            "masks": masks_np,
            "boxes": boxes_np[keep] if keep else boxes_np[:0],
            "scores": scores_np[keep] if keep else scores_np[:0],
            "elapsed_s": e,
        }
    return result


# ---------------------------------------------------------------------------
# Depth handling. Thresholds mirror sprayer_pipeline/config.py:
#   CANOPY_DEPTH_MIN_MM = 600, CANOPY_DEPTH_MAX_MM = 3000
# Depth .txt files are ASCII uint16 mm, stored 480 rows x 50 cols for a
# 640x480 RGB frame (horizontally decimated ~12.8x). 0 means invalid.
# ---------------------------------------------------------------------------
def depth_path_for(img_path: Path) -> Path:
    """Given .../<session>/RGB/<stem>-RGB[-BP].<ext>, return the matching
    .../<session>/depth/<stem>-Depth.<txt|bmp>. The user's VB capture code
    saves BOTH a .txt (raw mm, what bridge_server.py:1474 reads) and a .bmp
    (3-channel uint8 colormap, for human inspection only). We must prefer
    the .txt — the .bmp doesn't carry mm values."""
    session_dir = img_path.parent.parent  # drop RGB/
    stem = img_path.stem
    base = stem
    for suffix in ("-RGB-BP", "-RGB-bp", "-RGB", "-rgb"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    txt = session_dir / "depth" / f"{base}-Depth.txt"
    if txt.is_file():
        return txt
    return session_dir / "depth" / f"{base}-Depth.bmp"


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
                      min_box_side_px: int = 25,
                      dilate_px: int = 0) -> np.ndarray | None:
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
    kept_dims: list[tuple[int, int, int, int]] = []
    for cc_id in range(1, n_cc):
        x, y, w, h, area = stats[cc_id]
        if area < min_box_area_px:
            continue
        if w < min_box_side_px or h < min_box_side_px:
            continue
        roi[y:y + h, x:x + w] = True
        kept_dims.append((x, y, w, h))

    # Log the actual PRGB box dimensions once per session-folder so the user
    # can verify the ROI width matches what they expect.
    session_key = str(prgb_path.parent)
    if session_key not in _logged_prgb_dims and kept_dims:
        _logged_prgb_dims.add(session_key)
        for (x, y, w, h) in kept_dims:
            dilated_w = w + 2 * dilate_px
            dilated_h = h + 2 * dilate_px
            print(f"[prgb] {prgb_path.parent.name}: red box {w}x{h} px at "
                  f"({x},{y}); dilated ROI {dilated_w}x{dilated_h} "
                  f"(--prgb-dilate-px {dilate_px})", file=sys.stderr)

    if dilate_px > 0 and roi.any():
        k = 2 * int(dilate_px) + 1
        kernel = np.ones((k, k), np.uint8)
        roi = cv2.dilate(roi.astype(np.uint8), kernel).astype(bool)
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


# ---------------------------------------------------------------------------
# Soft-scoring helpers: contextual depth, NDVI, and a continuous quality score
# combining SAM 3 confidence, shape, color, depth, and NDVI signals. See
# compute_flower_soft_score() for the full design notes.
# ---------------------------------------------------------------------------
def load_ir(ir_path: Path, target_hw: tuple[int, int]) -> np.ndarray | None:
    """Load the IR (NIR) channel as a single-band float array in [0, 1].

    Mirrors load_depth_mm: BMP / PNG / TIFF input, single channel, upsampled
    to target_hw. Used for NDVI (positive vegetation/petal signal). Returns
    None if the file is missing or unreadable so callers can fall back to a
    depth-only scoring path."""
    if not ir_path.is_file():
        return None
    try:
        import cv2
        ir = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
        if ir is None:
            return None
        if ir.ndim == 3:
            ir = ir[..., 0]
        if ir.shape != target_hw:
            ir = cv2.resize(
                ir, (target_hw[1], target_hw[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        ir = ir.astype(np.float32)
        # Normalize to [0, 1] using dtype range. Some VB captures store
        # IR as 8-bit, others as 16-bit; auto-scale by the actual max so
        # NDVI is comparable across cameras.
        m = float(ir.max())
        if m > 1.5:
            ir /= m
        return ir
    except Exception:
        return None


def compute_ndvi(rgb_arr: np.ndarray, ir_arr: np.ndarray | None) -> np.ndarray | None:
    """NDVI = (NIR - Red) / (NIR + Red + eps), clamped to [-1, 1].

    Returns None if IR is missing. Both inputs must be aligned at the same
    H,W. RGB is uint8, IR is float in [0, 1]; we normalize R into [0, 1]
    first so the index has the right scale."""
    if ir_arr is None:
        return None
    if rgb_arr.ndim != 3 or rgb_arr.shape[2] < 3:
        return None
    red = rgb_arr[..., 0].astype(np.float32) / 255.0
    nir = ir_arr.astype(np.float32)
    if nir.shape[:2] != red.shape[:2]:
        return None
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    return np.clip(ndvi, -1.0, 1.0)


def _ring_around_mask(
    mask: np.ndarray, ring_px: int, exclude_self: bool = True
) -> np.ndarray:
    """Boolean ring of pixels JUST outside `mask` (within `ring_px`).

    Implemented as (dilate(mask) AND NOT mask). Used to sample the
    surrounding canopy / sky / ground for contextual gates."""
    import cv2 as _cv2
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    k = max(1, 2 * int(ring_px) + 1)
    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (k, k))
    dil = _cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    if exclude_self:
        return dil & ~mask
    return dil


def compute_contextual_depth_score(
    mask: np.ndarray, depth_mm: np.ndarray | None,
    ring_px: int, depth_min_mm: float, depth_max_mm: float,
    surr_min_canopy_frac: float, match_tol_mm: float,
    saturated_partial_credit: float = 0.7,
) -> float:
    """Continuous score in [0, 1] for whether `mask` sits on the canopy.

    Logic:
      1. Compute the ring around the mask (default 20 px outside).
      2. Surrounding canopy fraction = ring pixels with depth in valid
         range / total ring pixels. If too low (sky on both sides /
         ground), score = 0.
      3. If mask itself has valid depth pixels: how close is mask median
         depth to surrounding median? Within `match_tol_mm` => 1.0,
         linearly down to 0 at 2 * tol.
      4. If mask has NO valid depth (white petals saturating IR
         projector): give `saturated_partial_credit` (default 0.7)
         provided the surrounding IS canopy. Sky doesn't get this
         because step 2 already ruled it out.

    Returns 1.0 if depth_mm is None (signal absent => neutral).
    """
    if depth_mm is None:
        return 1.0
    mb = mask.astype(bool)
    if not mb.any():
        return 0.0
    valid = (depth_mm >= depth_min_mm) & (depth_mm <= depth_max_mm)
    ring = _ring_around_mask(mb, ring_px)
    if ring.sum() == 0:
        return 0.5  # mask covers the whole frame -- can't sample context
    surr_canopy_frac = float((ring & valid).sum()) / float(ring.sum())
    if surr_canopy_frac < surr_min_canopy_frac:
        return 0.0  # surrounding is sky / ground; mask isn't on canopy
    # We have canopy context. How does the mask compare?
    mask_valid = mb & valid
    if mask_valid.sum() == 0:
        # Saturated petal case: mask all-invalid but surroundings
        # confirm canopy. Partial credit -- this catches white petals
        # whose IR returns swamp the depth sensor.
        return float(saturated_partial_credit)
    mask_med = float(np.median(depth_mm[mask_valid]))
    surr_med = float(np.median(depth_mm[ring & valid]))
    diff = abs(mask_med - surr_med)
    # Smooth ramp: 0 mm diff -> 1.0; tol -> ~0.5; 2*tol -> 0.0
    if match_tol_mm <= 0:
        return 1.0 if diff == 0 else 0.0
    return float(max(0.0, 1.0 - diff / (2.0 * match_tol_mm)))


def compute_ndvi_score(
    mask: np.ndarray, ndvi_arr: np.ndarray | None, ring_px: int,
    petal_ndvi_mean: float, petal_ndvi_std: float,
    canopy_ndvi_min: float, canopy_ndvi_softness: float,
) -> float:
    """Continuous [0, 1] score combining petal-NDVI and canopy-context-NDVI.

    Real flower in canopy:
      - mask NDVI is moderate (white petals: red ~ NIR; pink: red > NIR);
        bell-curve centered at `petal_ndvi_mean`.
      - surrounding NDVI is high (canopy leaves dominate the ring);
        sigmoid above `canopy_ndvi_min`.

    Sky / cloud regions:
      - mask NDVI close to 0 (both R and NIR bright but similar) -- might
        score OK on the petal component...
      - ...but the surrounding-canopy component drops, so the combined
        score collapses. That asymmetry is what separates white petals
        from white clouds.

    Returns 1.0 if ndvi_arr is None (signal absent => neutral)."""
    if ndvi_arr is None:
        return 1.0
    mb = mask.astype(bool)
    if not mb.any():
        return 0.0
    ring = _ring_around_mask(mb, ring_px)
    mask_med = float(np.median(ndvi_arr[mb]))
    if ring.sum() > 0:
        surr_med = float(np.median(ndvi_arr[ring]))
    else:
        surr_med = mask_med
    # Petal: bell curve on mask NDVI (white -> ~0.0, pink -> ~0.1).
    if petal_ndvi_std <= 0:
        petal = 1.0 if abs(mask_med - petal_ndvi_mean) < 1e-3 else 0.0
    else:
        z = (mask_med - petal_ndvi_mean) / petal_ndvi_std
        petal = float(np.exp(-0.5 * z * z))
    # Canopy: sigmoid above threshold.
    if canopy_ndvi_softness <= 0:
        canopy = 1.0 if surr_med >= canopy_ndvi_min else 0.0
    else:
        x = (surr_med - canopy_ndvi_min) / canopy_ndvi_softness
        canopy = float(1.0 / (1.0 + np.exp(-x)))
    # Geometric mean: both must be reasonable.
    return float(np.sqrt(max(petal, 1e-6) * max(canopy, 1e-6)))


def _sigmoid(x: float, center: float, softness: float) -> float:
    if softness <= 0:
        return 1.0 if x >= center else 0.0
    return float(1.0 / (1.0 + np.exp(-(x - center) / softness)))


def compute_shape_score(
    circ: float, density: float, aspect: float,
    circ_center: float, circ_softness: float,
    density_center: float, density_softness: float,
    aspect_max: float, aspect_softness: float,
) -> float:
    """[0, 1] shape-quality score: sigmoid(circ) * sigmoid(density) *
    aspect-falloff. Each factor lets a borderline value contribute
    partially instead of binary-rejecting; combined product means a
    very weak signal in any one dimension still pulls the whole down."""
    s_circ = _sigmoid(circ, circ_center, circ_softness)
    s_dens = _sigmoid(density, density_center, density_softness)
    if aspect <= aspect_max:
        s_aspect = 1.0
    else:
        # Symmetric reverse sigmoid past the max.
        if aspect_softness <= 0:
            s_aspect = 0.0
        else:
            s_aspect = float(
                1.0 / (1.0 + np.exp((aspect - aspect_max) / aspect_softness))
            )
    return float(s_circ * s_dens * s_aspect)


def compute_flower_soft_score(
    mask: np.ndarray, sam_score: float,
    rgb_arr: np.ndarray, hsv_arr: np.ndarray,
    blossom_pix: np.ndarray | None,
    depth_mm: np.ndarray | None, ndvi_arr: np.ndarray | None,
    *,
    # Shape gate centers (sigmoids, not hard cutoffs):
    circ_center: float, circ_softness: float,
    density_center: float, density_softness: float,
    aspect_max: float, aspect_softness: float,
    # Color gate:
    color_frac_center: float, color_frac_softness: float,
    # Depth context:
    ring_px: int, depth_min_mm: float, depth_max_mm: float,
    surr_min_canopy_frac: float, match_tol_mm: float,
    # NDVI context:
    petal_ndvi_mean: float, petal_ndvi_std: float,
    canopy_ndvi_min: float, canopy_ndvi_softness: float,
    # Combination weights (geometric mean):
    w_sam: float, w_shape: float, w_color: float,
    w_depth: float, w_ndvi: float,
) -> tuple[float, dict]:
    """Final continuous quality score in [0, 1] for one flower mask.

    Returns (score, components_dict) so the diagnostics CSV can show
    per-mask why a borderline detection scored where it did. The
    weighted geometric mean (instead of arithmetic) means any one
    near-zero component drives the total to zero -- which is what we
    want for hard mismatches like 'is on sky' or 'is wrong color',
    while still letting strong evidence in some dimensions
    compensate for moderate evidence in others.
    """
    import cv2 as _cv2
    mb = mask.astype(bool)
    area = int(mb.sum())
    if area == 0:
        return 0.0, {
            "sam": 0.0, "shape": 0.0, "color": 0.0,
            "depth": 0.0, "ndvi": 0.0,
        }
    ys, xs = np.where(mb)
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    bbox_area = bw * bh
    density = float(area) / float(max(1, bbox_area))
    short = min(bw, bh)
    long_ = max(bw, bh)
    aspect = float(long_) / float(max(1, short))
    contours, _ = _cv2.findContours(
        mb.astype(np.uint8), _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        biggest = max(contours, key=_cv2.contourArea)
        perim = _cv2.arcLength(biggest, True)
        circ = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
    else:
        circ = 0.0
    # Color fraction (blossom-color pixels / mask pixels).
    if blossom_pix is not None:
        color_frac = float((mb & blossom_pix).sum()) / float(area)
    else:
        color_frac = 1.0
    s_sam = float(np.clip(sam_score, 0.0, 1.0))
    s_shape = compute_shape_score(
        circ, density, aspect,
        circ_center, circ_softness,
        density_center, density_softness,
        aspect_max, aspect_softness,
    )
    s_color = _sigmoid(color_frac, color_frac_center, color_frac_softness)
    s_depth = compute_contextual_depth_score(
        mb, depth_mm, ring_px,
        depth_min_mm, depth_max_mm,
        surr_min_canopy_frac, match_tol_mm,
    )
    s_ndvi = compute_ndvi_score(
        mb, ndvi_arr, ring_px,
        petal_ndvi_mean, petal_ndvi_std,
        canopy_ndvi_min, canopy_ndvi_softness,
    )
    # Weighted geometric mean. Floor each component at 1e-6 so log()
    # doesn't blow up; in practice a 0.0 means "extreme mismatch" and
    # the resulting score will be ~0.
    weights = [w_sam, w_shape, w_color, w_depth, w_ndvi]
    components = [s_sam, s_shape, s_color, s_depth, s_ndvi]
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0, {
            "sam": s_sam, "shape": s_shape, "color": s_color,
            "depth": s_depth, "ndvi": s_ndvi,
        }
    log_score = sum(
        w * float(np.log(max(c, 1e-6)))
        for w, c in zip(weights, components)
    ) / total_w
    score = float(np.exp(log_score))
    return score, {
        "sam": s_sam, "shape": s_shape, "color": s_color,
        "depth": s_depth, "ndvi": s_ndvi,
    }


_warned_bad_depth_bmps: set[str] = set()
_logged_prgb_dims: set[str] = set()


# ---------------------------------------------------------------------------
# Per-mask rejection audit. Tracks WHICH original SAM detection got rejected
# by WHICH filter, so we can (a) emit a per-mask JSONL diagnostic log and
# (b) render a debug overlay that color-codes every detection by its outcome.
# Replaces the lossy "masks_np = masks_np[keep]" pattern at every filter site
# with a call that also records rejections by their pre-filter SAM indices.
# ---------------------------------------------------------------------------
class MaskAudit:
    """Track per-mask rejection reasons through a chain of boolean keep
    arrays. Initialized once per (frame, prompt) before any filtering.
    Each .apply(keep, stage) call records, for every False entry in
    `keep`, that the corresponding ORIGINAL SAM mask was rejected at
    `stage` -- only the FIRST stage that rejects a mask is kept (later
    filters never see already-rejected masks anyway).
    """

    def __init__(self, n_original: int):
        # `surviving` maps current array indices -> original SAM indices.
        # Starts as identity; every .apply(keep, ...) compresses it to
        # the same length as the surviving masks.
        self.n_original = int(n_original)
        self.surviving = np.arange(int(n_original), dtype=np.int64)
        self.rejected_by: dict[int, str] = {}
        # Optional per-mask metadata captured at specific stages
        # (refined area, soft score, etc.), keyed by original index.
        self.meta: dict[int, dict] = {}

    def apply(self, keep: np.ndarray, stage: str) -> None:
        """Record rejections for the False entries in `keep` and shrink
        the surviving-index map to match. Idempotent if keep.all().
        """
        if keep is None or len(keep) == 0:
            return
        if not isinstance(keep, np.ndarray):
            keep = np.asarray(keep, dtype=bool)
        if keep.dtype != bool:
            keep = keep.astype(bool)
        if len(keep) != len(self.surviving):
            # Defensive: if a caller passed the wrong-length keep array,
            # don't corrupt the audit. Skip and warn once.
            print(
                f"[warn] MaskAudit.apply size mismatch: keep={len(keep)} "
                f"surviving={len(self.surviving)} stage={stage!r}",
                file=sys.stderr,
            )
            return
        if not keep.all():
            for i in np.where(~keep)[0]:
                orig_i = int(self.surviving[int(i)])
                if orig_i not in self.rejected_by:
                    self.rejected_by[orig_i] = stage
        self.surviving = self.surviving[keep]

    def force_drop_remaining(self, stage: str) -> None:
        """Mark every currently-surviving mask as rejected at this
        stage. Used when a code path empties masks_np unconditionally
        (e.g. the 'all SAM masks lacked enough blossom-color content'
        branch in flower refinement)."""
        for orig_i in self.surviving:
            i = int(orig_i)
            if i not in self.rejected_by:
                self.rejected_by[i] = stage
        self.surviving = np.array([], dtype=np.int64)

    def remap_after_split(self, parent_idx_per_new: list[int]) -> None:
        """Cluster splitting (e.g. watershed) replaces N parent masks
        with M children where each child's parent is in
        `parent_idx_per_new` (length M). The audit's `surviving` array
        currently has the parents; this method rewrites it to the
        children, mapping each child to its parent's original SAM idx.
        After this call, downstream .apply() calls operate on children.
        """
        if len(self.surviving) == 0:
            return
        new_surviving = np.array(
            [int(self.surviving[int(p)]) for p in parent_idx_per_new],
            dtype=np.int64,
        )
        self.surviving = new_surviving

    def set_meta(self, current_idx: int, **kw) -> None:
        """Attach metadata to the original mask at `current_idx` in the
        currently-surviving array (e.g., refined_area, soft_score)."""
        if 0 <= current_idx < len(self.surviving):
            orig_i = int(self.surviving[int(current_idx)])
            self.meta.setdefault(orig_i, {}).update(kw)

    def kept_originals(self) -> list[int]:
        return [int(i) for i in self.surviving]


def assign_centroids_to_zones(
    masks_np: np.ndarray, roi_mask_img: np.ndarray | None,
    n_cols: int = 2, n_rows: int = 5,
) -> tuple[list[tuple[int, int] | None], tuple[int, int, int, int] | None]:
    """For each mask, return the (col, row) zone its centroid lands in,
    or None if the mask is empty / outside the ROI bounding box.

    Zones tile the PRGB ROI's bounding box as `n_cols` x `n_rows`
    cells (default 2 columns wide, 5 rows tall -- matches the spray
    pipeline's per-tree treatment grid). Returns the per-mask zone
    list AND the (rx0, ry0, roi_w, roi_h) bbox used so the CSV writer
    can flag images where the ROI was missing entirely.
    """
    if roi_mask_img is None or not roi_mask_img.any():
        return [None] * len(masks_np), None
    roi_ys, roi_xs = np.where(roi_mask_img)
    rx0 = int(roi_xs.min())
    rx1 = int(roi_xs.max())
    ry0 = int(roi_ys.min())
    ry1 = int(roi_ys.max())
    roi_w = rx1 - rx0 + 1
    roi_h = ry1 - ry0 + 1
    if roi_w < n_cols or roi_h < n_rows:
        return [None] * len(masks_np), (rx0, ry0, roi_w, roi_h)
    cell_w = roi_w / float(n_cols)
    cell_h = roi_h / float(n_rows)
    zones: list[tuple[int, int] | None] = []
    for m in masks_np:
        mb = m.astype(bool)
        if mb.ndim == 3:
            mb = mb.any(axis=0)
        if not mb.any():
            zones.append(None)
            continue
        ys, xs = np.where(mb)
        cx = float(xs.mean())
        cy = float(ys.mean())
        # Clip to ROI box; centroid outside ROI -> no zone (the prgb
        # centroid gate should already have rejected these, but stay
        # defensive).
        if cx < rx0 or cx > rx1 or cy < ry0 or cy > ry1:
            zones.append(None)
            continue
        col = int((cx - rx0) // cell_w)
        row = int((cy - ry0) // cell_h)
        col = max(0, min(n_cols - 1, col))
        row = max(0, min(n_rows - 1, row))
        zones.append((col, row))
    return zones, (rx0, ry0, roi_w, roi_h)


def zone_count_csv_keys(n_cols: int = 2, n_rows: int = 5) -> list[str]:
    """CSV column names for per-zone flower counts. Order:
    flowers_c0_r0, flowers_c0_r1, ..., flowers_c1_r4.
    Stable so downstream readers can rely on it."""
    keys: list[str] = []
    for c in range(n_cols):
        for r in range(n_rows):
            keys.append(f"flowers_c{c}_r{r}")
    return keys


# Distinct, high-contrast color per rejection stage. RGB tuples.
# Order roughly matches the filter chain. Anything not listed falls
# back to grey so the user can spot novel stages.
_DEBUG_STAGE_COLORS: dict[str, tuple[int, int, int]] = {
    # Geometric / dataset-level
    "near_field":       (255, 165,   0),   # orange
    "depth_spread":     (255,  99,  71),   # tomato
    "depth_row_corr":   (220,  20,  60),   # crimson
    "on_smooth_plane":  (139,   0,   0),   # dark red
    "prgb_roi":         (255, 215,   0),   # gold
    "tree_mask":        (160,  82,  45),   # sienna
    # Refinement
    "refine_empty":     (105, 105, 105),   # dim grey
    "refine_min_area":  (255,   0,   0),   # red
    "refine_aspect":    (148,   0, 211),   # dark violet
    "refine_other":     (128, 128, 128),   # grey
    # Post-refinement
    "prgb_centroid":    (255, 140,   0),   # dark orange
    "sky_depth":        ( 65, 105, 225),   # royal blue
    "soft_score":       (138,  43, 226),   # blue violet
    "flower_quality":   (199,  21, 133),   # medium violet red
}
_DEBUG_KEPT_COLOR = (255, 105, 180)        # hot pink
_DEBUG_KEPT_OUTLINE = (50, 205,  50)       # lime green
_DEBUG_FALLBACK_COLOR = (180, 180, 180)


def _stage_color(stage: str) -> tuple[int, int, int]:
    return _DEBUG_STAGE_COLORS.get(stage, _DEBUG_FALLBACK_COLOR)


def _render_debug_overlay(
    img: "Image.Image", orig_masks: np.ndarray,
    rejected_by: dict[int, str], kept_set: set[int],
    out_path: Path, title: str,
    roi_mask: np.ndarray | None = None,
    upscale: int = 2,
) -> None:
    """Save a debug JPG of every PRE-filter SAM mask, with a side
    legend so the image is actually readable.

    Layout:
      [ image, upscaled 2x ] [ legend column ]

    Image:
      - kept masks: pink translucent fill, lime outline.
      - rejected masks: stage-specific colored outline + a small
        numbered tag at the centroid (the original SAM index).

    Legend column (right):
      - Title line.
      - For each rejection stage that fired: colored swatch, stage
        name, count (e.g. "(3)").
      - "kept" entry with its own color and count.
      - One line per rejected mask: "#idx stage" so you can find a
        specific detection by number on the image.

    Upscales the source image 2x before drawing so labels fit.
    """
    import cv2 as _cv2_dbg
    base = np.asarray(img.convert("RGB")).copy()
    h, w = base.shape[:2]
    if upscale != 1:
        base = _cv2_dbg.resize(
            base, (w * upscale, h * upscale),
            interpolation=_cv2_dbg.INTER_NEAREST,
        )
    overlay_img = base.copy()
    H, W = overlay_img.shape[:2]
    # Tint the ROI faintly so the user can see context.
    if roi_mask is not None and roi_mask.shape[:2] == (h, w):
        roi_up = (
            _cv2_dbg.resize(
                roi_mask.astype(np.uint8),
                (W, H),
                interpolation=_cv2_dbg.INTER_NEAREST,
            ).astype(bool)
        )
        tint = overlay_img.copy()
        tint[roi_up] = (
            tint[roi_up] * 0.88 + np.array([255, 255, 0]) * 0.12
        ).astype(np.uint8)
        overlay_img = tint

    def _upscale_mask(m: np.ndarray) -> np.ndarray:
        if upscale == 1:
            return m
        return _cv2_dbg.resize(
            m.astype(np.uint8),
            (W, H),
            interpolation=_cv2_dbg.INTER_NEAREST,
        ).astype(bool)

    # Pink fill for kept masks (alpha-blended), lime outline.
    for orig_i in sorted(kept_set):
        if orig_i >= len(orig_masks):
            continue
        m = orig_masks[int(orig_i)].astype(bool)
        if m.ndim == 3:
            m = m.any(axis=0)
        if not m.any():
            continue
        m_up = _upscale_mask(m)
        fill = overlay_img.copy()
        fill[m_up] = (
            fill[m_up] * 0.55
            + np.array(_DEBUG_KEPT_COLOR) * 0.45
        ).astype(np.uint8)
        overlay_img = fill
        contours, _ = _cv2_dbg.findContours(
            m_up.astype(np.uint8),
            _cv2_dbg.RETR_EXTERNAL,
            _cv2_dbg.CHAIN_APPROX_SIMPLE,
        )
        _cv2_dbg.drawContours(
            overlay_img, contours, -1, _DEBUG_KEPT_OUTLINE, 2,
        )
        ys_k, xs_k = np.where(m_up)
        cy_k = int(ys_k.mean())
        cx_k = int(xs_k.mean())
        _draw_index_tag(
            overlay_img, cx_k, cy_k, str(int(orig_i)),
            _DEBUG_KEPT_OUTLINE,
        )

    # Stage-colored outline + numbered tag for rejected masks.
    for orig_i in sorted(rejected_by.keys()):
        reason = rejected_by[orig_i]
        if orig_i >= len(orig_masks):
            continue
        m = orig_masks[int(orig_i)].astype(bool)
        if m.ndim == 3:
            m = m.any(axis=0)
        if not m.any():
            continue
        m_up = _upscale_mask(m)
        color = _stage_color(reason)
        contours, _ = _cv2_dbg.findContours(
            m_up.astype(np.uint8),
            _cv2_dbg.RETR_EXTERNAL,
            _cv2_dbg.CHAIN_APPROX_SIMPLE,
        )
        _cv2_dbg.drawContours(overlay_img, contours, -1, color, 2)
        ys_d, xs_d = np.where(m_up)
        cy = int(ys_d.mean())
        cx = int(xs_d.mean())
        _draw_index_tag(overlay_img, cx, cy, str(int(orig_i)), color)

    # Title bar across the top of the image panel.
    bar_h = 28
    titled = np.zeros((H + bar_h, W, 3), dtype=np.uint8)
    titled[bar_h:] = overlay_img
    titled[:bar_h] = (30, 30, 30)
    _cv2_dbg.putText(
        titled, title[:120], (6, bar_h - 8),
        _cv2_dbg.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1,
        _cv2_dbg.LINE_AA,
    )

    # Build the legend column.
    legend = _build_legend_panel(
        rejected_by, kept_set, total_h=titled.shape[0],
    )
    full = np.concatenate([titled, legend], axis=1)
    bgr = _cv2_dbg.cvtColor(full, _cv2_dbg.COLOR_RGB2BGR)
    _cv2_dbg.imwrite(
        str(out_path), bgr,
        [int(_cv2_dbg.IMWRITE_JPEG_QUALITY), 88],
    )


def _draw_index_tag(
    img_rgb: np.ndarray, cx: int, cy: int, label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a small filled box with a number inside, centered at
    (cx, cy). Used to label both kept and rejected detections so
    you can cross-reference with the legend column."""
    import cv2 as _cv2_t
    h, w = img_rgb.shape[:2]
    tw = 10 + 7 * len(label)
    th = 16
    x0 = max(0, min(w - tw, cx - tw // 2))
    y0 = max(0, min(h - th, cy - th // 2))
    # Filled box with black text for legibility against any color.
    _cv2_t.rectangle(
        img_rgb, (x0, y0), (x0 + tw, y0 + th), color, thickness=-1,
    )
    _cv2_t.rectangle(
        img_rgb, (x0, y0), (x0 + tw, y0 + th), (0, 0, 0), thickness=1,
    )
    _cv2_t.putText(
        img_rgb, label, (x0 + 4, y0 + th - 4),
        _cv2_t.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, _cv2_t.LINE_AA,
    )


def _build_legend_panel(
    rejected_by: dict[int, str],
    kept_set: set[int],
    total_h: int,
    width: int = 280,
) -> np.ndarray:
    """Right-side legend column for the debug overlay. Top half =
    stage swatch + count summary; bottom half = per-mask listing
    so you can find any numbered detection in the image and read
    its rejection reason."""
    import cv2 as _cv2_l
    panel = np.full((total_h, width, 3), 245, dtype=np.uint8)
    # Header.
    _cv2_l.rectangle(panel, (0, 0), (width, 28), (30, 30, 30), -1)
    _cv2_l.putText(
        panel, "Legend (stage : count)", (6, 20),
        _cv2_l.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1,
        _cv2_l.LINE_AA,
    )
    y = 38
    # Stage counts (descending).
    counts: dict[str, int] = {}
    for orig_i, stage in rejected_by.items():
        counts[stage] = counts.get(stage, 0) + 1
    # Always show "kept" first, in green.
    items: list[tuple[str, int, tuple[int, int, int]]] = [
        ("kept", len(kept_set), _DEBUG_KEPT_OUTLINE),
    ]
    for stage, c in sorted(counts.items(), key=lambda x: -x[1]):
        items.append((stage, c, _stage_color(stage)))
    for stage, c, color in items:
        # Color swatch.
        _cv2_l.rectangle(
            panel, (8, y - 11), (24, y + 3), color, -1,
        )
        _cv2_l.rectangle(
            panel, (8, y - 11), (24, y + 3), (0, 0, 0), 1,
        )
        _cv2_l.putText(
            panel, f"{stage}: {c}", (32, y),
            _cv2_l.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1,
            _cv2_l.LINE_AA,
        )
        y += 17
        if y > total_h - 100:
            break
    # Divider.
    y += 6
    _cv2_l.line(panel, (4, y), (width - 4, y), (180, 180, 180), 1)
    y += 14
    _cv2_l.putText(
        panel, "Per-mask (#idx stage)", (6, y),
        _cv2_l.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1,
        _cv2_l.LINE_AA,
    )
    y += 14
    # Sort kept first by index, then rejected by index.
    rows: list[tuple[int, str, tuple[int, int, int]]] = []
    for orig_i in sorted(kept_set):
        rows.append((int(orig_i), "kept", _DEBUG_KEPT_OUTLINE))
    for orig_i in sorted(rejected_by.keys()):
        rows.append(
            (int(orig_i), rejected_by[orig_i],
             _stage_color(rejected_by[orig_i])),
        )
    for orig_i, stage, color in rows:
        if y > total_h - 12:
            _cv2_l.putText(
                panel, "...", (8, y),
                _cv2_l.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1,
                _cv2_l.LINE_AA,
            )
            break
        _cv2_l.rectangle(
            panel, (6, y - 10), (18, y + 2), color, -1,
        )
        _cv2_l.putText(
            panel, f"#{orig_i} {stage}", (24, y),
            _cv2_l.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1,
            _cv2_l.LINE_AA,
        )
        y += 14
    return panel


def load_depth_mm(depth_path: Path, target_hw: tuple[int, int]) -> np.ndarray | None:
    """Load a depth file and return uint16 mm upsampled to target (H, W).
    Supported:
      - .txt : ASCII (what the user's bridge_server.py reads)
      - .bmp / .png : 16-bit single-channel raw mm
    Rejects 3-channel uint8 BMPs — those are the colormap *visualization* the
    VB capture code writes for human inspection, not raw mm depth. A warning
    is printed once per session-folder so the user notices."""
    if not depth_path.is_file():
        return None
    suffix = depth_path.suffix.lower()
    try:
        if suffix == ".txt":
            d = np.loadtxt(depth_path, dtype=np.float32)
        else:
            import cv2
            d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if d is None:
                return None
            if d.ndim == 3 or d.dtype == np.uint8:
                key = str(depth_path.parent)
                if key not in _warned_bad_depth_bmps:
                    _warned_bad_depth_bmps.add(key)
                    print(f"[warn] depth at {key} is {d.shape} {d.dtype} — looks "
                          f"like the colormap visualization, not raw mm. Upload "
                          f"the matching .txt files (raw mm) instead.",
                          file=sys.stderr)
                return None
            d = d.astype(np.int32)
    except Exception:
        return None
    if d.ndim != 2 or d.size == 0:
        return None
    H, W = target_hw
    if d.shape != (H, W):
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


def local_depth_std(mask, depth_mm: np.ndarray, window_size: int = 30) -> float:
    """Std-dev of valid depth pixels in a `window_size`-radius square around
    the mask's centroid. Low values mean the surrounding region sits on a
    smooth plane (orchard floor); high values mean a 3D-structured region
    (canopy with branches at varied depths).
    Returns 0.0 if too few valid pixels to compute."""
    mb = np.asarray(mask).astype(bool)
    if mb.ndim == 3:
        mb = mb.any(axis=0)
    if mb.sum() == 0:
        return 0.0
    ys, xs = np.nonzero(mb)
    cy = int(ys.mean()); cx = int(xs.mean())
    H, W = depth_mm.shape
    y0 = max(0, cy - window_size); y1 = min(H, cy + window_size + 1)
    x0 = max(0, cx - window_size); x1 = min(W, cx + window_size + 1)
    window = depth_mm[y0:y1, x0:x1]
    valid = window[window > 0]
    if valid.size < 20:
        return 0.0
    return float(np.std(valid.astype(np.float64)))


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
                         yellow_s_min: int = 80,
                         require_blossom_color: bool = False,
                         min_blossom_color_frac: float = 0.30,
                         blossom_white_s_max: int = 30,
                         blossom_white_v_min: int = 180,
                         blossom_pink_h_lo: int = 0,
                         blossom_pink_h_hi: int = 30,
                         blossom_pink_h_lo2: int = 150,
                         blossom_pink_h_hi2: int = 179,
                         blossom_pink_s_lo: int = 20,
                         blossom_pink_s_hi: int = 100,
                         blossom_pink_v_min: int = 110,
                         max_bbox_area_px: int = 0,
                         min_mask_density: float = 0.0,
                         sam3_boxes: np.ndarray | None = None,
                         ) -> tuple[np.ndarray, dict, list[int]]:
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
            "yellow_color": 0, "non_blossom_color": 0,
            "max_bbox_area": 0, "low_density": 0}
    areas: list[int] = []

    # Pre-compute the "could be apple blossom" pixel mask once per frame
    # (white OR pink in HSV — ports the bloom-stage gates from
    # flower_detector.py as a POSITIVE color check).
    blossom_pixel_mask = None
    if require_blossom_color and rgb_arr is not None:
        import cv2
        hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
        H_ch, S_ch, V_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        white = (S_ch <= blossom_white_s_max) & (V_ch >= blossom_white_v_min)
        in_pink_hue = (((H_ch >= blossom_pink_h_lo) & (H_ch <= blossom_pink_h_hi))
                       | ((H_ch >= blossom_pink_h_lo2) & (H_ch <= blossom_pink_h_hi2)))
        pink = (in_pink_hue
                & (S_ch >= blossom_pink_s_lo) & (S_ch <= blossom_pink_s_hi)
                & (V_ch >= blossom_pink_v_min))
        blossom_pixel_mask = white | pink
    for i, m in enumerate(masks_np):
        circ, sol, area, bx, by, bw, bh, cy = _mask_shape_stats(m)
        areas.append(area)
        if area < min_area:
            keep[i] = False; diag["min_cluster_px"] += 1; continue
        if area > max_area:
            keep[i] = False; diag["max_cluster_px"] += 1; continue
        # bbox checks: prefer SAM 3's predicted box (what's drawn in the
        # overlay) over the contour-of-largest-blob bbox, because SAM 3
        # sometimes predicts a huge box around a sparse-mask detection
        # and the contour bbox of the largest blob is small + misleading.
        if sam3_boxes is not None and i < len(sam3_boxes):
            sx1, sy1, sx2, sy2 = sam3_boxes[i]
            sbw = max(0.0, float(sx2) - float(sx1))
            sbh = max(0.0, float(sy2) - float(sy1))
        else:
            sbw, sbh = float(bw), float(bh)
        if max_bbox_area_px > 0 and sbw * sbh > max_bbox_area_px:
            keep[i] = False; diag["max_bbox_area"] += 1; continue
        if min_mask_density > 0 and sbw > 0 and sbh > 0:
            density = float(area) / (sbw * sbh)
            if density < min_mask_density:
                keep[i] = False; diag["low_density"] += 1; continue
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
        if require_blossom_color and blossom_pixel_mask is not None:
            mb = np.asarray(m).astype(bool)
            if mb.ndim == 3:
                mb = mb.any(axis=0)
            total = int(mb.sum())
            if total > 0:
                frac_blossom = float((mb & blossom_pixel_mask).sum()) / float(total)
                if frac_blossom < min_blossom_color_frac:
                    keep[i] = False
                    diag["non_blossom_color"] += 1
                    continue
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
    ap.add_argument("--flower-min-local-depth-std-mm", type=float, default=0.0,
                    help="Reject a flower mask whose surrounding depth region "
                         "(30 px window around the centroid) has std-dev below "
                         "this many mm. Smooth depth around a detection means "
                         "it sits on the orchard floor plane; canopy blossoms "
                         "are surrounded by branches at varied depths and have "
                         "much higher local depth std. Try 120-150 mm. "
                         "Default 0 = disabled. Independent of y-position.")
    ap.add_argument("--flower-local-depth-window-px", type=int, default=30,
                    help="Half-width in pixels of the depth-std window around "
                         "the mask centroid (default 30, so a 61x61 px window).")
    ap.add_argument("--flower-min-valid-depth-frac", type=float, default=0.50,
                    help="Fraction of refined flower-mask pixels that must "
                         "have valid canopy depth in [min_depth_mm, "
                         "max_depth_mm]. Sky / clouds match white-blossom HSV "
                         "exactly (low S, high V) so HSV alone can't tell them "
                         "apart from petals; a real flower sits on the canopy "
                         "(returns valid 1-3 m depth) while sky returns 0 / "
                         "noise. Default 0.50 = at least half the refined "
                         "petal pixels must lie on real canopy depth. "
                         "Set 0 to disable. Requires --depth.")
    ap.add_argument("--flower-depth-min-mm", type=float, default=500.0,
                    help="Lower bound (mm) for the depth-validity gate "
                         "(default 500 = 0.5 m). Pixels below this are "
                         "treated as invalid (sky / no-return / sensor "
                         "noise).")
    ap.add_argument("--flower-depth-max-mm", type=float, default=5000.0,
                    help="Upper bound (mm) for the depth-validity gate "
                         "(default 5000 = 5 m). Pixels above are treated as "
                         "background (distant trees, buildings, sky max-range "
                         "value).")
    ap.add_argument("--flower-refine-min-area-px", type=int, default=15,
                    help="After HSV-refining a SAM flower mask to its "
                         "blossom-color subset, drop the mask if the "
                         "refined area is below this many pixels. Was "
                         "hardcoded at 80; lowering catches small/distant "
                         "blossoms at the cost of more glints. The soft-"
                         "score gate downstream is the primary glint "
                         "rejector; this just removes extreme-tiny noise.")
    ap.add_argument("--flower-refine-max-aspect", type=float, default=3.5,
                    help="Max long-axis / short-axis ratio of the refined "
                         "petal mask. Was hardcoded at 2.5; raising lets "
                         "tilted / partial-view blossoms pass. Branches "
                         "produce 5:1+ refined ratios so this still "
                         "rejects them. The soft-score shape component "
                         "also penalizes elongation continuously.")
    # ---- Soft-score (continuous quality) flower gate. -----------------
    # Combines SAM 3 confidence, shape, color, contextual depth, and NDVI
    # into a single [0,1] score via weighted geometric mean. Replaces
    # brittle hard thresholds with sigmoid-shaped components: a flower at
    # circ=0.54 with a great score and good depth context still scores
    # high; a glint at circ=0.56 with low SAM confidence and sky context
    # scores low. See compute_flower_soft_score() for design.
    ap.add_argument("--flower-min-soft-score", type=float, default=0.0,
                    help="Minimum combined quality score in [0, 1] for a "
                         "flower mask to be kept. 0 = soft scoring "
                         "disabled (back-compat). Try 0.30-0.45. "
                         "Independent of and complementary to the hard "
                         "circularity / density / depth gates: those still "
                         "catch extremes; this catches the brittle middle.")
    # Component centers (where each sigmoid crosses 0.5):
    ap.add_argument("--flower-soft-circ-center", type=float, default=0.55)
    ap.add_argument("--flower-soft-circ-softness", type=float, default=0.10)
    ap.add_argument("--flower-soft-density-center", type=float, default=0.40)
    ap.add_argument("--flower-soft-density-softness", type=float, default=0.10)
    ap.add_argument("--flower-soft-aspect-max", type=float, default=2.5)
    ap.add_argument("--flower-soft-aspect-softness", type=float, default=0.5)
    ap.add_argument("--flower-soft-color-center", type=float, default=0.10,
                    help="Blossom-color fraction sigmoid center.")
    ap.add_argument("--flower-soft-color-softness", type=float, default=0.05)
    # Contextual-depth params:
    ap.add_argument("--flower-context-ring-px", type=int, default=20,
                    help="Pixels OUTSIDE the mask to sample as 'surrounding "
                         "canopy' for both depth and NDVI context checks.")
    ap.add_argument("--flower-context-min-canopy-frac", type=float, default=0.30,
                    help="Of the surrounding ring, the fraction that must "
                         "have valid canopy depth for the mask to be "
                         "considered canopy-embedded. Sky-adjacent masks "
                         "fail this and score 0 on depth.")
    ap.add_argument("--flower-context-depth-tol-mm", type=float, default=1500.0,
                    help="Tolerance for matching mask median depth to "
                         "surrounding canopy median (mm). Within tol -> "
                         "score ~1; at 2*tol -> score 0.")
    # NDVI params:
    ap.add_argument("--flower-petal-ndvi-mean", type=float, default=0.10,
                    help="Expected median NDVI for white/pink petal pixels "
                         "(petals reflect strongly in red, so NDVI is low "
                         "but non-zero). Bell-curve center.")
    ap.add_argument("--flower-petal-ndvi-std", type=float, default=0.20,
                    help="Std-dev of the petal-NDVI bell curve. Wider = "
                         "more permissive.")
    ap.add_argument("--flower-canopy-ndvi-min", type=float, default=0.30,
                    help="Surrounding-NDVI sigmoid center: real canopy "
                         "leaves are 0.4+; below ~0.2 indicates sky / "
                         "ground.")
    ap.add_argument("--flower-canopy-ndvi-softness", type=float, default=0.10)
    # Combination weights:
    ap.add_argument("--flower-soft-w-sam", type=float, default=1.0)
    ap.add_argument("--flower-soft-w-shape", type=float, default=1.5)
    ap.add_argument("--flower-soft-w-color", type=float, default=1.5)
    ap.add_argument("--flower-soft-w-depth", type=float, default=1.5)
    ap.add_argument("--flower-soft-w-ndvi", type=float, default=1.0)
    # ---- Debug instrumentation. Both default off so production runs
    # don't accumulate large debug artifacts. -----------------------
    ap.add_argument("--debug-rejection-log", action="store_true",
                    help="Write one JSONL line per pre-filter SAM "
                         "detection to <out>/rejections_per_mask.jsonl "
                         "with score, bbox, area, kept-bool, and the "
                         "name of the FIRST gate that rejected it. "
                         "Lets you answer 'why did this specific flower "
                         "get missed?' in seconds.")
    ap.add_argument("--debug-overlay", action="store_true",
                    help="Save a second '<image>_<prompt>_debug.jpg' "
                         "per frame showing ALL pre-filter SAM "
                         "detections, color-coded by outcome: "
                         "kept = pink fill, rejected = red outline "
                         "with the rejection-stage label. Use to "
                         "visually identify what nearly survived but "
                         "got dropped, vs what passed.")
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
    ap.add_argument("--prgb-dilate-px", type=int, default=0,
                    help="Dilate the extracted ROI mask by N pixels before applying "
                         "the overlap check. Useful when SAM 3's flower masks tend "
                         "to spill past the red-box edges (default 0; try 20-40 if "
                         "many real flowers near the box edges are rejected).")
    ap.add_argument("--prgb-extend-vertical", action="store_true",
                    help="Stretch the ROI mask vertically to span the full image "
                         "height (top to bottom of frame), preserving its original "
                         "horizontal extent. Use with --tile-grid 5 2 to mirror the "
                         "sprayer pipeline's 2-column x 5-row = 10-zone canopy "
                         "discretization.")
    ap.add_argument("--prgb-extend-horizontal", action="store_true",
                    help="Stretch the ROI mask horizontally to span the full image "
                         "width, preserving its vertical extent. Combine with "
                         "--prgb-extend-vertical for a full-frame ROI.")
    ap.add_argument("--prgb-pad-px", nargs=2, type=int, default=[0, 0],
                    metavar=("PAD_X", "PAD_Y"),
                    help="Add fixed padding on each side of the ROI bounding box "
                         "(in pixels). E.g. --prgb-pad-px 100 0 expands the ROI "
                         "100 px left and right (good for capturing full canopy "
                         "width when the PRGB box only marks the trunk).")
    # Tile-based inference for higher recall on small objects (blossoms).
    ap.add_argument("--tile-grid", nargs=2, type=int, default=[1, 1],
                    metavar=("ROWS", "COLS"),
                    help="Run SAM 3 on a grid of overlapping crops, then NMS-merge "
                         "the per-tile detections. (1, 1) = single-frame inference "
                         "(default). (2, 2) = 4 tiles, ~4x slower but lifts recall "
                         "on blossoms because each one occupies a larger fraction of "
                         "the input. (3, 3) = 9 tiles, ~9x slower for further gains.")
    ap.add_argument("--tile-overlap", type=float, default=0.2,
                    help="Tile overlap fraction so detections at tile borders aren't "
                         "lost (default 0.2 = 20%% overlap on each side).")
    ap.add_argument("--tile-nms-iou", type=float, default=0.5,
                    help="IoU threshold for de-duping detections that appear in "
                         "multiple tiles (default 0.5).")
    ap.add_argument("--show-tile-grid", action="store_true",
                    help="Draw cyan dashed rectangles on overlays showing the "
                         "tile boundaries SAM 3 actually saw. Useful for "
                         "diagnosing whether missed flowers fall at tile edges.")
    ap.add_argument("--show-roi", action="store_true",
                    help="Tint the PRGB ROI mask faint yellow on overlays so "
                         "it's obvious which region is being kept by --prgb.")
    ap.add_argument("--tile-within-roi", action="store_true",
                    help="Restrict SAM 3 tile inference to the bounding box of "
                         "the dilated PRGB ROI (instead of the full frame). "
                         "Each tile then covers only tree pixels, giving small "
                         "blossoms a much larger fraction of SAM 3's input — "
                         "typically 1.5-3x better recall on small objects. "
                         "Requires --prgb. Falls back to whole-frame tiling on "
                         "frames where no ROI was extracted.")
    ap.add_argument("--skip-no-roi", action="store_true",
                    help="Skip frames where the PRGB image yielded no detectable "
                         "ROI (no red boxes found). Stops --tile-within-roi from "
                         "wasting inference on grass/sky/background when the "
                         "sprayer wasn't targeting a tree. Logs skipped count "
                         "at the end of the run.")
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
    # Positive blossom-color check: require >= N% of mask pixels to look
    # white OR pink per the bloom-stage HSV gates in flower_detector.py.
    ap.add_argument("--flower-require-blossom-color", action="store_true",
                    help="Require detected flower masks to overlap apple-blossom "
                         "colored pixels (white OR pink in HSV). Cuts SAM 3 "
                         "false positives on bark / lit leaves / signs.")
    ap.add_argument("--flower-min-blossom-color-frac", type=float, default=0.30,
                    help="Minimum fraction of mask pixels that must be blossom-"
                         "colored for the mask to survive (default 0.30).")
    # Multi-prompt union for flowers — run multiple flower-related prompts
    # and NMS-merge their detections under a single canonical 'flower' label.
    # Trades cost for recall: each prompt catches blossoms the others miss.
    ap.add_argument("--flower-multi-prompts", nargs="+", default=None,
                    help="Replace the flower prompt with this list and NMS-merge "
                         "the detections. Example: "
                         "--flower-multi-prompts flower blossom 'white flower'. "
                         "Default None = single-prompt mode.")
    # SAM 3 occasionally returns a sparse mask whose bounding box covers a
    # whole branch but whose actual mask pixels are few (so it slips through
    # max_cluster_px). These two gates target that failure mode directly.
    ap.add_argument("--flower-max-bbox-area-px", type=int, default=10000,
                    help="Reject a flower detection whose bounding-box area "
                         "exceeds this many pixels. Catches SAM 3 mask outputs "
                         "where mask_area is small but the bbox spans a whole "
                         "branch (default 10000 = ~100x100 covers any plausible "
                         "cluster). Set 0 to disable.")
    ap.add_argument("--flower-min-mask-density", type=float, default=0.20,
                    help="Reject a flower detection where mask_area / bbox_area "
                         "is below this — sparse 'fingers across a branch' SAM 3 "
                         "outputs have density <0.2; real clusters are >0.4 "
                         "(default 0.20). Set 0 to disable.")
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
    # SAM 3's build_sam3_image_model() defaults to GPU placement; on CPU-only
    # nodes (login nodes, salloc without --gpus) the inference path then hits
    # `RuntimeError: No CUDA GPUs are available` when a tensor is moved with
    # an unconditional .cuda(). Explicit .to(device) keeps everything on the
    # selected device.
    try:
        model = model.to(args.device)
    except Exception as _move_err:
        print(
            f"[warn] could not move SAM 3 model to {args.device!r}: "
            f"{_move_err!r} — continuing with default placement.",
            file=sys.stderr,
        )
    # CPU bfloat16/float32 mismatch fix. SAM 3's inference paths are
    # decorated with @torch.autocast(device_type="cuda", dtype=bfloat16);
    # on GPU autocast unifies tensor dtypes inside that scope, but on
    # CPU autocast is disabled (you'll see the
    # "CUDA is not available ... Disabling autocast" UserWarning at
    # import time) and any bfloat16 weights or buffers in the model
    # collide with float32 inputs:
    #     RuntimeError: mat1 and mat2 must have the same dtype,
    #                   but got BFloat16 and Float
    # Cast the entire model to float32 on CPU. SAME numerical results
    # at slightly higher memory + slightly slower throughput, but
    # bfloat16 matmul on CPU is software-emulated and slow anyway, so
    # float32 is usually FASTER on CPU.
    if args.device == "cpu":
        try:
            model = model.float()
        except Exception as _cast_err:
            print(
                f"[warn] could not cast SAM 3 model to float32 on "
                f"CPU: {_cast_err!r} — bfloat16/float32 mismatch may "
                f"crash inference.",
                file=sys.stderr,
            )
    processor = Sam3Processor(model, confidence_threshold=args.threshold)

    csv_path = out_dir / "results.csv"
    # n_detections is the post-filter count (after --depth + tree-mask + flower
    # quality filter). n_raw is always SAM 3's raw count.
    # est_flowers is the density-based individual blossom estimate
    # (sum over kept detections of max(1, round(area / area_per_flower))).
    # Per-zone flower count columns: 10 zones in a 2-col x 5-row grid
    # over the PRGB ROI bbox (matches the spray pipeline's per-tree
    # treatment cells). Empty for non-flower prompts. Stable order so
    # downstream readers can index by name.
    zone_cols = zone_count_csv_keys(n_cols=2, n_rows=5)
    fieldnames = ["day", "category", "session", "image", "prompt",
                  "n_detections", "n_raw", "est_flowers",
                  "mean_score", "max_score", "elapsed_s",
                  "near_frac_mean", "near_frac_max",
                  "canopy_overlap_mean", "roi_overlap_mean", "track_ids"]
    fieldnames.extend(zone_cols)
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # Optional per-mask rejection log. One JSONL line per pre-filter SAM
    # detection across all (frame, prompt) pairs. Lets you grep for a
    # specific frame and see exactly which gate dropped each mask --
    # the answer to "why did THAT flower get missed?" is a one-liner.
    rejection_log_path = out_dir / "rejections_per_mask.jsonl"
    rejection_log_f = (
        open(rejection_log_path, "w", encoding="utf-8")
        if args.debug_rejection_log else None
    )

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
    skipped_no_roi = 0
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

                # PRGB ROI loaded EARLY so its bounding box can constrain the
                # tile grid (--tile-within-roi). Without this, tiles cover
                # grass / sky and waste SAM 3 perceptual budget.
                roi_mask_img = None
                if args.prgb:
                    roi_mask_img = extract_roi_mask(
                        prgb_path_for(img_path),
                        (img.height, img.width),
                        red_r_min=args.prgb_red_r_min,
                        red_gb_max=args.prgb_red_gb_max,
                        dilate_px=args.prgb_dilate_px,
                    )
                    if roi_mask_img is not None and (
                            args.prgb_extend_vertical or args.prgb_extend_horizontal
                            or any(p > 0 for p in args.prgb_pad_px)):
                        bb = roi_bounding_box(roi_mask_img)
                        if bb is not None:
                            rx, ry, rw, rh = bb
                            pad_x, pad_y = args.prgb_pad_px
                            # Apply optional fixed horizontal/vertical padding.
                            rx = max(0, rx - pad_x)
                            rw = min(img.width - rx, rw + 2 * pad_x)
                            ry = max(0, ry - pad_y)
                            rh = min(img.height - ry, rh + 2 * pad_y)
                            # Optional full-axis stretches.
                            if args.prgb_extend_vertical:
                                ry, rh = 0, img.height
                            if args.prgb_extend_horizontal:
                                rx, rw = 0, img.width
                            new_mask = np.zeros_like(roi_mask_img)
                            new_mask[ry:ry + rh, rx:rx + rw] = True
                            roi_mask_img = new_mask
                # Skip frames without a usable ROI when the user explicitly
                # asked for ROI-bounded analysis.
                if args.skip_no_roi and args.prgb and roi_mask_img is None:
                    skipped_no_roi += 1
                    continue

                tile_region = None
                if args.tile_within_roi and roi_mask_img is not None:
                    tile_region = roi_bounding_box(roi_mask_img)

                # Determine the actual prompt list passed to SAM 3. If the
                # user enabled --flower-multi-prompts, we substitute those
                # for the canonical flower prompt and merge the results
                # back under the canonical name post-inference.
                canonical_flower = next(
                    (p for p in prompts if "flower" in p.lower() or "blossom" in p.lower()),
                    None,
                )
                if args.flower_multi_prompts and canonical_flower is not None:
                    sam3_prompts = []
                    for p in prompts:
                        if p == canonical_flower:
                            sam3_prompts.extend(args.flower_multi_prompts)
                        else:
                            sam3_prompts.append(p)
                    # Deduplicate while preserving order.
                    seen = set(); ordered = []
                    for p in sam3_prompts:
                        if p not in seen:
                            seen.add(p); ordered.append(p)
                    sam3_prompts = ordered
                else:
                    sam3_prompts = prompts

                # Run SAM 3 (single-frame OR tiled inference depending on
                # --tile-grid). Returns per-prompt detections in full-frame
                # coordinates, with per-tile NMS-merging when tiling is on.
                infer = infer_per_prompt(
                    processor, img, sam3_prompts,
                    tile_rows=int(args.tile_grid[0]),
                    tile_cols=int(args.tile_grid[1]),
                    tile_overlap=args.tile_overlap,
                    tile_nms_iou=args.tile_nms_iou,
                    region=tile_region,
                )

                # Multi-prompt union: concatenate detections from each
                # alternate flower prompt and NMS-merge under the canonical
                # flower key, then drop the alternates so the downstream
                # loop sees one "flower" entry instead of N.
                if args.flower_multi_prompts and canonical_flower is not None:
                    all_masks: list[np.ndarray] = []
                    all_boxes: list[np.ndarray] = []
                    all_scores: list[np.ndarray] = []
                    total_elapsed = 0.0
                    for p in args.flower_multi_prompts:
                        if p in infer:
                            r = infer[p]
                            total_elapsed += float(r.get("elapsed_s", 0.0))
                            if r["masks"] is not None and len(r["masks"]) > 0:
                                all_masks.append(r["masks"])
                                all_boxes.append(r["boxes"])
                                all_scores.append(r["scores"])
                            if p != canonical_flower:
                                del infer[p]
                    if all_masks:
                        cat_m = np.concatenate(all_masks, axis=0)
                        cat_b = np.concatenate(all_boxes, axis=0)
                        cat_s = np.concatenate(all_scores, axis=0)
                        keep_idx = nms_indices(cat_b, cat_s, args.tile_nms_iou)
                        infer[canonical_flower] = {
                            "masks": cat_m[keep_idx] if keep_idx else cat_m[:0],
                            "boxes": cat_b[keep_idx] if keep_idx else cat_b[:0],
                            "scores": cat_s[keep_idx] if keep_idx else cat_s[:0],
                            "elapsed_s": total_elapsed,
                        }
                    else:
                        infer[canonical_flower] = {
                            "masks": None, "boxes": None,
                            "scores": None, "elapsed_s": total_elapsed,
                        }

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

                # IR + NDVI for the soft-score NDVI component. Loaded
                # once per image and reused across prompts. NDVI gives
                # us a positive vegetation/petal signal that depth
                # alone can't (sky and white petals look identical in
                # HSV; petals reflect strongly in red so NDVI is low
                # but non-zero, while canopy leaves are NDVI > 0.4).
                ir_arr = None
                ndvi_arr = None
                if args.flower_min_soft_score > 0:
                    ir_arr = load_ir(
                        ir_path_for(img_path), (img.height, img.width),
                    )

                # PRGB ROI mask was already computed at the top of the try
                # block so its bounding box could constrain the tile grid.

                for prompt in prompts:
                    t0 = time.time()
                    masks_np = infer[prompt]["masks"]
                    boxes_np = infer[prompt]["boxes"]
                    scores_np = infer[prompt]["scores"]
                    # elapsed_s reflects per-prompt SAM 3 work across all tiles.
                    inf_elapsed = float(infer[prompt].get("elapsed_s", 0.0))

                    n_raw = 0 if masks_np is None else len(masks_np)
                    n = n_raw
                    mean_s = float(np.mean(scores_np)) if scores_np is not None and len(scores_np) else 0.0
                    max_s = float(np.max(scores_np)) if scores_np is not None and len(scores_np) else 0.0

                    # Per-(frame, prompt) rejection audit. Tracks which
                    # ORIGINAL SAM detection got rejected by which filter
                    # so we can emit the per-mask JSONL log, render the
                    # debug overlay, and answer "why was this specific
                    # flower missed?" without reverse-engineering the
                    # filter chain.
                    audit = MaskAudit(n_raw)
                    # Snapshot the pre-filter masks / scores / boxes so
                    # the debug overlay can render every detection,
                    # not just the survivors. These are references, not
                    # copies -- masks_np itself gets reassigned by the
                    # filters, but the original ndarray remains alive.
                    orig_masks = masks_np
                    orig_scores = scores_np
                    orig_boxes = boxes_np
                    # Pre-init for downstream blocks: the visualization
                    # block defines roi_mask_for_overlay only when
                    # save_this_overlay is True; the debug-overlay block
                    # runs unconditionally and would NameError on
                    # non-saved frames otherwise.
                    roi_mask_for_overlay = None

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
                        audit.apply(keep, "near_field")
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
                    is_flower_for_depth_check = ("flower" in prompt.lower()
                                                  or "blossom" in prompt.lower())
                    apply_local_depth = (
                        is_flower_for_depth_check
                        and args.flower_min_local_depth_std_mm > 0
                    )
                    if (depth_mm is not None and n > 0
                            and (args.mask_min_depth_spread_mm > 0
                                 or args.mask_max_depth_row_corr < 1.0
                                 or apply_local_depth)):
                        keep = np.ones(n, dtype=bool)
                        diag_g = {"depth_spread": 0, "depth_row_corr": 0,
                                  "on_smooth_plane": 0}
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
                                continue
                            if apply_local_depth:
                                lstd = local_depth_std(
                                    m, depth_mm,
                                    window_size=args.flower_local_depth_window_px,
                                )
                                if lstd > 0 and lstd < args.flower_min_local_depth_std_mm:
                                    keep[i] = False
                                    diag_g["on_smooth_plane"] += 1
                                    continue
                        rt_key = (day, category, session, prompt)
                        rt = rejection_totals.setdefault(rt_key, {})
                        for k, v in diag_g.items():
                            rt[k] = rt.get(k, 0) + v
                        # Map the multi-bucket diag back into the audit
                        # by stage. Each entry in `keep` already reflects
                        # the FIRST bucket that fired in the loop above
                        # (the inner `continue` after setting keep[i] =
                        # False), so re-record by rerunning the same
                        # cheap geometry once -- only on rejected ones.
                        if depth_mm is not None and not keep.all():
                            for i in np.where(~keep)[0]:
                                m = masks_np[int(i)]
                                spread, corr, _ = mask_depth_geom(m, depth_mm)
                                if (args.mask_min_depth_spread_mm > 0
                                        and spread < args.mask_min_depth_spread_mm):
                                    stage = "depth_spread"
                                elif (args.mask_max_depth_row_corr < 1.0
                                        and corr > args.mask_max_depth_row_corr):
                                    stage = "depth_row_corr"
                                else:
                                    stage = "on_smooth_plane"
                                orig_i = int(audit.surviving[int(i)])
                                if orig_i not in audit.rejected_by:
                                    audit.rejected_by[orig_i] = stage
                            # Now compress audit.surviving in lockstep
                            # with masks_np below.
                            audit.surviving = audit.surviving[keep]
                        else:
                            audit.apply(keep, "depth_geom")
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
                        audit.apply(keep, "prgb_roi")
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
                        audit.apply(keep, "tree_mask")
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
                        # Refine masks to only contain real blossom-
                        # color pixels. SAM 3 sometimes returns a
                        # mask that wraps the flower AND surrounding
                        # leaves/branches; intersecting with the
                        # white-or-pink HSV mask drops the non-
                        # blossom area so per-mask area, area-based
                        # filters, and the saved overlay reflect just
                        # the actual flower. The COUNT (one mask per
                        # SAM detection) is preserved.
                        if args.flower_require_blossom_color:
                            try:
                                import cv2 as _cv2
                                hsv_full = _cv2.cvtColor(
                                    rgb_arr, _cv2.COLOR_RGB2HSV,
                                )
                                Hc = hsv_full[..., 0]
                                Sc = hsv_full[..., 1]
                                Vc = hsv_full[..., 2]
                                # HSV thresholds match
                                # flower_quality_keep defaults.
                                # Tightened to match the
                                # flower_quality_keep defaults: real
                                # blossom petals are nearly pure
                                # white (S < 20) and bright (V > 180);
                                # pink blossoms are saturated
                                # (S >= 30). Glossy leaf highlights
                                # (S 20-50) and brownish branch tips
                                # (low-saturation red) no longer
                                # qualify.
                                # Overlap the white & pink windows
                                # at S ∈ [20, 30] so pale-pink petal
                                # edges (which fall between near-
                                # white and saturated-pink) are not
                                # lost.
                                white_mask = (
                                    (Sc <= 30)
                                    & (Vc >= 180)
                                )
                                pink_mask = (
                                    (((Hc >= 0) & (Hc <= 30))
                                     | ((Hc >= 150) & (Hc <= 179)))
                                    & ((Sc >= 20) & (Sc <= 100))
                                    & (Vc >= 130)
                                )
                                # Yellow is intentionally NOT in the
                                # blossom set: trunk bark, dry leaves,
                                # ground straw all produce yellow-
                                # range pixels and would inflate the
                                # mask. White petals + (optional) pink
                                # is enough -- a side-view blossom
                                # without visible yellow stamens
                                # still has plenty of white petal
                                # pixels for the refined mask.
                                blossom_pix = white_mask | pink_mask
                                # NO pink-content gate -- most apple
                                # cultivars produce nearly pure-
                                # white blossoms (Gala, Honeycrisp,
                                # Goldrush, etc.) with only 1-3 px
                                # of visible pink at our resolution.
                                # Requiring pink content was killing
                                # the most common bloom type. Glints
                                # are still rejected by the shape
                                # gates downstream (circularity,
                                # density, aspect ratio, min refined
                                # area).
                                # Refine each mask: keep ONLY the
                                # blossom-color intersection. Drop
                                # masks whose refined area is below
                                # 20 px -- isolated leaf-glint /
                                # branch-tip refinements rarely
                                # exceed 15 px; real blossoms
                                # (single-petal cluster: 30-300 px)
                                # easily clear 20.
                                refined: list[np.ndarray] = []
                                keep_idx: list[int] = []
                                # Small dilation kernel to grow the
                                # refined mask back out to the petal
                                # edges (HSV thresholds chop pale-
                                # pink edges when the gradient
                                # crosses thresholds). Bounded by
                                # the ORIGINAL SAM mask so dilation
                                # can't pull in surrounding leaves.
                                _dil_k = _cv2.getStructuringElement(
                                    _cv2.MORPH_ELLIPSE, (5, 5),
                                )
                                # Aspect-ratio gate on the REFINED
                                # (petal-only) mask, not the
                                # original SAM bbox. SAM 3 often
                                # wraps a single flower together
                                # with adjacent leaves, producing
                                # an elongated original bbox even
                                # when the actual blossom is round.
                                # Computing on the blossom-color
                                # intersection isolates the petal
                                # cluster's true shape, while still
                                # rejecting branches / twigs (they
                                # produce long thin bright
                                # ridges along the bark).
                                MAX_ASPECT_RATIO = float(
                                    args.flower_refine_max_aspect
                                )
                                # Per-mask rejection labels for the
                                # refinement loop, indexed by current-
                                # array position. Mapped back to
                                # original SAM idx via audit.surviving
                                # below.
                                refine_reject: dict[int, str] = {}
                                for mi in range(masks_np.shape[0]):
                                    m_orig = masks_np[mi].astype(bool)
                                    if not m_orig.any():
                                        refine_reject[mi] = "refine_empty"
                                        continue
                                    m_ref_raw = m_orig & blossom_pix
                                    refined_area = int(m_ref_raw.sum())
                                    # 80 px minimum: trunk specular
                                    # spots and leaf glints sit in
                                    # the 15-50 px range. Real
                                    # blossoms refine to 80-300+ px.
                                    if refined_area < args.flower_refine_min_area_px:
                                        refine_reject[mi] = "refine_min_area"
                                        continue
                                    # Aspect ratio on the refined
                                    # petal mass (was the original
                                    # SAM bbox -- wrong target when
                                    # SAM groups flower with
                                    # adjacent leaves).
                                    ys_r, xs_r = np.where(m_ref_raw)
                                    bw_r = xs_r.max() - xs_r.min() + 1
                                    bh_r = ys_r.max() - ys_r.min() + 1
                                    short = min(bw_r, bh_r)
                                    long_ = max(bw_r, bh_r)
                                    if (short > 0
                                            and (long_ / short) > MAX_ASPECT_RATIO):
                                        refine_reject[mi] = "refine_aspect"
                                        continue
                                    m_ref = _cv2.dilate(
                                        m_ref_raw.astype(np.uint8),
                                        _dil_k, iterations=1,
                                    ).astype(bool)
                                    # Stay inside the original SAM
                                    # mask so dilation can't claim
                                    # surrounding leaves.
                                    m_ref = m_ref & m_orig
                                    # Record the refined area on the
                                    # audit so the per-mask JSONL log
                                    # can show it.
                                    audit.set_meta(
                                        mi, refined_area_px=int(m_ref.sum()),
                                    )
                                    refined.append(m_ref)
                                    keep_idx.append(mi)
                                # Apply the refinement decisions to
                                # the audit BEFORE we mutate masks_np
                                # / scores_np / boxes_np below: build a
                                # boolean keep aligned with the
                                # current array, attach per-mask
                                # rejection labels by stage name.
                                keep_refine = np.zeros(
                                    masks_np.shape[0], dtype=bool,
                                )
                                if keep_idx:
                                    keep_refine[np.asarray(
                                        keep_idx, dtype=int)] = True
                                # Record specific reject reasons
                                # FIRST so apply() doesn't overwrite
                                # them with a generic stage name.
                                for mi_r, stage_r in refine_reject.items():
                                    if 0 <= mi_r < len(audit.surviving):
                                        orig_i = int(audit.surviving[mi_r])
                                        if orig_i not in audit.rejected_by:
                                            audit.rejected_by[orig_i] = stage_r
                                # Now compress audit.surviving in
                                # lockstep with masks_np below. Any
                                # remaining False entries (mi not in
                                # refine_reject) get "refine_other".
                                for i in np.where(~keep_refine)[0]:
                                    orig_i = int(audit.surviving[int(i)])
                                    if orig_i not in audit.rejected_by:
                                        audit.rejected_by[orig_i] = "refine_other"
                                audit.surviving = audit.surviving[keep_refine]
                                if refined:
                                    masks_np = np.stack(refined, axis=0)
                                    keep_arr = np.asarray(keep_idx, dtype=int)
                                    if scores_np is not None:
                                        scores_np = scores_np[keep_arr]
                                    # Recompute boxes from the
                                    # REFINED masks so the
                                    # downstream density and
                                    # max-bbox-area gates in
                                    # flower_quality_keep operate
                                    # on the petal region, not the
                                    # inflated original SAM bbox.
                                    # Without this, a real flower
                                    # that SAM grouped with leaves
                                    # has a tight refined mask but
                                    # the original SAM bbox, so
                                    # density = mask_area / sam_box
                                    # comes out tiny and trips the
                                    # min_mask_density 0.40 gate.
                                    new_boxes = []
                                    for ref_m in refined:
                                        ys_b, xs_b = np.where(ref_m)
                                        new_boxes.append([
                                            float(xs_b.min()),
                                            float(ys_b.min()),
                                            float(xs_b.max() + 1),
                                            float(ys_b.max() + 1),
                                        ])
                                    boxes_np = np.asarray(
                                        new_boxes, dtype=np.float32,
                                    )
                                    n = int(masks_np.shape[0])
                                else:
                                    # All SAM masks lacked enough
                                    # blossom-color content -- skip
                                    # the rest of the per-prompt
                                    # processing for this frame.
                                    masks_np = np.zeros(
                                        (0, *masks_np.shape[1:]),
                                        dtype=bool,
                                    )
                                    if scores_np is not None:
                                        scores_np = scores_np[:0]
                                    if boxes_np is not None:
                                        boxes_np = boxes_np[:0]
                                    n = 0
                            except Exception as _refine_err:
                                # Don't silently swallow refinement
                                # bugs -- log to stderr so the next
                                # NameError-style regression is
                                # visible instead of disabling the
                                # whole color refinement loop.
                                print(
                                    f"[warn] flower refinement "
                                    f"failed: {_refine_err!r}",
                                    file=sys.stderr,
                                )
                        # Post-refinement PRGB centroid gate for
                        # flowers. The earlier PRGB-overlap filter
                        # ran on the WIDE unrefined SAM masks (before
                        # the HSV intersection trimmed them to
                        # petals); a refined petal mass that ends up
                        # mostly outside the red ROI box still passed
                        # because the original wider mask had 50%+
                        # overlap. Require the refined centroid to
                        # actually sit inside the ROI so flowers that
                        # the user expects to be in-bounds aren't
                        # drawn outside the cyan zone grid.
                        if (args.prgb and roi_mask_img is not None
                                and n > 0):
                            keep_c = np.ones(n, dtype=bool)
                            roi_h_full, roi_w_full = roi_mask_img.shape[:2]
                            for ci in range(n):
                                mb = masks_np[ci].astype(bool)
                                if mb.ndim == 3:
                                    mb = mb.any(axis=0)
                                if not mb.any():
                                    keep_c[ci] = False
                                    continue
                                ys_c, xs_c = np.where(mb)
                                cy_c = int(round(float(ys_c.mean())))
                                cx_c = int(round(float(xs_c.mean())))
                                if (0 <= cy_c < roi_h_full
                                        and 0 <= cx_c < roi_w_full):
                                    if not roi_mask_img[cy_c, cx_c]:
                                        keep_c[ci] = False
                                else:
                                    keep_c[ci] = False
                            audit.apply(keep_c, "prgb_centroid")
                            if not keep_c.all():
                                rt_key = (day, category, session, prompt)
                                rt = rejection_totals.setdefault(rt_key, {})
                                rt["prgb_centroid"] = (
                                    rt.get("prgb_centroid", 0)
                                    + int((~keep_c).sum())
                                )
                                masks_np = masks_np[keep_c]
                                if scores_np is not None:
                                    scores_np = scores_np[keep_c]
                                if boxes_np is not None:
                                    boxes_np = boxes_np[keep_c]
                                n = int(keep_c.sum())
                        # Depth-validity gate. White / cloudy sky has the
                        # SAME HSV signature as a white blossom (low S,
                        # high V) -- pure pixel-color filtering can't
                        # tell them apart. The unmistakable difference
                        # is depth: a real petal sits on the canopy
                        # (returns valid 1-3 m depth) while sky returns
                        # 0 mm or noise outside the valid range. Reject
                        # any refined mask whose petal pixels don't have
                        # enough valid-depth content. Targets the
                        # specific "branch silhouetted against bright
                        # sky" failure: SAM 3 wraps the branch + the
                        # adjacent bright sky into one mask, refinement
                        # keeps just the sky portion (bright low-S
                        # pixels), which then passes shape gates because
                        # a sky patch IS blob-shaped.
                        if (args.depth and depth_mm is not None
                                and args.flower_min_valid_depth_frac > 0
                                and n > 0):
                            keep_d = np.ones(n, dtype=bool)
                            d_lo = float(args.flower_depth_min_mm)
                            d_hi = float(args.flower_depth_max_mm)
                            valid_depth_pixels = (
                                (depth_mm >= d_lo) & (depth_mm <= d_hi)
                            )
                            for di in range(n):
                                mb = masks_np[di].astype(bool)
                                if mb.ndim == 3:
                                    mb = mb.any(axis=0)
                                total_d = int(mb.sum())
                                if total_d == 0:
                                    keep_d[di] = False
                                    continue
                                valid_in_mask = int(
                                    (mb & valid_depth_pixels).sum()
                                )
                                frac_valid = (
                                    valid_in_mask / float(total_d)
                                )
                                if (frac_valid
                                        < args.flower_min_valid_depth_frac):
                                    keep_d[di] = False
                            audit.apply(keep_d, "sky_depth")
                            if not keep_d.all():
                                rt_key = (day, category, session, prompt)
                                rt = rejection_totals.setdefault(rt_key, {})
                                rt["sky_depth"] = (
                                    rt.get("sky_depth", 0)
                                    + int((~keep_d).sum())
                                )
                                masks_np = masks_np[keep_d]
                                if scores_np is not None:
                                    scores_np = scores_np[keep_d]
                                if boxes_np is not None:
                                    boxes_np = boxes_np[keep_d]
                                n = int(keep_d.sum())
                        # Soft-score gate. Continuous quality score
                        # combining SAM 3 confidence, shape, color,
                        # contextual depth, and NDVI. See
                        # compute_flower_soft_score() docstring for
                        # the design. Disabled when
                        # --flower-min-soft-score 0 (default), so
                        # this is opt-in and fully back-compat.
                        if (args.flower_min_soft_score > 0
                                and n > 0):
                            # Compute NDVI lazily once we know the
                            # gate is on AND we have masks left to
                            # score. Reuses ir_arr loaded per-image.
                            if ndvi_arr is None and ir_arr is not None:
                                ndvi_arr = compute_ndvi(rgb_arr, ir_arr)
                            # blossom_pix is only computed inside the
                            # refinement try-block; recompute it here
                            # in HSV space for the color component
                            # (cheap; once per prompt-iteration).
                            try:
                                import cv2 as _cv2_score
                                _hsv = _cv2_score.cvtColor(
                                    rgb_arr, _cv2_score.COLOR_RGB2HSV,
                                )
                                _Hc, _Sc, _Vc = (
                                    _hsv[..., 0], _hsv[..., 1], _hsv[..., 2],
                                )
                                _white = (_Sc <= 30) & (_Vc >= 180)
                                _pink = (
                                    (((_Hc >= 0) & (_Hc <= 30))
                                     | ((_Hc >= 150) & (_Hc <= 179)))
                                    & ((_Sc >= 20) & (_Sc <= 100))
                                    & (_Vc >= 130)
                                )
                                _blossom_pix = _white | _pink
                            except Exception as _bp_err:
                                # Log instead of silently disabling --
                                # we burned a lot of cycles last week
                                # on a bare `except: pass` in this
                                # exact path. compute_flower_soft_score
                                # tolerates blossom_pix=None (color
                                # component falls back to 1.0), but at
                                # least we'll see the bug in stderr.
                                print(
                                    f"[warn] soft-score blossom_pix "
                                    f"build failed: {_bp_err!r}",
                                    file=sys.stderr,
                                )
                                _hsv = None
                                _blossom_pix = None
                            keep_soft = np.ones(n, dtype=bool)
                            soft_scores_log: list[float] = []
                            for si in range(n):
                                mb = masks_np[si].astype(bool)
                                if mb.ndim == 3:
                                    mb = mb.any(axis=0)
                                sam_s = (
                                    float(scores_np[si])
                                    if scores_np is not None and si < len(scores_np)
                                    else 1.0
                                )
                                soft, _comps = compute_flower_soft_score(
                                    mb, sam_s,
                                    rgb_arr, _hsv if _hsv is not None else rgb_arr,
                                    _blossom_pix,
                                    depth_mm, ndvi_arr,
                                    circ_center=args.flower_soft_circ_center,
                                    circ_softness=args.flower_soft_circ_softness,
                                    density_center=args.flower_soft_density_center,
                                    density_softness=args.flower_soft_density_softness,
                                    aspect_max=args.flower_soft_aspect_max,
                                    aspect_softness=args.flower_soft_aspect_softness,
                                    color_frac_center=args.flower_soft_color_center,
                                    color_frac_softness=args.flower_soft_color_softness,
                                    ring_px=args.flower_context_ring_px,
                                    depth_min_mm=args.flower_depth_min_mm,
                                    depth_max_mm=args.flower_depth_max_mm,
                                    surr_min_canopy_frac=args.flower_context_min_canopy_frac,
                                    match_tol_mm=args.flower_context_depth_tol_mm,
                                    petal_ndvi_mean=args.flower_petal_ndvi_mean,
                                    petal_ndvi_std=args.flower_petal_ndvi_std,
                                    canopy_ndvi_min=args.flower_canopy_ndvi_min,
                                    canopy_ndvi_softness=args.flower_canopy_ndvi_softness,
                                    w_sam=args.flower_soft_w_sam,
                                    w_shape=args.flower_soft_w_shape,
                                    w_color=args.flower_soft_w_color,
                                    w_depth=args.flower_soft_w_depth,
                                    w_ndvi=args.flower_soft_w_ndvi,
                                )
                                soft_scores_log.append(soft)
                                # Stash the score and per-component
                                # breakdown on the audit so the
                                # rejection JSONL log can show why a
                                # specific mask scored where it did.
                                audit.set_meta(
                                    si,
                                    soft_score=float(soft),
                                    soft_components=_comps,
                                )
                                if soft < args.flower_min_soft_score:
                                    keep_soft[si] = False
                            audit.apply(keep_soft, "soft_score")
                            if not keep_soft.all():
                                rt_key = (day, category, session, prompt)
                                rt = rejection_totals.setdefault(rt_key, {})
                                rt["soft_score"] = (
                                    rt.get("soft_score", 0)
                                    + int((~keep_soft).sum())
                                )
                                masks_np = masks_np[keep_soft]
                                if scores_np is not None:
                                    scores_np = scores_np[keep_soft]
                                if boxes_np is not None:
                                    boxes_np = boxes_np[keep_soft]
                                n = int(keep_soft.sum())
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
                            require_blossom_color=args.flower_require_blossom_color,
                            min_blossom_color_frac=args.flower_min_blossom_color_frac,
                            max_bbox_area_px=args.flower_max_bbox_area_px,
                            min_mask_density=args.flower_min_mask_density,
                            sam3_boxes=boxes_np,
                        )
                        # Roll up rejections per (session, prompt).
                        rt_key = (day, category, session, prompt)
                        rt = rejection_totals.setdefault(rt_key, {k: 0 for k in diag})
                        for k, v in diag.items():
                            rt[k] = rt.get(k, 0) + v
                        # Audit: flower_quality_keep returns a multi-
                        # bucket diag, so recover per-mask reasons by
                        # inspecting which gate fired in order. We
                        # don't have direct per-mask labels here, so
                        # we recompute the cheapest discriminator: any
                        # rejected mask gets stage = "flower_quality"
                        # in the audit. A finer breakdown would
                        # require flower_quality_keep to return per-
                        # mask reasons; out of scope for this commit.
                        audit.apply(keep, "flower_quality")
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
                        # Optionally draw tile boundaries / ROI tint for debugging.
                        # Visualize the LOGICAL zones (no overlap) so adjacent
                        # tiles don't visibly intrude on each other; inference
                        # still uses --tile-overlap internally for cross-tile
                        # boundary recall.
                        tile_rects_for_overlay = None
                        if args.show_tile_grid:
                            tile_rects_for_overlay = tile_coords_for(
                                img.width, img.height,
                                int(args.tile_grid[0]), int(args.tile_grid[1]),
                                0.0,
                                region=tile_region,
                            )
                        # When --show-roi is on, also overlay the 10-zone
                        # sprayer ROI grid (2 cols x 5 rows) on the PRGB
                        # ROI's bounding box. Renders as cyan dashed
                        # rectangles same as tile_rects.
                        if (args.show_roi and roi_mask_img is not None
                                and roi_mask_img.any()):
                            roi_ys, roi_xs = np.where(roi_mask_img)
                            rx0 = int(roi_xs.min())
                            rx1 = int(roi_xs.max())
                            ry0 = int(roi_ys.min())
                            ry1 = int(roi_ys.max())
                            roi_w_px = rx1 - rx0 + 1
                            roi_h_px = ry1 - ry0 + 1
                            zone_rects = []
                            for col in range(2):
                                cell_w = roi_w_px // 2
                                cx0 = rx0 + col * cell_w
                                cx1 = (
                                    rx1 if col == 1 else cx0 + cell_w - 1
                                )
                                for row in range(5):
                                    cell_h = roi_h_px // 5
                                    cy0 = ry0 + row * cell_h
                                    cy1 = (
                                        ry1 if row == 4 else cy0 + cell_h - 1
                                    )
                                    zone_rects.append(
                                        (cx0, cy0,
                                         max(1, cx1 - cx0),
                                         max(1, cy1 - cy0))
                                    )
                            if tile_rects_for_overlay is None:
                                tile_rects_for_overlay = zone_rects
                            else:
                                tile_rects_for_overlay = (
                                    list(tile_rects_for_overlay) + zone_rects
                                )
                        roi_mask_for_overlay = (
                            roi_mask_img if args.show_roi else None
                        )
                        # For flower overlays: pink masks only — strip the
                        # green bboxes, track-id labels, and the corner count
                        # badge so the user can visually verify segmentation
                        # without anything occluding the flowers. Count
                        # remains visible via the matplotlib title.
                        overlay_mask_color = None
                        overlay_boxes = boxes_np
                        overlay_track_ids = track_ids if track_ids else None
                        if is_flower_prompt:
                            overlay_mask_color = (1.0, 0.412, 0.706, 0.5)  # hot pink, 50% alpha
                            overlay_boxes = None
                            overlay_track_ids = None
                        overlay = make_overlay(
                            img, masks_np, overlay_boxes,
                            title=f"{prompt} (n={n})",
                            track_ids=overlay_track_ids,
                            tile_rects=tile_rects_for_overlay,
                            roi_mask=roi_mask_for_overlay,
                            mask_color=overlay_mask_color,
                            count_label=None,
                        )
                        overlay.save(op, quality=85)

                    # ---------- Debug overlay (--debug-overlay) ----------
                    # Render every PRE-filter SAM detection on a separate
                    # image, color-coded: pink fill = kept, red outline +
                    # text label = rejected (with the FIRST gate that
                    # killed it). Lets you instantly see what nearly
                    # survived vs what survived.
                    if (args.debug_overlay and is_flower_prompt
                            and orig_masks is not None and len(orig_masks) > 0):
                        try:
                            debug_path = (
                                op.parent / f"{op.stem}_debug.jpg"
                                if save_this_overlay
                                else (overlays_dir / slug
                                      / img_path.relative_to(args.root).with_suffix(".jpg"))
                            )
                            debug_path.parent.mkdir(parents=True, exist_ok=True)
                            kept_set = set(audit.kept_originals())
                            _render_debug_overlay(
                                img, orig_masks, audit.rejected_by,
                                kept_set, debug_path,
                                title=f"{prompt} debug "
                                      f"(kept={len(kept_set)}, "
                                      f"rejected={len(audit.rejected_by)})",
                                roi_mask=roi_mask_for_overlay,
                            )
                        except Exception as _dbg_err:
                            print(
                                f"[warn] debug overlay failed for "
                                f"{img_path}: {_dbg_err!r}",
                                file=sys.stderr,
                            )

                    # ---------- Per-zone flower counts (CSV) ----------
                    # For flower prompts only: assign each kept mask's
                    # centroid to one of the 10 sprayer ROI zones (2
                    # cols x 5 rows over the PRGB ROI bbox) and count.
                    zone_counts = {k: "" for k in zone_cols}
                    if is_flower_prompt and n > 0 and roi_mask_img is not None:
                        zones, _ = assign_centroids_to_zones(
                            masks_np, roi_mask_img, n_cols=2, n_rows=5,
                        )
                        # Initialize zeros for flower prompts so the
                        # CSV reader can distinguish "no flowers in this
                        # zone" from "no ROI / not a flower prompt".
                        for k in zone_cols:
                            zone_counts[k] = 0
                        for z in zones:
                            if z is None:
                                continue
                            c, r = z
                            zone_counts[f"flowers_c{c}_r{r}"] = (
                                int(zone_counts[f"flowers_c{c}_r{r}"]) + 1
                            )

                    # ---------- Per-mask rejection log (JSONL) ----------
                    if (rejection_log_f is not None
                            and orig_masks is not None and len(orig_masks) > 0):
                        try:
                            kept_set_jl = set(audit.kept_originals())
                            for orig_i in range(len(orig_masks)):
                                m = orig_masks[orig_i].astype(bool)
                                if m.ndim == 3:
                                    m = m.any(axis=0)
                                area = int(m.sum())
                                if area > 0:
                                    ys_j, xs_j = np.where(m)
                                    bbox = [int(xs_j.min()), int(ys_j.min()),
                                            int(xs_j.max()), int(ys_j.max())]
                                else:
                                    bbox = [0, 0, 0, 0]
                                kept_jl = orig_i in kept_set_jl
                                rejected_by = (
                                    audit.rejected_by.get(orig_i, "")
                                    if not kept_jl else ""
                                )
                                meta = audit.meta.get(orig_i, {})
                                sam_score = (
                                    float(orig_scores[orig_i])
                                    if orig_scores is not None
                                       and orig_i < len(orig_scores)
                                    else None
                                )
                                rec = {
                                    "image": str(img_path),
                                    "prompt": prompt,
                                    "orig_idx": orig_i,
                                    "sam_score": sam_score,
                                    "bbox_xyxy": bbox,
                                    "area_px": area,
                                    "kept": bool(kept_jl),
                                    "rejected_by": rejected_by,
                                }
                                # Inline the per-mask metadata captured
                                # by audit.set_meta() at filter sites
                                # (refined area, soft score, etc.).
                                for k_meta, v_meta in meta.items():
                                    rec[k_meta] = v_meta
                                rejection_log_f.write(
                                    json.dumps(rec, default=str) + "\n"
                                )
                        except Exception as _log_err:
                            print(
                                f"[warn] rejection log write failed for "
                                f"{img_path}: {_log_err!r}",
                                file=sys.stderr,
                            )

                    writer.writerow({
                        "day": day, "category": category, "session": session,
                        "image": str(img_path), "prompt": prompt,
                        "n_detections": n,
                        "n_raw": n_raw,
                        "est_flowers": est_flowers,
                        "mean_score": round(mean_s, 4),
                        "max_score": round(max_s, 4),
                        "elapsed_s": round(inf_elapsed + (time.time() - t0), 3),
                        "near_frac_mean": near_mean,
                        "near_frac_max": near_max,
                        "canopy_overlap_mean": canopy_overlap_mean,
                        "roi_overlap_mean": roi_overlap_mean,
                        "track_ids": ";".join(str(t) for t in track_ids) if track_ids else "",
                        **zone_counts,
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
        if rejection_log_f is not None:
            try:
                rejection_log_f.close()
            except Exception:
                pass

    # Flush the final session's trackers.
    if args.track and current_session_key is not None:
        flush_session_trackers(current_session_key, current_trackers)

    dt = time.time() - start
    print(f"[done] {total_imgs} images × {len(prompts)} prompts in {dt:.1f}s -> {csv_path}")
    if skipped_no_roi:
        print(f"[done] skipped {skipped_no_roi} frames with no detectable PRGB ROI")

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
                      "depth_spread", "depth_row_corr", "on_smooth_plane",
                      "min_cluster_px", "max_cluster_px",
                      "max_bbox_area", "low_density",
                      "top_row", "ground_row",
                      "edge_margin", "circularity", "solidity",
                      "yellow_color", "non_blossom_color"]
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
