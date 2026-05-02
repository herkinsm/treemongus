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


def _is_bfloat16_dtype(d):
    """Detect bfloat16 dtype passed as kwarg or positional arg."""
    if d is None:
        return False
    try:
        if d is torch.bfloat16:
            return True
    except Exception:
        pass
    if isinstance(d, str) and d == "bfloat16":
        return True
    return False


if not torch.cuda.is_available():
    # 1. Tensor *creators*: torch.zeros, torch.ones, etc. that take a
    #    `device=` kwarg. Remap cuda -> cpu AND bfloat16 -> float32.
    #    Bfloat16 on CPU is software-emulated; it also collides with
    #    the float32 model inputs we use, causing
    #      `mat1 and mat2 must have the same dtype, but got
    #       BFloat16 and Float`
    #    inside SAM 3's forward.
    def _make_cpu_fallback(orig):
        def _patched(*args, **kwargs):
            if _is_cuda_device_arg(kwargs.get("device")):
                kwargs["device"] = "cpu"
            if _is_bfloat16_dtype(kwargs.get("dtype")):
                kwargs["dtype"] = torch.float32
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

    # 2b. .bfloat16() method on Tensor / nn.Module: remap to float32
    #     on CPU. SAM 3 has hardcoded `.bfloat16()` calls that put
    #     activations into bfloat16; on CPU those collide with our
    #     float32 weights/inputs and matmul fails.
    torch.Tensor.bfloat16 = lambda self, *args, **kwargs: self.float()
    torch.nn.Module.bfloat16 = lambda self, *args, **kwargs: self.float()

    # 2c. .pin_memory() on Tensor: SAM 3's geometry_encoders.py:648
    #     calls `scale.pin_memory().to(device=..., non_blocking=True)`
    #     to set up an async CUDA host->device transfer. Pinned
    #     memory only makes sense for CUDA; on CPU `pin_memory()`
    #     itself raises "No CUDA GPUs are available". No-op it to
    #     return self so the .to(device=cpu) downstream still runs.
    torch.Tensor.pin_memory = lambda self, *args, **kwargs: self

    # 3. .to() method: when called with a cuda device (positional or
    #    kw), substitute cpu. When called with bfloat16 dtype, sub
    #    float32. Preserve dtype-only calls (.to(torch.float32))
    #    and other forms unchanged.
    _orig_tensor_to = torch.Tensor.to

    def _tensor_to_patched(self, *args, **kwargs):
        new_args = list(args)
        if new_args and _is_cuda_device_arg(new_args[0]):
            new_args[0] = "cpu"
        # Positional dtype: torch.Tensor.to(dtype) form.
        if new_args and _is_bfloat16_dtype(new_args[0]):
            new_args[0] = torch.float32
        if _is_cuda_device_arg(kwargs.get("device")):
            kwargs["device"] = "cpu"
        if _is_bfloat16_dtype(kwargs.get("dtype")):
            kwargs["dtype"] = torch.float32
        return _orig_tensor_to(self, *new_args, **kwargs)
    torch.Tensor.to = _tensor_to_patched

    _orig_module_to = torch.nn.Module.to

    def _module_to_patched(self, *args, **kwargs):
        new_args = list(args)
        if new_args and _is_cuda_device_arg(new_args[0]):
            new_args[0] = "cpu"
        if new_args and _is_bfloat16_dtype(new_args[0]):
            new_args[0] = torch.float32
        if _is_cuda_device_arg(kwargs.get("device")):
            kwargs["device"] = "cpu"
        if _is_bfloat16_dtype(kwargs.get("dtype")):
            kwargs["dtype"] = torch.float32
        return _orig_module_to(self, *new_args, **kwargs)
    torch.nn.Module.to = _module_to_patched

    # 3b. autocast context manager: SAM 3 decorates inference methods
    #     with @torch.autocast(device_type="cuda", dtype=bfloat16).
    #     On CPU PyTorch emits a UserWarning and the context is a
    #     no-op for activations -- but the dtype kwarg can still
    #     propagate via subclassing. Force-disable autocast entirely
    #     on CPU so no decorated method tries to coerce activations
    #     into bfloat16 mid-forward.
    _orig_autocast_init = torch.amp.autocast_mode.autocast.__init__

    def _autocast_init_patched(self, *args, **kwargs):
        # Force enabled=False; preserve other args so the call shape
        # stays compatible.
        kwargs["enabled"] = False
        return _orig_autocast_init(self, *args, **kwargs)
    torch.amp.autocast_mode.autocast.__init__ = _autocast_init_patched

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

    # 5. Explicit thread count for CPU inference. PyTorch reads
    #    OMP_NUM_THREADS at import time, but on some HPC nodes the
    #    env var doesn't propagate cleanly into a SLURM batch job's
    #    Python process; setting torch.set_num_threads() directly
    #    is the only reliable way. SAM 3's transformer attention
    #    is ~3-5x faster with 8 threads than with 1.
    import os as _os_thread
    try:
        _n_threads = int(
            _os_thread.environ.get(
                "OMP_NUM_THREADS",
                _os_thread.environ.get("SLURM_CPUS_PER_TASK", "8"),
            )
        )
        torch.set_num_threads(max(1, _n_threads))
        torch.set_num_interop_threads(max(1, _n_threads))
        print(
            f"[shim] CPU threading: torch.set_num_threads("
            f"{torch.get_num_threads()}), interop="
            f"{torch.get_num_interop_threads()}",
        )
    except Exception as _t_err:
        print(f"[shim] could not set torch threads: {_t_err!r}")

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


def parse_frame_timestamp(path: Path) -> tuple[int, ...]:
    """Parse the frame's wall-clock timestamp from its filename for
    chronological sorting.

    Filenames look like
        2023-4-20-8-40-58-417-RGB-BP.bmp
        ^   ^ ^  ^ ^  ^  ^   ^^^ ^^^
        Y   M D  H M  S  MS

    Plain alphabetical sort breaks here because the minute / second /
    millisecond fields have variable width: '9-37-2-100' sorts AFTER
    '9-37-12-100' lexicographically, but chronologically 9:37:02
    comes before 9:37:12. We instead split on '-' and sort by the
    numeric (Y, M, D, H, M, S, MS) tuple.

    Returns a tuple of ints suitable for sorted() / min() / max().
    Falls back to a tuple of zeros if parsing fails so the caller
    can keep the file in the list rather than crashing."""
    stem = path.stem
    for suffix in ("-RGB-BP", "-RGB-bp", "-RGB", "-rgb"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.split("-")
    out: list[int] = []
    for p in parts[:7]:
        try:
            out.append(int(p))
        except ValueError:
            return (0,) * 7  # fallback: keep a stable position
    while len(out) < 7:
        out.append(0)
    return tuple(out)


def doy_from_path(path: Path) -> int | None:
    """Extract day-of-year (1-366) from a frame's filename
    timestamp. Used by the DOY-based phenology stage selector to
    pick blossom-detection thresholds appropriate for early bloom
    / bloom / petal fall / fruiting. Returns None when the
    filename can't be parsed (sorted-fallback path)."""
    ts = parse_frame_timestamp(path)
    if ts == (0,) * 7:
        return None
    y, m, d = ts[0], ts[1], ts[2]
    if y < 2000 or y > 2100 or m < 1 or m > 12 or d < 1 or d > 31:
        return None
    import datetime
    try:
        return datetime.date(y, m, d).timetuple().tm_yday
    except ValueError:
        return None


def phenol_stage_from_doy(
    doy: int, bloom_peak_doy: int = 125,
    pre_days: int = 10, post_days: int = 6,
    petal_fall_days: int = 14,
) -> str:
    """Map day-of-year to phenological stage. Window widths come
    from the sprayer pipeline's flower_detector defaults so the
    boundaries match the reference implementation:

        early_bloom : doy <= peak - pre_days
        bloom       : peak - pre_days < doy <= peak + post_days
        petal_fall  : peak + post_days < doy <= peak + post_days + petal_fall_days
        fruiting    : doy > peak + post_days + petal_fall_days
    """
    doy = int(doy)
    peak = int(bloom_peak_doy)
    pre = int(pre_days)
    post = int(post_days)
    pf = int(petal_fall_days)
    if doy <= peak - pre:
        return "early_bloom"
    if doy <= peak + post:
        return "bloom"
    if doy <= peak + post + pf:
        return "petal_fall"
    return "fruiting"


def hsv_thresholds_for_stage(stage: str) -> dict:
    """Stage-specific HSV / b_minus_r thresholds for the white +
    pink petal masks. Ported from the sprayer pipeline's
    flower_detector. Each stage has been independently tuned
    against the reference operator's data:

      early_bloom : leaf buds dominate; STRICT thresholds.
      bloom       : open petals dominate; GENEROUS thresholds.
      petal_fall  : few petals remain; VERY STRICT to avoid FPs.
      fruiting    : essentially zero petals; TIGHTEST.

    Returns a dict that callers merge into args. Use 'pink_disabled'
    True for fruiting (where any pink hue is foliage / wildflower,
    not blossoms)."""
    if stage == "early_bloom":
        return {
            "white_s_max": 30, "white_v_min": 170,
            "pink_s_min": 5, "pink_s_max": 45, "pink_v_min": 130,
            "b_minus_r_max": 5, "pink_b_minus_r_max": -10,
            "pink_disabled": False,
        }
    if stage == "bloom":
        return {
            "white_s_max": 35, "white_v_min": 160,
            "pink_s_min": 5, "pink_s_max": 80, "pink_v_min": 80,
            "b_minus_r_max": 0, "pink_b_minus_r_max": 0,
            "pink_disabled": False,
        }
    if stage == "petal_fall":
        return {
            "white_s_max": 10, "white_v_min": 205,
            "pink_s_min": 5, "pink_s_max": 22, "pink_v_min": 170,
            "b_minus_r_max": -2, "pink_b_minus_r_max": -22,
            "pink_disabled": False,
        }
    if stage == "fruiting":
        return {
            "white_s_max": 4, "white_v_min": 240,
            "pink_s_min": 999, "pink_s_max": 0, "pink_v_min": 999,
            "b_minus_r_max": 0, "pink_b_minus_r_max": -999,
            "pink_disabled": True,
        }
    return {}


def compute_texture_signals(
    rgb_arr: np.ndarray, V: np.ndarray,
    ir_arr: np.ndarray | None = None,
    win: int = 9,
    texture_threshold: float = 2.5,
    edge_threshold: float = 6.0,
) -> dict:
    """Per-pixel texture / edge signals used to discriminate real
    flower petals from smooth sky and uniform foliage.

    Real petals have local intensity variation (small bright/dark
    structure between petals) AND sharp edges (petal-petal
    boundaries, anther/petal contrast). Smooth sky and uniform
    cloud have NEITHER. The four signals computed here -- V std,
    IR std, V-channel Sobel gradient, IR-channel Sobel gradient --
    each catch a different texture mode; the union (`has_texture`)
    is the strongest single per-pixel discriminator we have when
    HSV alone fails.

    Ported from the reference lai_estimation_system87 detector.
    Returns a dict so the caller can pass individual signals to
    other gates (e.g. confirmed_real which needs has_texture +
    valid_depth)."""
    try:
        import cv2 as _cv2_t
        from scipy.ndimage import uniform_filter
    except Exception:
        # Fall back to a permissive signal so callers don't break
        # if scipy is unavailable.
        return {
            "V_std": None, "IR_std": None,
            "edge_mag": None, "ir_edge_mag": None,
            "has_texture": np.ones(V.shape[:2], dtype=bool),
        }
    V_f = V.astype(np.float32)
    V_mean = uniform_filter(V_f, size=win)
    V_var = uniform_filter(V_f ** 2, size=win) - V_mean ** 2
    V_std = np.sqrt(np.maximum(V_var, 0))

    gray = _cv2_t.cvtColor(rgb_arr, _cv2_t.COLOR_RGB2GRAY)
    sobx = _cv2_t.Sobel(gray, _cv2_t.CV_32F, 1, 0, ksize=3)
    soby = _cv2_t.Sobel(gray, _cv2_t.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobx ** 2 + soby ** 2)

    IR_std = None
    ir_edge_mag = None
    if ir_arr is not None:
        IR_f = ir_arr.astype(np.float32)
        # Bring to 0-255 scale if it's normalized to [0, 1].
        if IR_f.max() <= 1.5:
            IR_f = IR_f * 255.0
        IR_mean = uniform_filter(IR_f, size=win)
        IR_var = uniform_filter(IR_f ** 2, size=win) - IR_mean ** 2
        IR_std = np.sqrt(np.maximum(IR_var, 0))
        ir_sobx = _cv2_t.Sobel(IR_f, _cv2_t.CV_32F, 1, 0, ksize=3)
        ir_soby = _cv2_t.Sobel(IR_f, _cv2_t.CV_32F, 0, 1, ksize=3)
        ir_edge_mag = np.sqrt(ir_sobx ** 2 + ir_soby ** 2)

    has_texture = (V_std > texture_threshold) | (edge_mag > edge_threshold)
    if IR_std is not None:
        has_texture = (
            has_texture
            | (IR_std > texture_threshold)
            | (ir_edge_mag > edge_threshold)
        )
    return {
        "V_std": V_std, "IR_std": IR_std,
        "edge_mag": edge_mag, "ir_edge_mag": ir_edge_mag,
        "has_texture": has_texture,
    }


def compute_sky_exclusions(
    H: np.ndarray, S: np.ndarray, V: np.ndarray,
    b_minus_r: np.ndarray, ir_8bit: np.ndarray | None,
    no_depth: np.ndarray, has_texture: np.ndarray,
    near_tree: np.ndarray | None,
    *,
    enable_smooth: bool = True,
    enable_warm: bool = True,
    enable_upper: bool = True,
    enable_overcast: bool = True,
    enable_grey: bool = True,
    enable_br: bool = True,
    ir_sky_ceil: int = 60,
    upper_frac: float = 0.30,
) -> np.ndarray:
    """Combine the multiple sky-failure-mode exclusions from the
    reference detector. Each sub-rule catches a different visual
    failure: smooth bright sky, warm golden-hour sky, top-strip
    cropped sky, overcast cloud, bright cloud edge, blue-tinted sky.
    The union goes into the not_flower exclusion combined with
    foliage / leaf-bud / etc. Disable individual sub-rules via
    flags if a specific mode mis-fires on your data."""
    h_img, w_img = H.shape[:2]
    sky = np.zeros_like(H, dtype=bool)
    # Pixels that are clearly NEAR a tree (within the dilated
    # near_tree band) get a free pass on every sky exclusion.
    # Real upper-canopy flowers are silhouetted against sky
    # behind them, so naively, sky_warm / sky_smooth / sky_grey
    # fire on the petal pixels themselves and refinement loses
    # them. near_tree captures "this pixel is on or adjacent to
    # branches with valid depth"; if so, it's almost certainly
    # canopy not sky.
    near_tree_safe = near_tree if near_tree is not None else None
    # Smooth bright sky: no texture, no depth, low IR.
    if enable_smooth and ir_8bit is not None:
        sky_smooth = (~has_texture) & no_depth & (ir_8bit < ir_sky_ceil)
        if near_tree_safe is not None:
            sky_smooth = sky_smooth & (~near_tree_safe)
        sky |= sky_smooth
    # Warm / golden-hour sky: no depth, V high, IR low.
    if enable_warm and ir_8bit is not None:
        sky_warm = no_depth & (V > 80) & (ir_8bit < 70)
        if near_tree_safe is not None:
            sky_warm = sky_warm & (~near_tree_safe)
        sky |= sky_warm
    # Top-strip sky: top fraction of image, no depth, not near tree.
    if enable_upper:
        upper = np.zeros_like(no_depth)
        upper[: int(h_img * upper_frac), :] = True
        if near_tree_safe is not None:
            sky_upper = upper & no_depth & (~near_tree_safe) & (S < 50) & (V > 80)
        else:
            sky_upper = upper & no_depth & (S < 50) & (V > 80)
        sky |= sky_upper
    # Overcast cloud: no depth, very low S, V in cloud range.
    if enable_overcast:
        sky_overcast = no_depth & (S < 12) & (V > 140) & (V <= 200)
        if near_tree_safe is not None:
            sky_overcast = sky_overcast & (~near_tree_safe)
        sky |= sky_overcast
    # Bright grey cloud: depth=0, V very high, S low, IR moderate.
    if enable_grey and ir_8bit is not None:
        sky_grey = no_depth & (V > 200) & (S < 20) & (ir_8bit < 100)
        if near_tree_safe is not None:
            sky_grey = sky_grey & (~near_tree_safe)
        sky |= sky_grey
    # B-R-positive sky catch-all: low S, high V, no depth, blue-shifted.
    if enable_br:
        sky_br = no_depth & (b_minus_r > 0) & (S < 30) & (V > 150)
        if near_tree_safe is not None:
            sky_br = sky_br & (~near_tree_safe)
        sky |= sky_br
    return sky


def compute_negative_pixel_masks(
    H: np.ndarray, S: np.ndarray, V: np.ndarray,
    *,
    enable_bark: bool = True,
    enable_dark: bool = True,
    enable_ground: bool = True,
) -> np.ndarray:
    """Per-pixel negative masks: things a flower can never be.
    Combines bark, dark material, and ground-grass exclusions
    from the reference detector. Returned as a single boolean
    mask suitable for OR-ing with other exclusions."""
    out = np.zeros_like(H, dtype=bool)
    if enable_bark:
        bark_brown = (
            (H >= 8) & (H <= 35) & (S > 50) & (V < 100)
        )
        out |= bark_brown
    if enable_dark:
        dark_material = V < 45
        out |= dark_material
    if enable_ground:
        ground_grass = (
            (H >= 25) & (H <= 95) & (V < 100) & (S > 20)
        )
        out |= ground_grass
    return out


def compute_confirmed_real(
    valid_depth: np.ndarray,
    has_texture: np.ndarray,
    near_tree_radius_px: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Spatial confirmation gate: a pixel is "real on the tree"
    iff it has valid depth OR (it's near a valid-depth pixel AND
    it has local texture).

    The dilated near_tree band catches IR-overexposed flower
    pixels (depth=0 because the petal saturated the IR projector)
    that are textured AND adjacent to branches with valid depth.
    Without this, those flowers get rejected by depth filters.

    Returns (confirmed_real, near_tree).
    """
    try:
        import cv2 as _cv2_c
    except Exception:
        return valid_depth.copy(), valid_depth.copy()
    k = max(1, 2 * int(near_tree_radius_px) + 1)
    kern = _cv2_c.getStructuringElement(_cv2_c.MORPH_ELLIPSE, (k, k))
    near_tree = _cv2_c.dilate(
        valid_depth.astype(np.uint8), kern, iterations=1,
    ).astype(bool)
    confirmed_real = valid_depth | (near_tree & has_texture)
    return confirmed_real, near_tree


def compute_density_score(
    masks_np: np.ndarray, frame_hw: tuple[int, int],
    sigma: float = 4.0, kernel_size: int = 15,
) -> float:
    """Estrada-style density score: union all kept flower masks
    into a binary image, Gaussian-blur it, sum. Provides an
    alternative counting metric robust to over- / under-
    segmentation. Returns 0.0 if no masks."""
    try:
        import cv2 as _cv2_d
    except Exception:
        return 0.0
    if masks_np is None or len(masks_np) == 0:
        return 0.0
    union = np.zeros(frame_hw, dtype=np.uint8)
    for m in masks_np:
        mb = m.astype(bool)
        if mb.ndim == 3:
            mb = mb.any(axis=0)
        union |= mb.astype(np.uint8)
    union *= 255
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    blurred = _cv2_d.GaussianBlur(union.astype(np.float32), (k, k), sigmaX=sigma)
    return float(blurred.sum() / 255.0)


def fill_anther_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Flood-fill interior holes in a binary uint8 mask.

    Real apple blossoms have yellow anthers / pollen in the center
    that fail the white S < N rule, leaving 5-10 px donut or
    crescent gaps in the refined mask. The distance transform of
    a donut produces 2-3 opposite-side ridge peaks that get
    counted as separate flowers; for YOLO bbox the donut shape
    also produces inflated bbox counts and lower mask density.

    Implementation: pad the mask with a 1-pixel 0-border so the
    frame edge is never treated as interior, then flood-fill the
    background from (0, 0). After the flood, exterior background
    is 255 and interior holes are still 0; bitwise NOT gives a
    holes-only mask that we OR back into the original. Adjacent
    blossoms stay separate because the background between them
    connects to the frame exterior (an exterior region), not
    interior holes.

    Falls back to the input unchanged if cv2 isn't available or
    the mask is empty."""
    try:
        import cv2 as _cv2_fh
    except Exception:
        return mask_u8
    if mask_u8.size == 0 or not mask_u8.any():
        return mask_u8
    h, w = mask_u8.shape[:2]
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask_u8
    flood_aux = np.zeros((h + 4, w + 4), dtype=np.uint8)
    _cv2_fh.floodFill(padded, flood_aux, (0, 0), 255)
    interior_holes = _cv2_fh.bitwise_not(padded)[1:-1, 1:-1]
    return _cv2_fh.bitwise_or(mask_u8, interior_holes)


def info_path_for(img_path: Path) -> Path:
    """Given .../<session>/RGB/<stem>-RGB-BP.<ext>, return
    .../<session>/Info/<bare_timestamp>.txt.

    The Info file is just named with the bare timestamp (no
    `-Info` or `-RGB-BP` suffix), e.g. for RGB
        2023-4-20-8-40-58-417-RGB-BP.bmp
    Info is
        2023-4-20-8-40-58-417.txt
    """
    session_dir = img_path.parent.parent
    base = img_path.stem
    for suffix in ("-RGB-BP", "-RGB-bp", "-RGB", "-rgb"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return session_dir / "Info" / f"{base}.txt"


def find_images(root: Path, only_rgb_folders: bool = True,
                sample_per_session: int | None = None,
                frame_range: tuple[int, int] | None = None,
                require_all_modalities: bool = False,
                require_info_modality: bool = False,
                sample_mode: str = "sequential",
                sample_stride: int = 20):
    """Yield (day, category, session, image_path) tuples.

    When require_all_modalities is True, the per-session image list
    is FIRST filtered to frames that have matching depth / IR / PRGB
    (and optionally Info) sibling files, and THEN frame_range /
    sample_per_session is applied. This way --frame-range 50 80
    means "the 50th-80th *complete* frame".

    Sorting: frames are sorted by the parsed (Y, M, D, H, M, S, MS)
    timestamp tuple from the filename, NOT alphabetically. Variable-
    width seconds/milliseconds fields make alphabetical sort
    chronologically wrong (e.g. '9-37-2-100' sorts after '9-37-12-100'
    lexicographically).

    sample_mode controls how `sample_per_session` picks frames:
      "sequential" : the FIRST N complete frames of the session
                     (chronological order). User-requested default.
      "even"       : N frames evenly spaced across the session
                     (the original behavior, kept for analyses
                     that want temporal coverage).
    frame_range overrides sample_per_session when supplied.
    """
    root = Path(root)
    # Support --root pointing at either the dataset parent (containing
    # multiple "2023 day *" folders) OR at a single day folder
    # directly. The latter is useful for test runs scoped to one day.
    if root.is_dir() and root.name.lower().startswith("2023 day"):
        day_dirs = [root]
    else:
        day_dirs = sorted(root.glob("2023 day *"))
    if not day_dirs:
        print(
            f"[scan] WARNING: no '2023 day *' folders found under {root}",
            file=sys.stderr,
        )
    for day_dir in day_dirs:
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
                    imgs = [p for p in sd.iterdir() if p.suffix.lower() in IMAGE_EXTS]
                    if not imgs:
                        continue
                    # Chronological sort by parsed timestamp.
                    imgs.sort(key=parse_frame_timestamp)
                    if require_all_modalities:
                        n_total = len(imgs)
                        miss_d = miss_i = miss_p = miss_info = 0
                        kept: list[Path] = []
                        for p in imgs:
                            ok_d = depth_path_for(p).is_file()
                            ok_i = ir_path_for(p).is_file()
                            ok_p = prgb_path_for(p).is_file()
                            ok_info = (
                                info_path_for(p).is_file()
                                if require_info_modality else True
                            )
                            if ok_d and ok_i and ok_p and ok_info:
                                kept.append(p)
                            else:
                                if not ok_d:
                                    miss_d += 1
                                if not ok_i:
                                    miss_i += 1
                                if not ok_p:
                                    miss_p += 1
                                if require_info_modality and not ok_info:
                                    miss_info += 1
                        imgs = kept
                        if len(imgs) < n_total:
                            extra = (
                                f" Info={miss_info}"
                                if require_info_modality else ""
                            )
                            print(f"[scan] {session_dir.name}: "
                                  f"{len(imgs)}/{n_total} complete frames "
                                  f"(missing depth={miss_d} IR={miss_i} "
                                  f"PRGB={miss_p}{extra})",
                                  file=sys.stderr)
                        if not imgs:
                            continue
                    if frame_range is not None:
                        a, b = frame_range
                        a = max(0, a)
                        b = min(len(imgs), b)
                        idxs = list(range(a, b))
                    elif sample_mode == "stride":
                        # Take every Nth complete frame in
                        # chronological order. Per-session count
                        # varies by session length; spacing is
                        # uniform across the orchard pass so we
                        # see every tree exactly once if the rig
                        # moved at constant speed. Independent of
                        # --sample-per-session (the count is
                        # determined by session length / stride).
                        stride = max(1, int(sample_stride))
                        idxs = list(range(0, len(imgs), stride))
                    elif sample_per_session is None or sample_per_session <= 0:
                        idxs = list(range(len(imgs)))
                    elif sample_mode == "sequential":
                        # First N chronologically-ordered complete
                        # frames. Matches the user's expectation of
                        # "100 sequential frames per session".
                        idxs = list(range(min(len(imgs), sample_per_session)))
                    else:
                        # Legacy evenly-spaced sampling for analyses
                        # that need representative coverage of the
                        # whole session timeline.
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
    .../<session>/depth/<stem>-Depth.<txt|bmp>.

    The user's VB capture code saves BOTH a .txt (raw mm) and a
    .bmp. Some sessions have the .txt saved at a smaller width
    than the RGB/BMP (e.g., 480x74 vs the full 480x640) -- when
    that happens load_depth_mm falls back to the greyscale .bmp.

    Path priority:
      1. depth/<base>-Depth.txt              (if present & full-res)
      2. depth/Image/<base>-Depth.bmp        (the user's capture
                                              writes BMPs to a
                                              `depth/Image/`
                                              subfolder)
      3. depth/<base>-Depth.bmp              (sibling of the .txt,
                                              fallback)
    """
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
    # Try the Image subfolder where the capture code writes BMPs.
    bmp_in_subdir = (
        session_dir / "depth" / "Image" / f"{base}-Depth.bmp"
    )
    if bmp_in_subdir.is_file():
        return bmp_in_subdir
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


def _decode_greyscale_depth_bmp(
    bmp_path: Path, target_hw: tuple[int, int],
    *, mm_min: float = 600.0, mm_max: float = 5000.0,
) -> np.ndarray | None:
    """Decode a greyscale-stored depth-visualization BMP to uint16 mm.

    Many capture pipelines write depth as an 8-bit GREYSCALE BMP
    (a 3-channel uint8 BMP where all three channels are equal --
    this is what cv2 returns for a single-channel BMP saved
    without a palette). The brightness encodes depth: 0 = invalid
    (no return), 255 = saturated / past max range, 1-254 linearly
    span [mm_min, mm_max].

    Returns None if the BMP is a true 3-channel COLOR BMP (e.g.,
    a JET/TURBO colormap) -- those need a separate color decoder.
    """
    try:
        import cv2 as _cv2
    except Exception:
        return None
    b = _cv2.imread(str(bmp_path), _cv2.IMREAD_UNCHANGED)
    if b is None:
        return None
    if b.ndim == 3:
        # Check if it's actually greyscale (all channels equal).
        if not ((b[..., 0] == b[..., 1]).all()
                and (b[..., 1] == b[..., 2]).all()):
            # True colour -- can't decode as greyscale depth.
            return None
        g = b[..., 0]
    else:
        g = b
    if g.dtype != np.uint8:
        return None
    # Map greyscale -> mm. 0 = invalid (kept as 0); 255 = past
    # sensor max (kept as 0 -- treated as invalid by downstream
    # depth gates that check (depth > 0) & (depth < 60000)).
    valid = (g > 0) & (g < 255)
    d = np.zeros(g.shape, dtype=np.float32)
    if valid.any():
        d[valid] = (
            mm_min + (g[valid].astype(np.float32) - 1.0)
            / 253.0 * (mm_max - mm_min)
        )
    H, W = target_hw
    if d.shape != (H, W):
        pim = Image.fromarray(d, mode="F").resize(
            (W, H), Image.BILINEAR,
        )
        d = np.asarray(pim)
    return np.clip(d, 0, 65535).astype(np.uint16)


def load_depth_mm(depth_path: Path, target_hw: tuple[int, int]) -> np.ndarray | None:
    """Load a depth file and return uint16 mm upsampled to target (H, W).

    Source-priority chain:
      1. .txt  (ASCII raw mm) -- used UNLESS it's truncated to a
                                 width drastically smaller than
                                 the target frame width (some
                                 capture pipelines save .txt at
                                 lower resolution than the BMP /
                                 RGB; bilinear upsampling 8x
                                 destroys depth boundaries).
      2. .bmp/.png greyscale (depth viz, 0=invalid, 1-254 linear mm)
      3. .bmp/.png 16-bit single-channel raw mm

    Falls through to the .bmp (greyscale) when:
      - .txt is missing
      - .txt's column count < 50% of target width (truncated)
    """
    H, W = target_hw
    if not depth_path.is_file():
        return None
    suffix = depth_path.suffix.lower()
    txt_was_truncated = False
    txt_d: np.ndarray | None = None
    if suffix == ".txt":
        try:
            txt_d = np.loadtxt(depth_path, dtype=np.float32)
        except Exception:
            txt_d = None
        if txt_d is not None and txt_d.ndim == 2:
            if txt_d.shape[1] < W * 0.5:
                # The .txt was saved truncated. Fall back to the
                # matching BMP if present (it's full-resolution).
                txt_was_truncated = True
                key = str(depth_path.parent)
                if key not in _warned_bad_depth_bmps:
                    _warned_bad_depth_bmps.add(key)
                    print(
                        f"[warn] depth .txt at {key} is "
                        f"{txt_d.shape} (truncated horizontally; "
                        f"target W={W}). Falling back to greyscale "
                        f"BMP for full-resolution depth.",
                        file=sys.stderr,
                    )
        else:
            txt_d = None

    if txt_was_truncated or suffix in (".bmp", ".png"):
        # Look for the matching BMP. depth_path_for returns the
        # .txt path; the BMP can live alongside it OR in an
        # 'Image' subfolder (the user's capture pipeline writes
        # to depth/Image/).
        if suffix == ".txt":
            stem = depth_path.stem  # e.g. "...-Depth"
            cand_paths = [
                depth_path.with_suffix(".bmp"),
                depth_path.parent / "Image" / f"{stem}.bmp",
                depth_path.parent / "Image" / f"{stem}.BMP",
            ]
        else:
            cand_paths = [depth_path]
        for cp in cand_paths:
            if not cp.is_file():
                continue
            d = _decode_greyscale_depth_bmp(cp, target_hw)
            if d is not None:
                return d
            # Greyscale decode failed -- maybe it's a 16-bit raw
            # BMP. Try the legacy path.
            try:
                import cv2
                d_raw = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED)
                if d_raw is None or d_raw.ndim == 3 or d_raw.dtype == np.uint8:
                    continue
                d_raw = d_raw.astype(np.int32)
                if d_raw.shape != (H, W):
                    pim = Image.fromarray(
                        d_raw.astype(np.float32), mode="F",
                    ).resize((W, H), Image.BILINEAR)
                    d_raw = np.asarray(pim)
                return np.clip(d_raw, 0, 65535).astype(np.uint16)
            except Exception:
                continue
        if txt_was_truncated and txt_d is not None:
            # No BMP available -- have to upsample the truncated
            # .txt. Better than nothing.
            pass

    # .txt path (full-res or truncated-with-no-BMP fallback)
    if txt_d is None and suffix == ".txt":
        try:
            txt_d = np.loadtxt(depth_path, dtype=np.float32)
        except Exception:
            return None
    if txt_d is None or txt_d.ndim != 2 or txt_d.size == 0:
        return None
    if txt_d.shape != (H, W):
        pim = Image.fromarray(
            txt_d.astype(np.float32), mode="F",
        ).resize((W, H), Image.BILINEAR)
        txt_d = np.asarray(pim)
    return np.clip(txt_d, 0, 65535).astype(np.uint16)


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
                        use_otsu: bool = True,
                        seed_dilate_px: int = 0) -> list[np.ndarray]:
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

    # Optional seed dilation — gives each marker a multi-pixel buffer
    # before watershed so the resulting boundaries are placed cleanly
    # halfway between adjacent peaks instead of clinging to single-
    # pixel seeds. Mirrors the reference flower_detector's
    # cv2.dilate(seed, MORPH_ELLIPSE 5x5) before cv2.watershed. Each
    # label is dilated separately to preserve label IDs.
    if seed_dilate_px > 0:
        seed_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(seed_dilate_px) + 1, 2 * int(seed_dilate_px) + 1),
        )
        new_markers = np.zeros_like(markers)
        # Dilate in label-id order so larger labels don't overwrite
        # smaller ones at overlap points (they shouldn't overlap at
        # this stage, but defensive).
        for lid in range(1, n_labels):
            single = (markers == lid).astype(np.uint8)
            dilated = cv2.dilate(single, seed_kern, iterations=1)
            # Restrict dilation to the cluster mask so we don't seed
            # outside the blossom boundary.
            new_markers[(dilated > 0) & sub & (new_markers == 0)] = lid
        markers = new_markers

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
                         edge_margin_sides: int = -1,
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
                         max_mask_green_frac: float = 0.0,
                         green_h_lo: int = 35, green_h_hi: int = 85,
                         green_s_min: int = 40, green_v_min: int = 35,
                         green_blossom_override_frac: float = 0.20,
                         min_peaks_per_1000px: float = 0.0,
                         peak_min_distance_px: int = 5,
                         peak_threshold_abs: int = 80,
                         peak_min_area_px: int = 60,
                         peak_min_distance_px2: int = 0,
                         peak_prominence_min: float = 5.0,
                         min_anther_holes_per_1000px: float = 0.0,
                         anther_petal_v_min: int = 100,
                         anther_hole_min_area_px: int = 2,
                         anther_hole_max_area_px: int = 60,
                         anther_min_area_px: int = 100,
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
            "max_bbox_area": 0, "low_density": 0,
            "leaf_green": 0, "low_peak_density": 0,
            "low_anther_density": 0}
    areas: list[int] = []

    # Pre-compute the "could be apple blossom" pixel mask once per frame
    # (white OR pink in HSV — ports the bloom-stage gates from
    # flower_detector.py as a POSITIVE color check).
    blossom_pixel_mask = None
    green_pixel_mask = None
    if (require_blossom_color or max_mask_green_frac > 0) and rgb_arr is not None:
        import cv2
        hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
        H_ch, S_ch, V_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        # Compute blossom_pixel_mask whenever we need to gate on
        # blossom presence -- either for the require_blossom_color
        # check or as the override for the leaf-green check.
        if require_blossom_color or max_mask_green_frac > 0:
            white = (S_ch <= blossom_white_s_max) & (V_ch >= blossom_white_v_min)
            in_pink_hue = (((H_ch >= blossom_pink_h_lo) & (H_ch <= blossom_pink_h_hi))
                           | ((H_ch >= blossom_pink_h_lo2) & (H_ch <= blossom_pink_h_hi2)))
            pink = (in_pink_hue
                    & (S_ch >= blossom_pink_s_lo) & (S_ch <= blossom_pink_s_hi)
                    & (V_ch >= blossom_pink_v_min))
            blossom_pixel_mask = white | pink
        if max_mask_green_frac > 0:
            # Leaf green: tree foliage. We tag every "obviously green"
            # pixel and later reject any flower mask whose green-pixel
            # fraction exceeds max_mask_green_frac. Catches the case
            # where a leaf with sky/cloud behind it gets segmented as
            # a "flower" because the bright sky pixels register as
            # white inside the SAM mask -- the leaf's green pixels
            # still dominate, so the green-frac gate fires.
            green_pixel_mask = (
                (H_ch >= green_h_lo) & (H_ch <= green_h_hi)
                & (S_ch >= green_s_min) & (V_ch >= green_v_min)
            )

    # Bright local-peak map for the flower-vs-leaf texture
    # discriminator. Each apple blossom is a bright petal with a
    # darker anther center, so a flower cluster has multiple V-
    # channel local maxima at small scale; a leaf is mostly
    # uniform with at most a few specular highlights.
    #
    # Implementation has THREE refinements over a naive
    # `V == max_in_window` check:
    #   1. Prominence filter: V at the peak must be at least
    #      peak_prominence_min above the local mean. Suppresses
    #      uniform-bright plateaus (sky, smooth leaves) where
    #      every pixel ties for the max.
    #   2. Plateau collapse: each connected component of
    #      candidate-peak pixels (a uniform-bright disc, e.g., a
    #      single petal) becomes ONE peak (its anchor pixel),
    #      not N peaks (one per pixel of the disc).
    #   3. Multi-scale: an optional second scale (wider Gaussian
    #      + wider max window) catches the cluster-centroid peak
    #      of multi-blossom clusters whose individual blossoms
    #      are smoothed away at the narrow scale. Each scale gets
    #      its own plateau-collapse, then the two centroid masks
    #      are OR'd. Real flowers register at one OR both scales;
    #      leaves register at neither.
    peak_pixel_mask = None
    if min_peaks_per_1000px > 0 and rgb_arr is not None:
        try:
            import cv2
            from scipy.ndimage import (
                maximum_filter, uniform_filter, label,
            )

            def _peaks_at_scale(V_blur_, mf_size_):
                V_max_ = maximum_filter(
                    V_blur_, size=mf_size_, mode="nearest",
                )
                V_mean_ = uniform_filter(
                    V_blur_, size=mf_size_, mode="nearest",
                )
                cand = (
                    (V_blur_ == V_max_)
                    & (V_blur_ >= float(peak_threshold_abs))
                    & ((V_blur_ - V_mean_)
                       >= float(peak_prominence_min))
                )
                if not cand.any():
                    return np.zeros_like(cand)
                p_lbl, n_pl = label(cand)
                if n_pl == 0:
                    return np.zeros_like(cand)
                # Collapse each plateau to its anchor (first
                # row-major-order pixel) so a uniform-bright
                # blossom contributes ONE peak, not N.
                flat = p_lbl.ravel()
                valid = flat > 0
                if not valid.any():
                    return np.zeros_like(cand)
                vlbl = flat[valid]
                vidx = np.where(valid)[0]
                _, first_idx = np.unique(vlbl, return_index=True)
                anchors = vidx[first_idx]
                out = np.zeros(p_lbl.size, dtype=bool)
                out[anchors] = True
                return out.reshape(p_lbl.shape)

            hsv_p = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
            V_p = hsv_p[..., 2].astype(np.float32)
            # Scale 1 (narrow, blossom / anther-petal scale).
            V_blur = cv2.GaussianBlur(V_p, (3, 3), 0.8)
            mf_size = 2 * max(1, int(peak_min_distance_px)) + 1
            peak_pixel_mask = _peaks_at_scale(V_blur, mf_size)
            # Scale 2 (wider, cluster-centroid scale).
            if peak_min_distance_px2 > 0:
                sigma2 = max(1.5, peak_min_distance_px2 / 4.0)
                k2 = max(5, 2 * (peak_min_distance_px2 // 3) + 1)
                if k2 % 2 == 0:
                    k2 += 1
                V_blur2 = cv2.GaussianBlur(V_p, (k2, k2), sigma2)
                mf_size2 = 2 * peak_min_distance_px2 + 1
                peaks2 = _peaks_at_scale(V_blur2, mf_size2)
                peak_pixel_mask = peak_pixel_mask | peaks2
        except Exception:
            peak_pixel_mask = None

    # Anther-hole pixel map for the flower-vs-leaf POSITIVE
    # discriminator. Each apple blossom has a yellow / dark
    # anther cluster in the center: SMALL DARK pixels surrounded
    # by BRIGHT petal pixels. A leaf-with-sky-behind has dark
    # leaf pixels, but they are NOT surrounded by bright pixels
    # (they are adjacent to bright sky on one side). So a global
    # flood-fill of the bright-pixel mask from a frame corner
    # marks every reachable dark pixel as 'exterior'. The dark
    # pixels that remain unreached are 'interior dark' --
    # surrounded by bright on all sides -- which is exactly the
    # anther-hole signature. We label them once per frame and
    # count per mask in the loop.
    anther_label_img = None
    anther_cc_areas: np.ndarray | None = None
    if min_anther_holes_per_1000px > 0 and rgb_arr is not None:
        try:
            import cv2
            hsv_a = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
            V_a = hsv_a[..., 2]
            bright_u8 = (
                (V_a >= int(anther_petal_v_min)).astype(np.uint8) * 255
            )
            if bright_u8.any():
                h_a, w_a = bright_u8.shape
                padded = np.zeros((h_a + 2, w_a + 2), dtype=np.uint8)
                padded[1:-1, 1:-1] = bright_u8
                flood_aux = np.zeros(
                    (h_a + 4, w_a + 4), dtype=np.uint8,
                )
                cv2.floodFill(padded, flood_aux, (0, 0), 255)
                interior = cv2.bitwise_not(padded)[1:-1, 1:-1]
                if interior.any():
                    n_cc_a, labels_a, stats_a, _ = (
                        cv2.connectedComponentsWithStats(
                            interior, connectivity=8,
                        )
                    )
                    if n_cc_a > 1:
                        anther_label_img = labels_a
                        anther_cc_areas = stats_a[
                            :, cv2.CC_STAT_AREA,
                        ].astype(np.int64)
        except Exception:
            anther_label_img = None
            anther_cc_areas = None

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
        # Edge-margin filter. Top/bottom uses --flower-edge-margin-px
        # (D455 stereo artefacts); left/right uses
        # --flower-edge-margin-sides-px when >= 0, else the same.
        # Setting sides to 0 lets blossom bboxes that crop at the
        # L/R edge of the frame survive (half-trees at frame edges).
        em_tb = edge_margin
        em_lr = edge_margin_sides if edge_margin_sides >= 0 else edge_margin
        if em_tb > 0 or em_lr > 0:
            if (bx < em_lr or by < em_tb
                    or (bx + bw) > (img_w - 1 - em_lr)
                    or (by + bh) > (img_h - 1 - em_tb)):
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
        # Leaf-with-sky-behind rejection. A leaf that has bright sky
        # behind it can pass the blossom-color frac (the sky pixels
        # inside the mask register as white). But the mask still has
        # a substantial fraction of green leaf pixels. Reject if the
        # green-pixel fraction exceeds the cap.
        #
        # IMPORTANT: a real flower cluster ON a branch naturally
        # picks up surrounding leaves in the SAM mask (typical
        # green frac 30-50%). We don't want to reject those. Two
        # safeguards:
        #   1. The cap (max_mask_green_frac) is itself fairly high
        #      (~0.50 in run scripts) so leaf-dominated masks have
        #      to be clearly leaf-y.
        #   2. If the mask's blossom-color fraction is at or above
        #      green_blossom_override_frac, it's a real cluster;
        #      skip the green check entirely. Leaf-with-sky has
        #      substantially less blossom-color coverage than a
        #      real cluster, so this override fires only for the
        #      clusters we want to keep.
        if max_mask_green_frac > 0 and green_pixel_mask is not None:
            mb = np.asarray(m).astype(bool)
            if mb.ndim == 3:
                mb = mb.any(axis=0)
            total = int(mb.sum())
            if total > 0:
                frac_green = (
                    float((mb & green_pixel_mask).sum()) / float(total)
                )
                # Compute blossom frac for the override even if the
                # blossom-color gate above didn't fire (e.g.,
                # require_blossom_color is on but the mask passed
                # min_blossom_color_frac).
                if blossom_pixel_mask is not None:
                    frac_blossom = (
                        float((mb & blossom_pixel_mask).sum())
                        / float(total)
                    )
                else:
                    frac_blossom = 0.0
                if (frac_green > max_mask_green_frac
                        and frac_blossom < green_blossom_override_frac):
                    keep[i] = False
                    diag["leaf_green"] += 1
                    continue
        # Bright-peak density gate. Apple blossoms have local V
        # maxima at petal scale; leaves don't (uniform brightness
        # with at most a few specular highlights). Reject masks
        # with too few peaks per 1000 px of mask area. Skipped
        # below peak_min_area_px (small masks naturally have 0-1
        # peaks just due to size and shouldn't be judged).
        if min_peaks_per_1000px > 0 and peak_pixel_mask is not None:
            mb = np.asarray(m).astype(bool)
            if mb.ndim == 3:
                mb = mb.any(axis=0)
            mb_area = int(mb.sum())
            if mb_area >= peak_min_area_px:
                n_peaks = int((mb & peak_pixel_mask).sum())
                density = (n_peaks * 1000.0) / float(mb_area)
                if density < min_peaks_per_1000px:
                    keep[i] = False
                    diag["low_peak_density"] += 1
                    continue
        # Anther-hole density gate (POSITIVE flower signal).
        # An anther hole is a small dark spot surrounded entirely
        # by bright petal pixels. Leaves and leaf-with-sky masks
        # don't have this pattern: their dark pixels are adjacent
        # to other dark pixels or to bright sky, never fully
        # enclosed by bright petals. We count anther-hole CCs
        # whose pixels fall inside the mask, filtering by size
        # so noise (1-pixel specks) and large branch gaps are
        # excluded. Skipped below anther_min_area_px.
        if (min_anther_holes_per_1000px > 0
                and anther_label_img is not None
                and anther_cc_areas is not None):
            mb = np.asarray(m).astype(bool)
            if mb.ndim == 3:
                mb = mb.any(axis=0)
            mb_area = int(mb.sum())
            if mb_area >= anther_min_area_px:
                # Find which anther-hole CCs intersect this mask.
                # Take the labels image at mask pixels and unique
                # them. Each unique non-zero label is an anther
                # CC at least partially inside the mask. Filter
                # by precomputed CC area for size validity.
                lbls_in = anther_label_img[mb]
                if lbls_in.size > 0:
                    uniq = np.unique(lbls_in)
                    uniq = uniq[uniq > 0]
                    n_anther = 0
                    for cc_id in uniq:
                        a = int(anther_cc_areas[int(cc_id)])
                        if (anther_hole_min_area_px <= a
                                <= anther_hole_max_area_px):
                            n_anther += 1
                    density = (n_anther * 1000.0) / float(mb_area)
                    if density < min_anther_holes_per_1000px:
                        keep[i] = False
                        diag["low_anther_density"] += 1
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


def filter_far_trunks(
    trunk_boxes: list,
    trunk_scores: list,
    depth_mm: np.ndarray | None,
    trunk_masks: np.ndarray | None = None,
    *,
    max_depth_mm: float = 3000.0,
    min_valid_pixels: int = 5,
) -> tuple[list, list]:
    """Reject trunks whose median valid depth is beyond max_depth_mm.
    Targets the failure mode where SAM 3 detects trunks on background
    trees (5-10 m back-row) and they become Voronoi anchors,
    capturing background-canopy regions that then leak through the
    canopy-overlap gate as 'real' flower detections on the wrong tree.

    Asymmetric on purpose: only rejects when we have CONFIRMED far
    depth. Trunks with insufficient valid-depth pixels (<5) get the
    benefit of the doubt -- a thin partial-frame trunk may have
    sparse depth on its actual mask but be on the foreground tree.

    Returns (kept_boxes, kept_scores) -- same shape as input.
    """
    if not trunk_boxes or depth_mm is None or max_depth_mm <= 0:
        return trunk_boxes, trunk_scores
    H_img, W_img = depth_mm.shape[:2]
    use_masks = (
        trunk_masks is not None
        and len(trunk_masks) == len(trunk_boxes)
    )
    kept_boxes: list = []
    kept_scores: list = []
    for i, (box, score) in enumerate(zip(trunk_boxes, trunk_scores)):
        if use_masks:
            m = trunk_masks[i]
            if hasattr(m, "ndim") and m.ndim == 3:
                m = m.any(axis=0)
            mb = np.asarray(m).astype(bool)
            if mb.shape != (H_img, W_img):
                mb = None
        else:
            mb = None
        if mb is not None:
            d_in = depth_mm[mb]
        else:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W_img, x2); y2 = min(H_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            d_in = depth_mm[y1:y2, x1:x2].ravel()
        valid = d_in[(d_in > 0) & (d_in < 60000)]
        if valid.size < min_valid_pixels:
            kept_boxes.append(box)
            kept_scores.append(score)
            continue
        med = float(np.median(valid))
        if med > max_depth_mm:
            continue
        kept_boxes.append(box)
        kept_scores.append(score)
    return kept_boxes, kept_scores


def filter_painted_stake_trunks(
    trunk_boxes: list,
    trunk_scores: list,
    rgb_arr: np.ndarray,
    trunk_masks: np.ndarray | None = None,
    *,
    max_green_dominant_pct: float = 0.35,
    green_dominance_threshold: int = 15,
    min_brown_pct: float = 0.20,
) -> tuple[list, list]:
    """Reject trunk detections that are predominantly green-
    painted (apple-orchard support stakes) rather than wooden
    trunks.

    Heuristic: real apple trunks read brown/grey/red-shifted in
    RGB (R near or above G, slight blue deficit). Painted green
    stakes have G consistently dominant by 15+ over R and B.

    Color statistics are computed on the SAM 3 INSTANCE MASK
    when one is provided, NOT just the bbox. This is critical:
    a real trunk with leaves draped in front would have ~70%
    green pixels in its bbox (because the bbox includes the
    leaves), but only ~5-10% green in the actual trunk mask
    (which is just bark pixels). Mask-based stats correctly
    keep real trunks; bbox-based stats would over-reject.

    Falls back to bbox-based stats if no mask provided (e.g.,
    a future detector that returns boxes only).

    For each trunk pixel set, count pixels that are:
      - GREEN-DOMINANT: G > R + 15 AND G > B + 15
      - BROWN/GREY:     R near G near B (within 20) OR R > G + 10

    Reject the trunk if more than --max-green-dominant-pct of
    its pixels are green-dominant AND it has insufficient brown/
    grey pixels (less than --min-brown-pct). Dual check handles
    edge cases like a trunk with foliage in front (lots of green
    but also brown).

    Returns (kept_boxes, kept_scores) -- same shape as input
    but filtered.
    """
    if not trunk_boxes:
        return [], []
    kept_boxes: list = []
    kept_scores: list = []
    H_img, W_img = rgb_arr.shape[:2]
    use_masks = (
        trunk_masks is not None
        and len(trunk_masks) == len(trunk_boxes)
    )
    for i, (box, score) in enumerate(zip(trunk_boxes, trunk_scores)):
        # Source pixels: prefer mask if available so leaves
        # adjacent to the trunk don't pollute the color stats.
        if use_masks:
            m = trunk_masks[i]
            if hasattr(m, "ndim") and m.ndim == 3:
                m = m.any(axis=0)
            mb = np.asarray(m).astype(bool)
            if mb.shape != (H_img, W_img):
                # Some checkpoints return per-mask sized arrays;
                # fall back to bbox if dims don't match.
                pixels = None
            else:
                ys_t, xs_t = np.where(mb)
                if ys_t.size == 0:
                    continue
                pixels = rgb_arr[ys_t, xs_t]
        else:
            pixels = None
        if pixels is None:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W_img, x2); y2 = min(H_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = rgb_arr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pixels = crop.reshape(-1, 3)
        r = pixels[:, 0].astype(np.int16)
        g = pixels[:, 1].astype(np.int16)
        b = pixels[:, 2].astype(np.int16)
        green_dom = (
            (g > r + green_dominance_threshold)
            & (g > b + green_dominance_threshold)
        )
        gray_ish = (np.abs(r - g) < 20) & (np.abs(g - b) < 20)
        red_warm = (r > g + 10) & (r > b + 10)
        brown_ish = gray_ish | red_warm
        total = float(pixels.shape[0])
        if total <= 0:
            continue
        green_pct = float(green_dom.sum()) / total
        brown_pct = float(brown_ish.sum()) / total
        if (green_pct > max_green_dominant_pct
                and brown_pct < min_brown_pct):
            continue
        kept_boxes.append(box)
        kept_scores.append(score)
    return kept_boxes, kept_scores


def partition_canopy_by_trunks(
    canopy_mask: np.ndarray,
    trunk_boxes: list,
    min_trunk_area_px: int = 1,
) -> tuple[list[dict], np.ndarray]:
    """Partition a canopy mask via Voronoi-style nearest-trunk
    assignment. Each canopy pixel is labeled with the index of
    the closest trunk's centroid.

    Returns (components, partition_labels) where:
      components is one dict per trunk-anchored canopy region:
        {"bbox": (x1, y1, x2, y2), "area": int, "label_id": int,
         "labels": partition_labels, "trunk_box": (x1, y1, x2, y2)}
      partition_labels is an (H, W) int32 array where 0 = outside
      canopy and 1..K = canopy pixels assigned to trunk index K-1.

    The partition uses L2 distance from each canopy pixel to each
    trunk's bbox centroid. Trunks merge their canopy region only
    if they're so close their nearest-neighbor cells contain no
    canopy pixels (rare with real trunks). Touching/overlapping
    canopies of physically separate trees get split because their
    trunks are spatially distinct.
    """
    h, w = canopy_mask.shape
    out_labels = np.zeros((h, w), dtype=np.int32)
    if not trunk_boxes:
        return [], out_labels
    centers = np.array(
        [
            [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]
            for b in trunk_boxes
        ],
        dtype=np.float32,
    )
    ys, xs = np.where(canopy_mask)
    if ys.size == 0:
        return [], out_labels
    pix = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    # (N, K) squared distances:
    diffs = pix[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sum(diffs * diffs, axis=2)
    nearest = np.argmin(dists, axis=1)  # (N,)
    # Label the partition (1-based so 0 = bg).
    out_labels[ys, xs] = (nearest + 1).astype(np.int32)
    components: list[dict] = []
    for t_idx, tbox in enumerate(trunk_boxes):
        sel = nearest == t_idx
        if not sel.any():
            continue
        cy = ys[sel]
        cx = xs[sel]
        area = int(cy.size)
        if area < min_trunk_area_px:
            continue
        bbox = (int(cx.min()), int(cy.min()),
                int(cx.max()), int(cy.max()))
        components.append({
            "bbox": bbox,
            "area": area,
            "label_id": int(t_idx + 1),
            "labels": out_labels,
            "trunk_box": tuple(int(round(x)) for x in tbox),
        })
    return components, out_labels


def augment_canopy_with_edge_trees(
    canopy_mask: np.ndarray,
    depth_mm: np.ndarray,
    rgb_arr: np.ndarray | None,
    *,
    depth_min_mm: float = 600.0,
    depth_max_mm: float = 3000.0,
    min_area_px: int = 400,
    edge_band_px: int = 20,
    min_height_px: int = 100,
    max_top_row: int = 250,
    min_aspect_ratio: float = 1.2,
    max_depth_std_mm: float = 1500.0,
    min_green_frac: float = 0.15,
) -> np.ndarray:
    """Add tree-shaped foreground-depth CCs touching the L/R
    frame edges to the canopy mask.

    The sprayer pipeline's build_tree_mask anchors only on the
    center column band (285-354), so half-trees at the L/R frame
    edges are not in its result. This wrapper recovers them but
    is RESTRICTIVE -- only CCs that look like trees (tall +
    narrow + depth-coherent + green) get added. Without these
    shape filters, large grass regions or tree-and-grass merged
    CCs would also pass and contaminate the canopy mask with
    ground pixels.

    Shape filter chain:
      1. Touches L or R edge within edge_band_px         (geometry)
      2. area >= min_area_px                              (size floor)
      3. bbox height >= min_height_px                     (vertical extent)
      4. bbox top <= max_top_row                          (extends up)
      5. height/width >= min_aspect_ratio                 (TALL & NARROW
                                                           -- rejects grass
                                                           which is wide+short)
      6. depth std < max_depth_std_mm                     (depth-coherent
                                                           -- a tree has
                                                           similar depth
                                                           top to bottom;
                                                           grass has a
                                                           ground-depth
                                                           gradient)
      7. green-leaf pixel frac >= min_green_frac          (foliage-like
                                                           -- rejects
                                                           non-vegetation)
    """
    try:
        import cv2 as _cv2
    except Exception:
        return canopy_mask
    if canopy_mask is None or depth_mm is None:
        return canopy_mask
    h, w = depth_mm.shape
    depth_f = depth_mm.astype(np.float32)
    fg = (
        (depth_f >= float(depth_min_mm))
        & (depth_f <= float(depth_max_mm))
    )
    H_c = S_c = V_c = None
    green_pixel = None
    if rgb_arr is not None and rgb_arr.shape[:2] == (h, w):
        try:
            hsv = _cv2.cvtColor(
                rgb_arr.astype(np.uint8), _cv2.COLOR_RGB2HSV,
            )
            H_c, S_c, V_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            sky_blue = (H_c >= 85) & (H_c <= 145) & (S_c > 8)
            sky_bright = (V_c > 185) & (S_c < 15)
            fg = fg & ~(sky_blue | sky_bright)
            # Leaf-green pixel mask for the foliage check.
            green_pixel = (
                (H_c >= 35) & (H_c <= 85)
                & (S_c >= 40) & (V_c >= 35)
            )
        except Exception:
            pass
    fg_u8 = fg.astype(np.uint8) * 255
    n_cc, labels, stats, _ = _cv2.connectedComponentsWithStats(
        fg_u8, connectivity=8,
    )
    if n_cc <= 1:
        return canopy_mask
    augmented = canopy_mask.astype(bool).copy()
    added = False
    for cc_id in range(1, n_cc):
        x, y, ww, hh, area = stats[cc_id]
        # 2. size
        if area < min_area_px:
            continue
        # 3. height
        if hh < min_height_px:
            continue
        # 4. top extends up
        if y > max_top_row:
            continue
        # 5. aspect ratio (tall + narrow)
        if ww > 0 and (hh / float(ww)) < min_aspect_ratio:
            continue
        # 1. edge contact
        x_max = x + ww - 1
        touches_edge = (
            x <= edge_band_px or x_max >= w - 1 - edge_band_px
        )
        if not touches_edge:
            continue
        # CC mask for the remaining checks.
        cc_pix = (labels == cc_id)
        # 6. depth coherence
        cc_d = depth_f[cc_pix]
        cc_d = cc_d[(cc_d >= float(depth_min_mm))
                    & (cc_d <= float(depth_max_mm))]
        if cc_d.size >= 50:
            if float(cc_d.std()) > float(max_depth_std_mm):
                continue
        # 7. green foliage frac
        if min_green_frac > 0 and green_pixel is not None:
            green_in_cc = float((cc_pix & green_pixel).sum())
            frac_g = green_in_cc / float(area)
            if frac_g < min_green_frac:
                continue
        augmented |= cc_pix
        added = True
    if not added:
        return canopy_mask
    return augmented


def build_canopy_from_sam_trees(
    tree_masks,
    tree_scores,
    depth_mm: np.ndarray | None,
    frame_shape: tuple[int, int],
    *,
    min_score: float = 0.15,
    depth_min_mm: float = 600.0,
    depth_max_mm: float = 3000.0,
    min_pixels: int = 500,
    position_min_lower_frac: float = 0.20,
    require_depth_check: bool = True,
    min_valid_depth_frac: float = 0.30,
    rgb_arr: np.ndarray | None = None,
    rgb_fallback_min_vegetation_frac: float = 0.20,
    max_top_row: int = 280,
    min_aspect_ratio: float = 0.5,
    max_depth_row_corr: float = 0.70,
) -> np.ndarray:
    """Build a canopy mask from SAM 3 tree segmentations.

    SAM 3 prompted with 'apple tree' (or similar) returns one
    mask per detected tree with a confidence score. This helper
    unions them into a single canopy mask after a few sanity
    filters:

      1. Confidence score >= min_score                 (skip low-conf)
      2. Mask area >= min_pixels                       (size floor)
      3. Mask extends into the lower portion of the
         frame: max(y) >= H * (1 - position_min_lower_frac)
                                                       (rejects sky-only
                                                        detections)
      4. Median depth of valid-depth pixels in mask
         is within [depth_min_mm, depth_max_mm]        (rejects far
                                                        background trees)

    Returns a (H, W) bool canopy mask. Empty if no SAM trees
    pass the filters.
    """
    h, w = frame_shape
    canopy = np.zeros((h, w), dtype=bool)
    if tree_masks is None or len(tree_masks) == 0:
        return canopy
    if tree_scores is None:
        tree_scores = [1.0] * len(tree_masks)
    pos_threshold_y = int(h * (1.0 - position_min_lower_frac))
    valid_depth = None
    if depth_mm is not None and require_depth_check:
        valid_depth = (depth_mm > 0) & (depth_mm < 60000)
    # Optional RGB-based vegetation pixel mask. Used as a fallback
    # when a SAM tree mask has too little valid depth to verify it
    # via the depth check. Real apple trees (even blossom-dominant
    # ones) have substantial vegetation pixel content (green leaves
    # AND/OR white/pink blossoms); far-background detections do not.
    veg_pixel = None
    if (rgb_fallback_min_vegetation_frac > 0
            and rgb_arr is not None
            and rgb_arr.shape[:2] == (h, w)):
        try:
            import cv2 as _cv2_v
            hsv_v = _cv2_v.cvtColor(
                rgb_arr.astype(np.uint8), _cv2_v.COLOR_RGB2HSV,
            )
            H_v = hsv_v[:, :, 0]
            S_v = hsv_v[:, :, 1]
            V_v = hsv_v[:, :, 2]
            green_v = (
                (H_v >= 35) & (H_v <= 85)
                & (S_v >= 40) & (V_v >= 35)
            )
            white_v = (S_v <= 50) & (V_v >= 120)
            pink_v = (
                (((H_v >= 0) & (H_v <= 20))
                 | ((H_v >= 150) & (H_v <= 179)))
                & (S_v >= 20) & (V_v >= 100)
            )
            veg_pixel = green_v | white_v | pink_v
        except Exception:
            veg_pixel = None
    for mask, score in zip(tree_masks, tree_scores):
        if float(score) < min_score:
            continue
        m = np.asarray(mask).astype(bool)
        if m.ndim == 3:
            m = m.any(axis=0)
        if m.shape != (h, w):
            continue
        if int(m.sum()) < min_pixels:
            continue
        # Position check: must extend into the lower portion
        # of the frame so sky-only / distant-line detections
        # are rejected.
        ys_any = m.any(axis=1)
        ys = np.where(ys_any)[0]
        if ys.size == 0 or int(ys.max()) < pos_threshold_y:
            continue
        # Extends-up check: a TREE extends into the upper
        # canopy band (top row <= max_top_row). A GROUND BAND
        # in the middle of the frame stays in the lower
        # portion (top row >> max_top_row). This is the
        # primary discriminator for SAM detecting grass /
        # ground as 'tree'.
        if int(ys.min()) > max_top_row:
            continue
        # Aspect ratio: trees are not extremely wide. A wide
        # horizontal grass band gets aspect h/w << 0.5.
        xs_any = m.any(axis=0)
        xs = np.where(xs_any)[0]
        if xs.size == 0:
            continue
        bb_w = int(xs.max() - xs.min() + 1)
        bb_h = int(ys.max() - ys.min() + 1)
        if bb_w > 0 and (bb_h / float(bb_w)) < min_aspect_ratio:
            continue
        # Depth-row correlation: ground has monotonic depth
        # increase with row (correlation ~ 1); a tree has
        # similar depth top-to-bottom (correlation near 0).
        # Reject masks with strong row-depth correlation.
        if (max_depth_row_corr < 1.0
                and depth_mm is not None
                and valid_depth is not None):
            ys_in, xs_in = np.where(m & valid_depth)
            if ys_in.size >= 100:
                ds_in = depth_mm[ys_in, xs_in].astype(np.float64)
                yf = ys_in.astype(np.float64)
                yf -= yf.mean()
                df = ds_in - ds_in.mean()
                denom = float(
                    np.sqrt((yf * yf).sum())
                    * np.sqrt((df * df).sum())
                )
                if denom > 0:
                    corr = float((yf * df).sum() / denom)
                    if abs(corr) > max_depth_row_corr:
                        continue
        # Depth check with RGB fallback. Two-tier:
        #   1. If valid_depth_frac >= threshold, use median-depth
        #      check (the trustworthy path).
        #   2. If valid_depth_frac is too low to median-check,
        #      fall back to RGB: require the mask to have at
        #      least rgb_fallback_min_vegetation_frac of pixels
        #      that look like leaves OR blossoms. This admits
        #      apple trees with poor D455 depth returns (thin
        #      foliage, edge-of-FOV) while still rejecting far-
        #      background detections (which have neither valid
        #      depth NOR vegetation color).
        if (require_depth_check and depth_mm is not None
                and valid_depth is not None):
            mask_area = float(m.sum())
            md = depth_mm[m & valid_depth]
            valid_frac = (
                float(md.size) / mask_area if mask_area > 0 else 0.0
            )
            if valid_frac >= min_valid_depth_frac and md.size >= 50:
                # Tier 1: depth median check
                med = float(np.median(md))
                if med < depth_min_mm or med > depth_max_mm:
                    continue
            else:
                # Tier 2: RGB vegetation-fraction fallback
                if (veg_pixel is None
                        or rgb_fallback_min_vegetation_frac <= 0):
                    # No RGB fallback available -- reject
                    continue
                veg_frac = (
                    float((m & veg_pixel).sum())
                    / float(mask_area)
                ) if mask_area > 0 else 0.0
                if veg_frac < rgb_fallback_min_vegetation_frac:
                    continue
        canopy |= m
    return canopy


def fill_small_canopy_holes(
    canopy_mask: np.ndarray,
    *,
    max_hole_area_px: int = 200,
) -> np.ndarray:
    """Fill SMALL interior holes in the canopy mask.

    After sky / background-depth / stake exclusion, the tree
    canopy mask often has many small holes (sky pixels between
    branches that were correctly removed). Visually the tree
    becomes a constellation of blobs instead of one silhouette.
    Filling small holes restores the tree's outer shape so
    branches connecting leafy regions don't appear as missing
    seams.

    Algorithm: flood-fill the canopy COMPLEMENT from a frame
    corner; any non-canopy pixel NOT reached by the flood is an
    "interior hole" surrounded by canopy. Connected-component
    label the holes, fill only those whose area <=
    max_hole_area_px (avoids re-adding large excluded regions
    like stakes or wide background patches).
    """
    try:
        import cv2 as _cv2
    except Exception:
        return canopy_mask
    if canopy_mask is None:
        return canopy_mask
    cm = canopy_mask.astype(bool)
    if not cm.any():
        return cm
    h, w = cm.shape
    cm_u8 = cm.astype(np.uint8) * 255
    # Pad with 1-px zero border so the flood from (0, 0) always
    # starts in the EXTERIOR (regardless of whether the canopy
    # touches the original frame edge).
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = cm_u8
    flood_aux = np.zeros((h + 4, w + 4), dtype=np.uint8)
    _cv2.floodFill(padded, flood_aux, (0, 0), 255)
    # interior_holes = pixels that are 0 in original AND NOT
    # reached by flood. After flood, EXTERIOR is 255 and
    # INTERIOR HOLES are still 0. Inverting gives holes.
    interior_holes = _cv2.bitwise_not(padded)[1:-1, 1:-1]
    if not interior_holes.any():
        return cm
    # CC-label the holes, keep only small ones to fill.
    n_h, h_labels, h_stats, _ = _cv2.connectedComponentsWithStats(
        interior_holes, connectivity=8,
    )
    out = cm.copy()
    for hid in range(1, n_h):
        area = int(h_stats[hid, _cv2.CC_STAT_AREA])
        if area <= int(max_hole_area_px):
            out = out | (h_labels == hid)
    return out


def crop_canopy_below_trunks(
    canopy_mask: np.ndarray,
    trunk_boxes: list,
    *,
    buffer_below_px: int = 30,
) -> np.ndarray:
    """Crop each canopy CC at the bottom of its trunk(s).

    A SAM tree mask often extends below the actual tree into
    the GROUND, especially when foreground depth at the grass
    is similar to the tree's depth. The trunk's bbox bottom
    marks where the trunk meets the ground, so anything in
    the same canopy CC below trunk_y2 + buffer is ground and
    should be removed.

    Implementation:
      1. CC-label the canopy mask.
      2. For each CC, find every trunk bbox whose interior
         contains at least one canopy-mask pixel of that CC.
      3. Cut the CC at row max(trunk_y2 for those trunks) +
         buffer_below_px. Pixels of this CC below that row
         are removed.

    CCs that don't overlap any trunk bbox are left alone.
    Buffer accounts for low-hanging branches just below the
    trunk bbox bottom.
    """
    if not trunk_boxes:
        return canopy_mask
    if canopy_mask is None:
        return canopy_mask
    cm = canopy_mask.astype(bool).copy()
    if not cm.any():
        return cm
    try:
        import cv2 as _cv2
    except Exception:
        return cm
    h, w = cm.shape
    cm_u8 = cm.astype(np.uint8) * 255
    n_cc, labels, _stats, _ = _cv2.connectedComponentsWithStats(
        cm_u8, connectivity=8,
    )
    if n_cc <= 1:
        return cm
    out = cm.copy()
    for cc_id in range(1, n_cc):
        cc_pix = (labels == cc_id)
        if not cc_pix.any():
            continue
        cc_trunk_y2: list[int] = []
        for tb in trunk_boxes:
            x1 = max(0, int(round(float(tb[0]))))
            y1 = max(0, int(round(float(tb[1]))))
            x2 = min(w - 1, int(round(float(tb[2]))))
            y2 = min(h - 1, int(round(float(tb[3]))))
            if x2 < x1 or y2 < y1:
                continue
            if cc_pix[y1:y2 + 1, x1:x2 + 1].any():
                cc_trunk_y2.append(y2)
        if not cc_trunk_y2:
            continue
        cut_row = max(cc_trunk_y2) + int(buffer_below_px)
        if cut_row < h - 1:
            below = np.zeros_like(out)
            below[cut_row + 1:, :] = True
            out[cc_pix & below] = False
    return out


def refine_canopy_mask(
    canopy_mask: np.ndarray,
    depth_mm: np.ndarray | None,
    rgb_arr: np.ndarray | None,
    *,
    upward_dilate_px: int = 20,
    close_px: int = 9,
    depth_min_mm: float = 600.0,
    depth_max_mm: float = 3000.0,
    exclude_sky: bool = True,
) -> np.ndarray:
    """Post-process the canopy mask:

      1. UPWARD expansion into foreground depth -- catches tree
         tops that SAM missed (sparse leaves with sky showing
         through). Per column, the canopy mask is allowed to
         extend up to upward_dilate_px rows ABOVE its current
         topmost pixel, into pixels that are foreground depth
         AND not sky. Downward expansion is NOT allowed (avoids
         pulling in ground).

      2. Sky exclusion -- pixels that are clearly sky (blue OR
         bright + low saturation) get removed from the canopy
         even if they were originally in it (SAM masks
         sometimes include sky-coloured patches between
         branches).

      3. Morphological closing -- bridges small gaps between
         nearby CCs (e.g., a tree split into two CCs because of
         a thin gap of sky / branches in the middle). Uses a
         small ELLIPSE kernel so it doesn't over-merge across
         large distances.

    Returns the refined canopy mask.
    """
    try:
        import cv2 as _cv2
    except Exception:
        return canopy_mask
    if canopy_mask is None:
        return canopy_mask
    cm = canopy_mask.astype(bool).copy()
    if not cm.any():
        return cm
    h, w = cm.shape

    # Foreground-depth + non-sky pixel mask. Used both for the
    # upward expansion (only add foreground-depth pixels) AND
    # the sky exclusion (remove sky pixels from the canopy).
    sky_pix = None
    fg_clean: np.ndarray | None = None
    if depth_mm is not None:
        fg = (
            (depth_mm.astype(np.float32) >= float(depth_min_mm))
            & (depth_mm.astype(np.float32) <= float(depth_max_mm))
        )
        if (rgb_arr is not None and rgb_arr.shape[:2] == (h, w)
                and exclude_sky):
            try:
                hsv = _cv2.cvtColor(
                    rgb_arr.astype(np.uint8), _cv2.COLOR_RGB2HSV,
                )
                H_c = hsv[:, :, 0]
                S_c = hsv[:, :, 1]
                V_c = hsv[:, :, 2]
                sky_blue = (H_c >= 85) & (H_c <= 145) & (S_c > 8)
                sky_bright = (V_c > 185) & (S_c < 15)
                sky_pix = sky_blue | sky_bright
                fg_clean = fg & ~sky_pix
            except Exception:
                fg_clean = fg
        else:
            fg_clean = fg

    # 1. UPWARD EXPANSION. For each column, find the topmost row
    # that has a True in cm. Then allow pixels in that column to
    # be added down to topmost-1, topmost-2, ..., topmost-N (i.e.,
    # ABOVE the existing canopy by up to upward_dilate_px rows),
    # provided those pixels are foreground depth + not sky.
    if upward_dilate_px > 0 and fg_clean is not None:
        any_in_col = cm.any(axis=0)
        # First True row per column. argmax returns 0 when no True.
        first_true = np.argmax(cm, axis=0).astype(np.int32)
        # If column has no True, set to h so zone is empty.
        first_true = np.where(any_in_col, first_true, h)
        # Build the upward-expansion zone: rows < first_true[col]
        # AND rows >= max(first_true[col] - upward_dilate_px, 0).
        row_idx = np.arange(h, dtype=np.int32)[:, None]
        upper_bound = first_true[None, :]
        lower_bound = np.maximum(
            first_true[None, :] - int(upward_dilate_px), 0,
        )
        zone = (row_idx >= lower_bound) & (row_idx < upper_bound)
        cm = cm | (zone & fg_clean)

    # 2. SKY EXCLUSION. Drop any pixels that look unambiguously
    # like sky (blue or bright + desaturated).
    if sky_pix is not None:
        cm = cm & ~sky_pix

    # 2b. BACKGROUND-DEPTH EXCLUSION. Drop pixels with VALID
    # background depth (> max). This removes background grass /
    # building / fence visible THROUGH leaf gaps in the SAM mask
    # while keeping pixels where depth is invalid (== 0; e.g.,
    # thin foliage where D455 didn't return a measurement). Sky
    # is handled separately by the HSV check above.
    if depth_mm is not None:
        bg_depth = depth_mm.astype(np.float32) > float(depth_max_mm)
        cm = cm & ~bg_depth

    # 3. CLOSING. Bridge small gaps between nearby CCs.
    if close_px > 0:
        kern = _cv2.getStructuringElement(
            _cv2.MORPH_ELLIPSE,
            (2 * int(close_px) + 1, 2 * int(close_px) + 1),
        )
        cm_u8 = cm.astype(np.uint8) * 255
        cm_u8 = _cv2.morphologyEx(cm_u8, _cv2.MORPH_CLOSE, kern)
        cm = cm_u8 > 0
        # Re-apply sky exclusion after closing (closing can fill
        # small sky holes between branches).
        if sky_pix is not None:
            cm = cm & ~sky_pix

    return cm


def filter_canopy_by_tree_shape(
    canopy_mask: np.ndarray,
    depth_mm: np.ndarray,
    rgb_arr: np.ndarray | None,
    *,
    min_aspect_ratio: float = 1.2,
    max_depth_std_mm: float = 1500.0,
    min_green_frac: float = 0.15,
    min_area_px: int = 400,
    max_top_row: int = 0,
) -> np.ndarray:
    """Post-process the canopy mask: remove connected components
    that don't look like a tree.

    Runs AFTER build_tree_mask + edge-tree augmentation. Catches
    cases where build_tree_mask's central anchor pulled grass
    into the canopy because grass extends up into the central
    columns (camera tilt, orchard slope), or where edge
    augmentation accidentally added a wide foreground-depth CC.

    Tree-shape filters per CC (any failure -> remove that CC):
      1. height / width >= min_aspect_ratio       (TALL & NARROW;
                                                   rejects grass
                                                   bands which are
                                                   wide + short)
      2. depth std < max_depth_std_mm             (depth-coherent;
                                                   rejects ground
                                                   gradient)
      3. green-leaf fraction >= min_green_frac    (foliage-like;
                                                   rejects fences
                                                   / soil)
      4. area >= min_area_px                      (size floor)
    """
    try:
        import cv2 as _cv2
    except Exception:
        return canopy_mask
    if canopy_mask is None:
        return canopy_mask
    cm_bool = canopy_mask.astype(bool)
    if not cm_bool.any():
        return canopy_mask
    cm_u8 = cm_bool.astype(np.uint8) * 255
    n_cc, labels, stats, _ = _cv2.connectedComponentsWithStats(
        cm_u8, connectivity=8,
    )
    if n_cc <= 1:
        return canopy_mask
    h, w = cm_bool.shape
    green_pixel = None
    if (rgb_arr is not None and rgb_arr.shape[:2] == (h, w)
            and min_green_frac > 0):
        try:
            hsv = _cv2.cvtColor(
                rgb_arr.astype(np.uint8), _cv2.COLOR_RGB2HSV,
            )
            H_c = hsv[:, :, 0]
            S_c = hsv[:, :, 1]
            V_c = hsv[:, :, 2]
            green_pixel = (
                (H_c >= 35) & (H_c <= 85)
                & (S_c >= 40) & (V_c >= 35)
            )
        except Exception:
            green_pixel = None
    out = np.zeros_like(cm_bool)
    depth_f = (
        depth_mm.astype(np.float32) if depth_mm is not None else None
    )
    n_kept = 0
    n_drop_aspect = 0
    n_drop_depth = 0
    n_drop_green = 0
    n_drop_area = 0
    n_drop_top_row = 0
    for cc_id in range(1, n_cc):
        x, y, ww, hh, area = stats[cc_id]
        if area < min_area_px:
            n_drop_area += 1
            continue
        # max_top_row check: a tree extends UP into the upper
        # canopy band (bbox top row <= max_top_row). Small ground
        # patches / wildflower blobs in the lower frame have
        # bbox top well below this. 0 disables.
        if max_top_row > 0 and int(y) > int(max_top_row):
            n_drop_top_row += 1
            continue
        if ww > 0 and (hh / float(ww)) < min_aspect_ratio:
            n_drop_aspect += 1
            continue
        cc_pix = (labels == cc_id)
        if depth_f is not None and max_depth_std_mm > 0:
            cc_d = depth_f[cc_pix]
            cc_d = cc_d[(cc_d > 0) & (cc_d < 60000)]
            if cc_d.size >= 50 and float(cc_d.std()) > max_depth_std_mm:
                n_drop_depth += 1
                continue
        if green_pixel is not None:
            frac_g = float((cc_pix & green_pixel).sum()) / float(area)
            if frac_g < min_green_frac:
                n_drop_green += 1
                continue
        out |= cc_pix
        n_kept += 1
    return out


def canopy_component_depth_medians(
    components: list[dict], depth_mm: np.ndarray,
    *, min_pixels: int = 50,
) -> dict[int, float]:
    """Return {label_id: median_depth_mm} for each canopy
    component, computed only over pixels with valid (>0, finite,
    < 60000 mm) depth. Components with fewer than ``min_pixels``
    valid-depth pixels are omitted from the result so callers
    can fall back to other heuristics for them.
    """
    out: dict[int, float] = {}
    if not components or depth_mm is None:
        return out
    labels_img = components[0].get("labels")
    if labels_img is None:
        return out
    valid = (depth_mm > 0) & (depth_mm < 60000)
    for c in components:
        lid = int(c.get("label_id", 0))
        if lid <= 0:
            continue
        pix = (labels_img == lid) & valid
        n = int(pix.sum())
        if n < min_pixels:
            continue
        out[lid] = float(np.median(depth_mm[pix]))
    return out


def extract_canopy_components(
    canopy_mask: np.ndarray, min_area_px: int = 500,
) -> list[dict]:
    """Split a per-frame canopy mask into per-tree connected
    components and return one dict per tree:
        {"bbox": (x1, y1, x2, y2), "area": int, "label_id": int,
         "labels": np.ndarray}
    Used by the per-session canopy IoU tracker so each tree
    gets a stable tree_id followed across frames.

    The label-id and full label image are returned so the caller
    can assign individual flower masks to the right tree by max
    pixel overlap, instead of relying on bbox-only IoU."""
    try:
        import cv2 as _cv2_cc
    except Exception:
        return []
    if canopy_mask is None or not canopy_mask.any():
        return []
    u8 = canopy_mask.astype(np.uint8)
    n_cc, labels, stats, _ = _cv2_cc.connectedComponentsWithStats(
        u8, connectivity=8,
    )
    out: list[dict] = []
    for cc_id in range(1, n_cc):
        x, y, w, h, area = stats[cc_id]
        if area < min_area_px:
            continue
        out.append({
            "bbox": (int(x), int(y), int(x + w), int(y + h)),
            "area": int(area),
            "label_id": int(cc_id),
            "labels": labels,
        })
    return out


class CanopyTracker:
    """IoU tracker for per-frame canopy connected components.
    Assigns a tree_id to each canopy CC and follows it across
    frames as the camera moves down the orchard row.

    Same greedy IoU matching as IoUTracker; bigger max_age
    default since trees are slower-moving relative to the
    camera than individual flowers, and the canopy mask can
    transiently shrink (e.g., tree clipped at frame edge for a
    few frames)."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
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

    def step(self, components: list[dict]) -> list[int]:
        """Update with one frame's CCs. Returns tree_id per CC,
        in the same order as the input list."""
        self.frame += 1
        if not components:
            return []
        active = [
            (tid, t) for tid, t in self.tracks.items()
            if self.frame - t["last_frame"] <= self.max_age + 1
        ]
        active.sort(key=lambda x: -x[1].get("max_area", 0))
        n = len(components)
        det_to_tid = [-1] * n
        used = [False] * n
        for tid, t in active:
            best_iou = self.iou_threshold
            best_d = -1
            for d in range(n):
                if used[d]:
                    continue
                iou = self._iou(t["bbox"], components[d]["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_d = d
            if best_d >= 0:
                used[best_d] = True
                det_to_tid[best_d] = tid
                t["bbox"] = components[best_d]["bbox"]
                t["last_frame"] = self.frame
                t["n_frames"] += 1
                a = int(components[best_d]["area"])
                if a > t.get("max_area", 0):
                    t["max_area"] = a
        for d in range(n):
            if used[d]:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "bbox": tuple(int(x) for x in components[d]["bbox"]),
                "first_frame": self.frame,
                "last_frame": self.frame,
                "n_frames": 1,
                "max_area": int(components[d]["area"]),
            }
            det_to_tid[d] = tid
        return det_to_tid

    def n_unique(self, min_frames: int = 1) -> int:
        return sum(
            1 for t in self.tracks.values()
            if t["n_frames"] >= min_frames
        )

    def summary(self) -> list[dict]:
        return [
            {"tree_id": tid, **t}
            for tid, t in self.tracks.items()
        ]


def assign_flowers_to_trees(
    flower_masks: np.ndarray,
    components: list[dict],
    tree_ids: list[int],
) -> list[int]:
    """For each flower mask, return the tree_id of the canopy
    component it MOST overlaps with (by pixel count). Returns
    -1 for masks with zero overlap on any component (e.g., a
    flower somehow outside all canopies; ought to be rare with
    the tree-mask gate active).

    Uses the labels image from extract_canopy_components so a
    flower mask is assigned to its actual containing tree
    rather than just the bbox-IoU best-match (which would mis-
    assign a flower at the boundary of two adjacent trees)."""
    if not components:
        return [-1] * len(flower_masks)
    # All components share the same labels image (it's the
    # connected-components result for this frame's canopy
    # mask). Take it from the first one.
    labels = components[0].get("labels")
    if labels is None:
        return [-1] * len(flower_masks)
    label_to_tree: dict[int, int] = {
        c["label_id"]: tid for c, tid in zip(components, tree_ids)
    }
    out: list[int] = []
    for m in flower_masks:
        mb = m.astype(bool)
        if mb.ndim == 3:
            mb = mb.any(axis=0)
        if not mb.any():
            out.append(-1)
            continue
        # Count pixels in each component's CC label.
        flower_labels = labels[mb]
        if flower_labels.size == 0:
            out.append(-1)
            continue
        unique, counts = np.unique(
            flower_labels[flower_labels > 0], return_counts=True,
        )
        if unique.size == 0:
            out.append(-1)
            continue
        best_label = int(unique[counts.argmax()])
        out.append(int(label_to_tree.get(best_label, -1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\matth\OneDrive\Desktop\Postdoc\Image\All2023")
    ap.add_argument("--out", default=r"C:\Users\matth\OneDrive\Desktop\Postdoc\Image\sam3_out")
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS,
                    help="One or more text prompts. Default: apple, branch, trunk, "
                         "flower, leaf, fruitlet.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sample-per-session", type=int, default=20,
                    help="How many frames to analyze per session. With "
                         "--sample-mode sequential (default), takes the FIRST "
                         "N chronologically-ordered complete frames. With "
                         "--sample-mode even, evenly spaces N across the "
                         "whole session. Use 0 for all frames.")
    ap.add_argument("--sample-mode", choices=("sequential", "even", "stride"),
                    default="sequential",
                    help="How --sample-per-session picks frames. 'sequential' "
                         "= the first N (matches typical 'first 100 frames' "
                         "intuition). 'even' = N evenly-spaced (good for "
                         "characterizing a whole session timeline). "
                         "'stride' = take every --sample-stride-th valid "
                         "frame (ignores --sample-per-session; gives "
                         "consistent temporal density per session).")
    ap.add_argument("--sample-stride", type=int, default=20,
                    help="When --sample-mode is 'stride', take every Nth "
                         "complete frame in chronological order (default "
                         "20 = ~5%% of frames). Per-session count varies "
                         "by session length, but each frame is roughly "
                         "20 frames apart so coverage is uniform across "
                         "the orchard pass.")
    ap.add_argument("--frame-range", nargs=2, type=int, metavar=("START", "END"),
                    help="Process consecutive frames imgs[START:END] from each "
                         "session instead of sampling. Tracker-friendly "
                         "subset. Overrides --sample-per-session when set.")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--all-folders", action="store_true",
                    help="Search every leaf folder, not just RGB/.")
    ap.add_argument("--require-all-modalities", action="store_true",
                    help="Skip any frame whose timestamp doesn't have matching "
                         "files in all four folders (RGB, depth, IR, PRGB). "
                         "Guarantees strict cross-modality alignment for the run.")
    ap.add_argument("--require-info-modality", action="store_true",
                    help="Also require an Info file (e.g. <stem>-Info.txt in "
                         "<session>/Info/) alongside RGB/depth/IR/PRGB. Only "
                         "applies when --require-all-modalities is also set.")
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--save-canopy-masks", action="store_true",
                    help="Save the per-frame canopy mask as a .npz file "
                         "at <out>/canopy_masks/<rel_path>.npz so it "
                         "can be reused downstream (Label Studio "
                         "polygon overlays, per-tree analytics, "
                         "external visualization). Each .npz contains "
                         "a single 'canopy' boolean array and "
                         "optionally per-tree partition labels and "
                         "trunk bboxes when --track-canopy is on.")
    ap.add_argument("--save-canopy-overlay", action="store_true",
                    help="Save a HUMAN-VIEWABLE per-frame JPG of the "
                         "RGB image with the canopy mask drawn on top "
                         "(red translucent fill + bright outline), at "
                         "<out>/canopy_overlays/<rel_path>.jpg. Use "
                         "this to debug 'why is my canopy mask "
                         "wrong' visually -- e.g., when an edge tree "
                         "is missing from the mask, or when grass is "
                         "being added as canopy by --canopy-include-"
                         "edge-trees. When --track-canopy is on, also "
                         "draws the per-tree partition labels in "
                         "different colors and the detected trunk "
                         "bboxes.")
    ap.add_argument("--save-depth-fg-overlay", action="store_true",
                    help="Save a per-frame DIAGNOSTIC JPG showing "
                         "the RAW foreground-depth mask (every pixel "
                         "with depth in [--depth-min-mm, --depth-max-"
                         "mm] AND not sky-blue / not sky-bright), at "
                         "<out>/depth_fg_overlays/<rel_path>.jpg. "
                         "Compare to canopy_overlays to see whether "
                         "the depth data captures a tree but the "
                         "filter chain rejects it (filter issue) or "
                         "the depth itself doesn't have the tree "
                         "(data issue). Different colors mark each "
                         "depth band: green=close (600-1500mm), "
                         "yellow=mid (1500-2200mm), orange=far "
                         "(2200-3000mm).")
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--save-empty-overlays", action="store_true",
                    help="Also save overlay JPGs for frames where the "
                         "prompt produced 0 detections (e.g., frames "
                         "with no visible flowers). Useful for debugging "
                         "-- you can confirm SAM 3 saw nothing rather "
                         "than guessing why a frame is missing from the "
                         "overlay folder. Off by default since it 6-10x "
                         "the overlay count on bulk runs.")
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
    ap.add_argument("--tree-mask-dilate-px", type=int, default=0,
                    help="Dilate the canopy mask by this many pixels "
                         "BEFORE the overlap check. Catches flowers on "
                         "branches / twigs that extend slightly past "
                         "the bulk of the canopy mask -- those would "
                         "otherwise fail tree-mask-min-overlap because "
                         "the canopy mask CC filter dropped the thin "
                         "branch as a small fragment. Recommended 20-40 "
                         "for whole-image YOLO labeling. Default 0.")
    ap.add_argument("--tree-mask-min-canopy-frac", type=float, default=0.01,
                    help="Skip the tree-mask gate when the canopy mask "
                         "covers less than this fraction of the frame. "
                         "Without this, partial-frame trees that produce "
                         "a tiny canopy mask have ALL their flowers "
                         "rejected by the 10%% overlap requirement "
                         "(small canopy * 10%% = essentially none). "
                         "Default 0.01 = 1%% of frame.")
    ap.add_argument("--flower-require-tree-in-frame", action="store_true",
                    help="Reject ALL flower detections in any frame "
                         "where the pipeline finds neither a meaningful "
                         "FOREGROUND canopy mask (>= --tree-mask-min-"
                         "canopy-frac AND median depth <= "
                         "--flower-foreground-canopy-max-depth-mm) "
                         "NOR any detected trunks (when --track-canopy "
                         "is on). On those 'no foreground tree in view' "
                         "frames, any 'flower' detection is a false "
                         "positive (grass weeds, tiny background tree, "
                         "distant orchard rows, dust). Recommended on "
                         "for whole-image YOLO labeling.")
    ap.add_argument("--flower-foreground-canopy-max-depth-mm",
                    type=float, default=2500.0,
                    help="When --flower-require-tree-in-frame is on, "
                         "the canopy mask is treated as 'foreground' "
                         "only if its median valid-depth value is at or "
                         "below this threshold (default 2500 mm = 2.5 "
                         "m). A tiny background-row canopy patch may "
                         "still pass the size threshold but sits at "
                         "3-5 m; this depth check rejects it. Set 0 "
                         "to disable the depth side of the check "
                         "(canopy size alone gates).")
    ap.add_argument("--canopy-include-edge-trees", action="store_true",
                    help="After build_tree_mask runs, also add "
                         "foreground-depth connected components that "
                         "touch the LEFT or RIGHT frame edges to the "
                         "canopy mask. Recovers half-trees at frame "
                         "edges that the sprayer pipeline's center-"
                         "anchored algorithm rejects: when multiple "
                         "trees are in frame at different depths the "
                         "median center_depth falls between them and "
                         "the +/- 500 mm band excludes both. Edge "
                         "trees are by construction half-cropped, so "
                         "this gate exempts them from the global "
                         "depth-band check.")
    ap.add_argument("--canopy-edge-tree-min-area-px", type=int,
                    default=800,
                    help="Minimum area for an edge-touching CC to be "
                         "added as canopy by --canopy-include-edge-"
                         "trees. Default 800 px filters small noise.")
    ap.add_argument("--canopy-edge-tree-min-height-px", type=int,
                    default=150,
                    help="Minimum bbox HEIGHT for an edge-touching CC "
                         "to be added as canopy. Trees are tall. "
                         "Default 150 px (~ 1/3 of a 480-row frame).")
    ap.add_argument("--canopy-edge-tree-max-top-row", type=int,
                    default=200,
                    help="Maximum bbox TOP ROW for an edge-touching CC "
                         "to be added as canopy. Tree CCs extend into "
                         "the upper frame. Default 200.")
    ap.add_argument("--canopy-edge-tree-min-aspect-ratio", type=float,
                    default=1.5,
                    help="Minimum bbox HEIGHT / WIDTH ratio for an "
                         "edge-touching CC to be added as canopy. "
                         "Trees are tall + narrow; grass is wide + "
                         "short. Default 1.5 = height must be at least "
                         "1.5x width. THIS IS THE KEY FILTER for "
                         "rejecting grass CCs that touch the frame "
                         "edges -- without it, a 600x250 grass region "
                         "passes the height + top-row checks.")
    ap.add_argument("--canopy-edge-tree-max-depth-std-mm", type=float,
                    default=800.0,
                    help="Maximum depth standard deviation (mm) within "
                         "an edge-touching CC for it to be added as "
                         "canopy. A tree silhouette has similar depth "
                         "from top to bottom (low std); grass has a "
                         "ground-plane depth gradient (high std). "
                         "Default 800 mm.")
    ap.add_argument("--canopy-edge-tree-min-green-frac", type=float,
                    default=0.30,
                    help="Minimum fraction of CC pixels that must be "
                         "leaf-green (HSV H 35-85, S>=40, V>=35) for "
                         "the CC to be added as canopy. Trees are "
                         "vegetation; ground / fence / building edges "
                         "are not. Default 0.30. Set 0 to disable.")
    ap.add_argument("--canopy-edge-tree-max-depth-mm", type=float,
                    default=3000.0,
                    help="Maximum depth (mm) for the foreground-depth "
                         "band used by edge-tree augmentation. Tighter "
                         "than the global --depth-max-mm; default 3000 "
                         "mm so far-row trees aren't pulled in.")
    # SAM 3-based canopy detection. PRIMARY canopy source when
    # enabled. SAM 3 prompted with 'apple tree' (or similar)
    # returns pixel-accurate tree segmentations directly; we
    # union them into the canopy mask after filtering by score,
    # size, position, and median depth. When non-empty, this
    # REPLACES the build_tree_mask + edge augmentation + tree-
    # shape filter chain. When empty (or disabled), we fall
    # back to that chain so existing setups keep working.
    ap.add_argument("--canopy-sam-prompt", default="",
                    help="If set, run SAM 3 with this prompt and "
                         "use its segmentations as the canopy mask. "
                         "Recommended: 'apple tree' or 'tree canopy'. "
                         "When SAM 3 returns one or more masks "
                         "passing the filters (--canopy-sam-min-*), "
                         "they become the canopy_mask_img directly, "
                         "skipping build_tree_mask + edge "
                         "augmentation + tree-shape filter. Falls "
                         "back to build_tree_mask if SAM finds "
                         "nothing in a frame. Default '' = disabled.")
    ap.add_argument("--canopy-sam-multi-prompts", nargs="+",
                    default=None,
                    help="Additional SAM prompts to try for the "
                         "canopy. All masks across these prompts "
                         "are unioned (after individual filtering) "
                         "to form the SAM canopy. Useful when the "
                         "primary --canopy-sam-prompt misses some "
                         "trees -- different phrasings catch "
                         "different cases. Try: 'apple tree' 'tree "
                         "branches' 'tree canopy' 'fruit tree'. The "
                         "primary prompt is automatically included.")
    ap.add_argument("--canopy-sam-rgb-fallback-min-veg-frac",
                    type=float, default=0.20,
                    help="When a SAM tree mask has too little valid "
                         "depth to verify (<= --canopy-sam-min-valid"
                         "-depth-frac), fall back to checking RGB "
                         "vegetation content. The mask must have at "
                         "least this fraction of pixels that look "
                         "like green leaves OR white/pink blossoms. "
                         "Default 0.20 (20%%). Admits trees with "
                         "poor D455 depth returns while still "
                         "rejecting far-background detections.")
    ap.add_argument("--canopy-sam-only", action="store_true",
                    help="When SAM canopy finds anything, use ONLY "
                         "the SAM canopy. The heuristic build_tree"
                         "_mask + edge augmentation only runs as a "
                         "FULL fallback when SAM is empty. Prevents "
                         "the heuristic from adding grass/ground "
                         "noise when SAM has a clean tree mask.")
    # Post-process canopy refinement: upward expansion + sky
    # exclusion + morphological closing. Runs on the FINAL
    # canopy mask after SAM and heuristic are unioned.
    ap.add_argument("--canopy-refine", action="store_true",
                    help="Apply post-processing refinement to the "
                         "final canopy mask: (1) expand upward into "
                         "foreground-depth pixels per column to "
                         "catch tree tops SAM missed, (2) remove "
                         "pixels that are sky (HSV check), (3) "
                         "morphological closing to bridge nearby "
                         "tree CC fragments. Recommended.")
    ap.add_argument("--canopy-refine-upward-dilate-px", type=int,
                    default=20,
                    help="Maximum rows ABOVE the canopy mask's "
                         "current topmost pixel that the refinement "
                         "can extend into (filling in tree tops "
                         "where SAM cut off). Foreground-depth + "
                         "non-sky pixels in this band are added. "
                         "Downward expansion is NOT allowed (avoids "
                         "pulling in ground). Default 20.")
    ap.add_argument("--canopy-refine-close-px", type=int, default=9,
                    help="Morphological closing kernel radius for "
                         "bridging nearby canopy CC fragments. "
                         "Default 9 (= 19x19 ellipse) -- bridges "
                         "gaps up to ~18 px without over-merging. "
                         "Set 0 to disable closing.")
    ap.add_argument("--canopy-sam-max-top-row", type=int, default=280,
                    help="A SAM tree mask must extend UPWARD into the "
                         "upper canopy band (bbox top row <= this). "
                         "Trees extend high; grass / ground bands in "
                         "the middle of the frame stay below. PRIMARY "
                         "discriminator against SAM segmenting grass "
                         "as 'tree'. Default 280.")
    ap.add_argument("--canopy-sam-min-aspect-ratio", type=float,
                    default=0.5,
                    help="Minimum bbox h/w ratio for a SAM tree mask. "
                         "Default 0.5 admits full-canopy trees "
                         "(aspect ~1) while rejecting horizontal "
                         "grass bands (aspect << 0.5). 0 disables.")
    ap.add_argument("--canopy-sam-max-depth-row-corr", type=float,
                    default=0.70,
                    help="Reject SAM tree masks whose depth correlates "
                         "with row (depth-row correlation > this). "
                         "Ground has monotonic depth increase with "
                         "row (corr ~ 1); a tree's depth is roughly "
                         "constant top-to-bottom (corr near 0). "
                         "Default 0.70. Set 1.0 to disable.")
    ap.add_argument("--canopy-sam-min-score", type=float, default=0.15,
                    help="Minimum SAM 3 detection score for a tree "
                         "mask to be included in the canopy. Default "
                         "0.15.")
    ap.add_argument("--canopy-sam-min-pixels", type=int, default=500,
                    help="Minimum mask area (in px) for a SAM 3 tree "
                         "detection to be included in the canopy. "
                         "Default 500.")
    ap.add_argument("--canopy-sam-min-lower-frac", type=float,
                    default=0.20,
                    help="The mask must extend into the lower "
                         "fraction of the frame: max(y) >= H * (1 - "
                         "this). Rejects sky-only / horizon-only "
                         "detections. Default 0.20 (mask must reach "
                         "into the lower 20%% of the frame).")
    ap.add_argument("--canopy-sam-depth-check", action="store_true",
                    default=True,
                    help="Require SAM tree masks to have median depth "
                         "in [--depth-min-mm, --depth-max-mm]. Default "
                         "on; rejects distant background trees that "
                         "SAM might still segment.")
    ap.add_argument("--no-canopy-sam-depth-check",
                    dest="canopy_sam_depth_check",
                    action="store_false",
                    help="Disable the depth check on SAM tree masks. "
                         "Useful when depth is unreliable.")
    ap.add_argument("--canopy-sam-min-valid-depth-frac", type=float,
                    default=0.30,
                    help="Require at least this fraction of a SAM "
                         "tree mask's pixels to have VALID DEPTH "
                         "(non-zero, < 60000 mm). Default 0.30. "
                         "Critical for rejecting far-background "
                         "detections: D455 sensor max range gives "
                         "zero valid returns past ~5 m, so a far-"
                         "background SAM mask has almost no valid "
                         "depth. Without this floor, the median-"
                         "depth check silently passes ('not enough "
                         "info, benefit of doubt') and the bogus "
                         "far-background detection becomes canopy.")
    ap.add_argument("--canopy-sam-supplement-with-edge-aug",
                    action="store_true", default=True,
                    help="When SAM canopy is in use, ALSO run the "
                         "edge-tree augmentation on top (union of "
                         "SAM masks + edge augmentation). Recovers "
                         "trees SAM partially or completely missed: "
                         "tops cut off, adjacent trees not detected, "
                         "half-trees at frame edges. Default ON.")
    ap.add_argument("--no-canopy-sam-supplement-with-edge-aug",
                    dest="canopy_sam_supplement_with_edge_aug",
                    action="store_false",
                    help="Use SAM canopy alone, without supplementing "
                         "with edge augmentation. Tighter mask but "
                         "may miss trees SAM didn't segment fully.")

    # Post-processing tree-shape filter. Runs on the FINAL canopy
    # mask (after build_tree_mask + edge augmentation + dilation)
    # and removes any CC that doesn't look like a tree. This is
    # the SECOND LINE OF DEFENSE: build_tree_mask's central anchor
    # can pull grass into the canopy when the orchard floor
    # extends up into cols 285-354, and the per-CC tree-shape
    # filter rejects those wide+short bands.
    ap.add_argument("--canopy-filter-by-tree-shape", action="store_true",
                    help="After all canopy mask construction, run a "
                         "per-CC filter that drops CCs not shaped "
                         "like a tree (wide-and-short = ground; "
                         "depth-incoherent = ground; not green = "
                         "fence/soil). Critical safety net for cases "
                         "where build_tree_mask's central anchor or "
                         "edge augmentation accidentally pulled non-"
                         "tree CCs into the canopy mask.")
    ap.add_argument("--canopy-filter-min-aspect-ratio", type=float,
                    default=1.2,
                    help="Minimum bbox HEIGHT / WIDTH for a canopy CC "
                         "to survive the post-processing filter. "
                         "Default 1.2 (modestly tall). Trees easily "
                         "pass this; horizontal grass bands fail.")
    ap.add_argument("--canopy-filter-max-depth-std-mm", type=float,
                    default=1500.0,
                    help="Maximum depth std (mm) within a canopy CC "
                         "for it to survive the post-processing "
                         "filter. Default 1500 mm. Trees have depth "
                         "consistent within +/- 750 mm; ground has a "
                         "depth gradient often exceeding this.")
    ap.add_argument("--canopy-filter-min-green-frac", type=float,
                    default=0.15,
                    help="Minimum leaf-green fraction within a canopy "
                         "CC for it to survive the post-processing "
                         "filter. Default 0.15 (lenient -- pink "
                         "blossoms reduce green frac, but real trees "
                         "still have substantial green leaves).")
    ap.add_argument("--canopy-filter-min-area-px", type=int,
                    default=400,
                    help="Minimum area for a canopy CC to survive "
                         "the post-processing filter. Default 400. "
                         "Small specks have unreliable shape stats.")
    ap.add_argument("--canopy-filter-max-top-row", type=int,
                    default=0,
                    help="If > 0, reject canopy CCs whose bbox top "
                         "row is BELOW this. A tree extends UP into "
                         "the upper canopy band (top row <= this). "
                         "Small ground / grass blobs in the lower "
                         "frame have top row >> this and get "
                         "rejected. Default 0 = disabled. Try 300.")
    # Trunk-based canopy refinements. Both use the SAM 3 trunk
    # detections that --track-canopy --canopy-track-method=trunk
    # already produces. Run AFTER the canopy mask is built and
    # filtered, BEFORE the trunk-Voronoi partition.
    ap.add_argument("--canopy-add-trunk-masks", action="store_true",
                    help="Union the SAM-detected trunk masks into "
                         "the canopy mask. A tree's two leafy regions "
                         "that were separate CCs (because the visible "
                         "trunk between them was filtered out) get "
                         "bridged through the trunk pixels into one "
                         "CC. The trunk mask itself is narrow so it "
                         "doesn't add ground.")
    ap.add_argument("--canopy-trunk-vertical-extension-px", type=int,
                    default=100,
                    help="When --canopy-add-trunk-masks is on, also "
                         "draw a thin vertical line from each trunk's "
                         "TOP going UP by this many pixels (5 px "
                         "wide centered on the trunk's column "
                         "centroid). Bridges upper canopy regions "
                         "that sit ABOVE the visible trunk to the "
                         "trunk-bearing main canopy. The trunk's "
                         "visible top isn't always its actual top -- "
                         "leaves often occlude the upper trunk "
                         "section. Default 100. Set 0 to disable.")
    ap.add_argument("--canopy-exclude-painted-stakes", action="store_true",
                    help="Remove painted-green-stake pixels from the "
                         "canopy mask. Stakes are saturated green at "
                         "low brightness (HSV H in green range, S "
                         "high, V low) and are visually distinct from "
                         "leaf green (higher V). Tunable via the "
                         "--canopy-stake-* flags below.")
    ap.add_argument("--canopy-stake-hue-min", type=int, default=30,
                    help="Stake hue lower bound (default 30, in "
                         "OpenCV 0-179).")
    ap.add_argument("--canopy-stake-hue-max", type=int, default=90,
                    help="Stake hue upper bound (default 90).")
    ap.add_argument("--canopy-stake-sat-min", type=int, default=100,
                    help="Stake minimum saturation (default 100). "
                         "Real stakes are saturated; leaves under "
                         "overcast lighting are less saturated.")
    ap.add_argument("--canopy-stake-val-max", type=int, default=120,
                    help="Stake maximum brightness V (default 120). "
                         "Stakes are dark (painted) -- leaves are "
                         "typically brighter.")
    ap.add_argument("--canopy-fill-small-holes", action="store_true",
                    help="After all canopy exclusions (sky, "
                         "background depth, stakes), fill small "
                         "interior holes in the canopy mask via "
                         "flood-fill + size-filtered re-add. Makes "
                         "the tree appear as a continuous silhouette "
                         "instead of a constellation of leafy blobs "
                         "separated by sky-gap holes that the "
                         "exclusions correctly removed but which "
                         "should be VISUALLY part of the tree mask.")
    ap.add_argument("--canopy-max-hole-area-px", type=int, default=300,
                    help="Maximum size of an interior hole that "
                         "--canopy-fill-small-holes will fill. "
                         "Larger holes (e.g., a stake or a wide "
                         "background patch) are left empty so we "
                         "don't undo those exclusions. Default 300.")
    ap.add_argument("--canopy-crop-below-trunk", action="store_true",
                    help="For each canopy CC that overlaps a SAM "
                         "trunk bbox, crop the CC at row "
                         "(trunk_y2 + buffer). Removes the GROUND "
                         "that the canopy mask incorrectly extended "
                         "into below the tree. The trunk's bbox "
                         "bottom marks where the trunk meets the "
                         "ground, so anything below that in the same "
                         "CC is ground.")
    ap.add_argument("--canopy-crop-below-trunk-buffer-px", type=int,
                    default=30,
                    help="Buffer (rows) added below trunk_y2 before "
                         "cropping the canopy CC. Accounts for low-"
                         "hanging branches that extend a bit below "
                         "the trunk bbox. Default 30.")
    ap.add_argument("--flower-max-behind-foreground-mm",
                    type=float, default=0.0,
                    help="After flowers are assigned to canopy "
                         "components (--track-canopy with trunk or "
                         "cc method), reject any flower whose host "
                         "tree's median depth is more than this many "
                         "mm behind the FOREGROUND tree (= the "
                         "closest canopy component in the frame). "
                         "Catches the 'flower on a background tree' "
                         "case: the per-mask depth cap passes if the "
                         "background tree is within the depth_max "
                         "limit, but the foreground tree is closer, "
                         "so this gate rejects flowers on the farther "
                         "tree relative to the closest one. Default "
                         "0.0 = disabled. Try 1000 mm.")
    # Canopy (per-tree) tracking. Assigns a tree_id to each canopy
    # connected component and follows it across frames via IoU on
    # the bbox of the CC. Each kept flower is then assigned to its
    # containing tree by max pixel overlap with the CC labels image.
    ap.add_argument("--track-canopy", action="store_true",
                    help="Enable per-tree tracking. Per session, the "
                         "CanopyTracker runs over canopy connected "
                         "components in each frame, assigning each tree "
                         "a tree_id followed across frames. Flowers "
                         "are then assigned to their containing tree's "
                         "id via max pixel overlap. Adds two output "
                         "CSVs: trees_summary.csv (one row per session "
                         "x tree) and trees_per_frame.csv (one row per "
                         "tree-frame appearance). Requires --tree-mask.")
    ap.add_argument("--canopy-track-iou", type=float, default=0.3,
                    help="IoU threshold for matching canopy CCs to "
                         "existing tree tracks across frames. Default "
                         "0.3 -- canopy bboxes shift more frame-to-"
                         "frame than flowers.")
    ap.add_argument("--canopy-track-max-age", type=int, default=5,
                    help="Frames a tree can be unseen before its "
                         "tree_id retires. Higher than the flower "
                         "tracker default (3) because trees can be "
                         "transiently clipped at frame edges.")
    ap.add_argument("--canopy-track-min-cc-area", type=int, default=500,
                    help="Minimum CC pixel area to count as a tree. "
                         "Below this, treat as canopy noise / fragment "
                         "and skip.")
    ap.add_argument("--canopy-track-method",
                    choices=("cc", "trunk"), default="cc",
                    help="How to partition the canopy mask into per-"
                         "tree regions. 'cc' = connected components "
                         "(default; cheap, fails on physically-touching "
                         "tree canopies). 'trunk' = SAM 3 detects "
                         "trunks per frame and the canopy is partitioned "
                         "by nearest-trunk assignment (Voronoi). 'trunk' "
                         "correctly distinguishes overlapping trees "
                         "because trunks remain spatially distinct even "
                         "when canopies merge in 2D. Adds ~1-2 sec/frame "
                         "of SAM 3 inference for the trunk prompt.")
    ap.add_argument("--canopy-trunk-min-score", type=float, default=0.20,
                    help="Min SAM 3 score for a trunk detection to be "
                         "used as a canopy partition anchor (default "
                         "0.20). Higher = fewer false trunks, lower = "
                         "more recall on partial-frame trunks.")
    ap.add_argument("--canopy-trunk-prompt", type=str,
                    default="apple tree trunk",
                    help="Text prompt used to detect trunks for "
                         "canopy partitioning when "
                         "--canopy-track-method=trunk.")
    ap.add_argument("--canopy-trunk-reject-green-stakes",
                    action="store_true",
                    help="Apply a color filter that rejects 'trunk' "
                         "detections matching painted-green support "
                         "stakes. Apple orchards routinely use green-"
                         "painted metal stakes next to young trees; "
                         "SAM 3 with the trunk prompt picks them up "
                         "as trunks. Without this filter, stake "
                         "positions become Voronoi anchors and the "
                         "canopy partition gets mis-assigned.")
    ap.add_argument("--canopy-trunk-max-green-pct", type=float,
                    default=0.35,
                    help="Reject trunk if more than this fraction of "
                         "its bbox pixels are green-dominant "
                         "(G > R+threshold AND G > B+threshold). "
                         "Default 0.35 = 35%%.")
    ap.add_argument("--canopy-trunk-green-threshold", type=int,
                    default=15,
                    help="A pixel is 'green-dominant' if G exceeds R "
                         "and B by at least this many counts. "
                         "Default 15.")
    ap.add_argument("--canopy-trunk-min-brown-pct", type=float,
                    default=0.20,
                    help="Override the green rejection if at least "
                         "this fraction of bbox pixels are brown / "
                         "grey / red-warm (real wooden trunk colors). "
                         "Catches trunks with foliage in front. "
                         "Default 0.20 = 20%%.")
    ap.add_argument("--canopy-trunk-max-depth-mm", type=float,
                    default=3000.0,
                    help="Reject trunk detections whose median valid "
                         "depth exceeds this many millimeters. Targets "
                         "the failure mode where SAM 3 detects trunks "
                         "on background-row trees (5+ m) and they "
                         "become Voronoi anchors, capturing background "
                         "canopy and leaking background flowers into "
                         "the count. Default 3000 (3 m). Set 0 to "
                         "disable. Asymmetric: only rejects when we "
                         "have CONFIRMED far depth -- partial-frame "
                         "trunks with sparse depth get the benefit "
                         "of the doubt.")
    ap.add_argument("--canopy-trunk-depth-min-pixels", type=int,
                    default=5,
                    help="Min valid-depth pixels required in a trunk "
                         "mask before --canopy-trunk-max-depth-mm "
                         "fires. Default 5: avoids over-rejecting "
                         "thin partial-frame trunks where most of "
                         "the trunk has no valid depth return.")
    ap.add_argument("--use-build-tree-mask", action="store_true",
                    help="Use the more robust tree_mask.build_tree_mask "
                         "function instead of compute_canopy_mask. Adds "
                         "HSV-based sky exclusion + ROI-anchored region "
                         "growing + forward-branch recovery on top of "
                         "the depth band, which captures partial-frame "
                         "trees that compute_canopy_mask drops because "
                         "of D455 sensor edge artifacts. Recommended "
                         "for whole-image YOLO labeling runs where "
                         "canopy mask is the primary spatial gate.")
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
    ap.add_argument("--flower-white-s-max", type=int, default=30,
                    help="Max HSV saturation for the 'white blossom' mask "
                         "in the in-loop refinement step (default 30). "
                         "Raise to capture more off-white / cream petals.")
    ap.add_argument("--flower-white-v-min", type=int, default=180,
                    help="Min HSV value (brightness) for the 'white "
                         "blossom' mask (default 180). LOWER for golden-"
                         "hour / overcast scenes where white petals come "
                         "back at V=140-170 instead of 200+.")
    ap.add_argument("--flower-pink-v-min", type=int, default=130,
                    help="Min HSV value for the 'pink blossom' mask "
                         "(default 130). Lower for shadowed/sidelit "
                         "blossoms.")
    # Cool-shadow + green-dominance + leaf-bud rules ported from the
    # sprayer pipeline's flower_detector. b_minus_r and g_minus_r are
    # signed RGB differences. These reject most non-flower bright
    # objects (specular leaf glints, sun-bleached cuticles, white
    # cars, building paint) that pass the HSV + depth checks.
    ap.add_argument("--flower-b-minus-r-max", type=int, default=10,
                    help="Max value of (B - R) for a pixel to count as "
                         "a WHITE petal. Real apple petals reflect "
                         "cool/blue sky onto their undersides, so they "
                         "test as B >= R or only slightly warmer. "
                         "Specular leaf glints, sun-bleached white "
                         "objects, and warm-toned false positives have "
                         "B - R > 0. The reference sprayer pipeline "
                         "uses 0 (very strict) during full bloom; we "
                         "default to 10 to be friendlier with the "
                         "lighting variation in your dataset, but "
                         "drop to 0 or -2 if you still see warm-toned "
                         "FPs sneaking through.")
    ap.add_argument("--flower-pink-b-minus-r-max", type=int, default=0,
                    help="Max (B - R) for a PINK pixel. Pink petals "
                         "are reddish but still cool-shifted; warm-leaf "
                         "glints that test as 'red hue' are excluded. "
                         "Default 0.")
    ap.add_argument("--flower-g-minus-r-max", type=int, default=12,
                    help="Max (G - R) before a pixel is treated as "
                         "GREEN-DOMINANT (foliage) and excluded from "
                         "blossom masks. Bright greenish leaves that "
                         "happen to pass the low-S gate get killed by "
                         "this. Default 12 matches the reference.")
    ap.add_argument("--flower-top-frame-penalty-row", type=int, default=100,
                    help="In rows 0..N (top of frame), apply a stricter "
                         "white-petal rule (S<15, V>200, B-R<=-5). "
                         "Specular leaf glints concentrate at the "
                         "canopy crown where sun strikes waxy cuticles "
                         "at near-grazing angles. Set 0 to disable.")
    # Phenology / DOY adaptation. The sprayer pipeline's
    # flower_detector adapts every threshold to bloom stage; we
    # ported those stage-specific values into hsv_thresholds_for_stage.
    ap.add_argument("--flower-phenology",
                    choices=("off", "auto", "early_bloom", "bloom",
                             "petal_fall", "fruiting"),
                    default="off",
                    help="Adapt HSV / B-R thresholds to apple bloom "
                         "stage. 'off' = use the explicit CLI flags "
                         "as-is (default, preserves back-compat). "
                         "'auto' = parse DOY from each frame's filename "
                         "and pick the stage. Or force a specific "
                         "stage. The reference sprayer pipeline uses "
                         "auto and adapts every day separately -- this "
                         "is the single biggest accuracy lever across "
                         "phenologies.")
    ap.add_argument("--flower-bloom-peak-doy", type=int, default=125,
                    help="Day-of-year for full-bloom peak. Bloom window "
                         "is [peak-10, peak+6]; petal_fall extends 14 "
                         "more days; fruiting after that. Default 125 "
                         "for a 'normal' Ohio apple year. Adjust per "
                         "season / variety.")
    # Anther hole fill. Real blossoms have yellow centers that fail
    # the white-petal S threshold, producing donut-shaped masks. The
    # distance transform on a donut gets multiple peaks (one blossom
    # counted as multiple flowers).
    # Reference-detector additions: texture map, IR positive signal,
    # multiple sky exclusions, confirmed_real gate, two-tier core mask,
    # density score, confidence scaling, negative pixel masks.
    ap.add_argument("--flower-use-texture", action="store_true",
                    help="Compute V_std + IR_std + Sobel gradient maps and "
                         "require pixels to have texture (or valid depth) to "
                         "count as flower. Single biggest discriminator "
                         "between real petals and smooth sky/cloud.")
    ap.add_argument("--flower-texture-threshold", type=float, default=2.5,
                    help="Min V_std / IR_std for a pixel to be 'textured'. "
                         "Default 2.5 from reference detector.")
    ap.add_argument("--flower-edge-threshold", type=float, default=6.0,
                    help="Min Sobel gradient magnitude for 'edge'. "
                         "Default 6.0 from reference detector.")
    ap.add_argument("--flower-ir-positive-min", type=int, default=80,
                    help="If a pixel's IR > this AND it's in fg_flower, "
                         "treat as flower (positive signal). Petals reflect "
                         "NIR strongly. Default 80; set 256 to disable.")
    ap.add_argument("--flower-ir-petal-min", type=int, default=100,
                    help="Tier 3 of REGION mask: pixels with IR > this AND "
                         "V > 50 AND S < 70 are 'IR-confirmed petals'. "
                         "Captures dim shaded blossoms that fail HSV.")
    ap.add_argument("--flower-ir-sky-ceil", type=int, default=60,
                    help="Pixels with IR < this AND no_depth AND no texture "
                         "are smooth-sky. Default 60.")
    ap.add_argument("--flower-confirmed-real", action="store_true",
                    help="Apply the 'confirmed_real = valid_depth OR "
                         "(near_tree AND has_texture)' spatial gate. "
                         "Recovers IR-overexposed flowers near branches.")
    ap.add_argument("--flower-max-depth-cap-mm", type=float, default=0.0,
                    help="Per-mask maximum-depth cap. ALWAYS runs, even "
                         "when --flower-depth-coverage-threshold has "
                         "triggered fallback. For each kept flower mask, "
                         "if at least --flower-max-depth-cap-min-pixels "
                         "pixels of the mask have valid depth AND the "
                         "median of those valid-depth pixels is greater "
                         "than this cap, reject the mask. Catches "
                         "background-row trees with valid depth in the "
                         "5-10 m range that the global fallback would "
                         "otherwise let through. Default 0.0 = disabled. "
                         "Try 3500 (mm) when fallback is on; the cap is "
                         "the actual bound rather than the band-min.")
    ap.add_argument("--flower-max-depth-cap-min-pixels", type=int, default=20,
                    help="Minimum number of valid-depth pixels in the "
                         "mask before --flower-max-depth-cap-mm fires. "
                         "Below this, we don't have enough depth info "
                         "to decide -- give the mask the benefit of the "
                         "doubt. Default 20.")
    ap.add_argument("--flower-depth-coverage-threshold", type=float, default=0.0,
                    help="Per-frame depth-coverage fallback. When the "
                         "fraction of frame pixels with valid canopy "
                         "depth (in [depth-min, depth-max] mm) is below "
                         "this threshold, ALL depth-dependent gates for "
                         "the frame are skipped automatically: "
                         "depth-near-frac, local-depth-std, "
                         "confirmed_real, and the soft-score depth "
                         "component. Catches the edge case where the "
                         "D455 sensor returns sparse depth (frame at "
                         "transitional view, distant trees, sensor edge "
                         "artifacts) and the cascade of depth-based "
                         "rejections produces n=0 even when SAM saw real "
                         "flowers. Default 0.0 = disabled (always use "
                         "depth gates). Try 0.20 = skip when < 20%% of "
                         "frame has valid depth.")
    ap.add_argument("--flower-near-tree-radius-px", type=int, default=15,
                    help="Dilation radius for 'near_tree' band. Pixels "
                         "within this many px of a valid-depth pixel can "
                         "still pass confirmed_real if textured.")
    # Multiple sky exclusion sub-rules (each opt-in independently):
    ap.add_argument("--flower-exclude-sky-smooth", action="store_true",
                    help="Exclude smooth bright sky (~has_texture & no_depth & low_IR).")
    ap.add_argument("--flower-exclude-sky-warm", action="store_true",
                    help="Exclude warm/golden-hour sky (no_depth & V>80 & IR<70).")
    ap.add_argument("--flower-exclude-sky-upper", action="store_true",
                    help="Exclude top-strip sky (top 30%% & no_depth & low_S & V>80).")
    ap.add_argument("--flower-exclude-sky-overcast", action="store_true",
                    help="Exclude overcast cloud (no_depth & S<12 & V 140-200).")
    ap.add_argument("--flower-exclude-sky-grey", action="store_true",
                    help="Exclude bright grey cloud (no_depth & V>200 & S<20 & IR<100).")
    # Two-tier REGION + CORE masks:
    ap.add_argument("--flower-two-tier-mask", action="store_true",
                    help="Compute a REGION mask (broad, for area / bbox) plus "
                         "a CORE mask (strict white/pink only, for confident "
                         "individual flower counting). The CORE coverage "
                         "percentage drives the confidence-scaling "
                         "adjustment to est_flowers per cluster.")
    # Negative pixel masks (per-rule opt-in):
    ap.add_argument("--flower-exclude-bark", action="store_true",
                    help="Exclude bark-brown pixels (H 8-35, S>50, V<100).")
    ap.add_argument("--flower-exclude-dark", action="store_true",
                    help="Exclude dark-material pixels (V<45).")
    ap.add_argument("--flower-exclude-ground-grass", action="store_true",
                    help="Exclude ground-grass pixels (H 25-95, V<100, S>20).")
    # Density score and confidence scaling:
    ap.add_argument("--flower-compute-density-score", action="store_true",
                    help="Compute Estrada-style Gaussian-blur density score "
                         "and write to results.csv as 'flower_density_sum'.")
    ap.add_argument("--flower-confidence-scale", action="store_true",
                    help="Scale a cluster's est_flowers count down when its "
                         "CORE coverage is < 50%% and area > 500 px. Reduces "
                         "over-count on warm-petal/foliage-mixed clusters.")
    ap.add_argument("--flower-fill-anther-holes", action="store_true",
                    help="Flood-fill interior holes in the refined "
                         "petal mask before downstream processing. "
                         "Closes the 5-10 px gaps left by yellow "
                         "anthers / pollen in the center of real "
                         "blossoms (which fail the low-S white rule). "
                         "Improves mask density, stabilizes bbox area "
                         "estimates, and prevents donut-shaped masks "
                         "from being peak-counted as 2-3 flowers. "
                         "Recommended on for flower YOLO labeling.")
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
    ap.add_argument("--prgb-skip-centroid-check", action="store_true",
                    help="Skip the post-refinement check that requires a "
                         "flower mask's centroid to lie inside the PRGB ROI. "
                         "The earlier --prgb-min-overlap fraction check still "
                         "applies. Without this, a flower at the edge of the "
                         "ROI whose petals extend slightly outside (so its "
                         "centroid lands a few pixels past the dilated box) "
                         "gets rejected even though most of the mask is "
                         "inside. Recommended for orchards where lateral "
                         "branches reach beyond the trunk-anchored ROI.")
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
                         "the TOP or BOTTOM frame edge (default 15; D455 stereo "
                         "edge artefacts). Use --flower-edge-margin-sides-px to "
                         "configure the LEFT/RIGHT sides separately.")
    ap.add_argument("--flower-edge-margin-sides-px", type=int, default=-1,
                    help="Override --flower-edge-margin-px for the LEFT and "
                         "RIGHT edges only. Default -1 = inherit from "
                         "--flower-edge-margin-px. Set to 0 to allow flowers "
                         "whose bbox touches the L/R edges through (useful "
                         "for half-trees at frame edges where the partial "
                         "canopy crops blossom bboxes; the original edge_margin "
                         "was meant for D455 stereo artefacts that mostly "
                         "affect top/bottom rows).")
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
    # Leaf-with-sky-behind rejection: reject masks whose leaf-green
    # pixel fraction exceeds a cap. A leaf with bright sky behind it
    # can pass the blossom-color gate (sky pixels register as
    # 'white'); checking the green fraction inside the SAM mask
    # catches that case directly.
    ap.add_argument("--flower-max-mask-green-frac", type=float, default=0.0,
                    help="Reject any flower mask whose fraction of "
                         "leaf-green HSV pixels exceeds this threshold "
                         "(default 0 = disabled). Catches the 'leaf "
                         "with sky behind it' false positive: the bright "
                         "sky inside the mask passes the white-pink "
                         "blossom color check, but the leaf's green "
                         "pixels still dominate. Try 0.30 (30%%).")
    ap.add_argument("--flower-green-hue-min", type=int, default=35,
                    help="Lower hue bound for the leaf-green pixel "
                         "definition (OpenCV 0-179; default 35).")
    ap.add_argument("--flower-green-hue-max", type=int, default=85,
                    help="Upper hue bound for the leaf-green pixel "
                         "definition (OpenCV 0-179; default 85).")
    ap.add_argument("--flower-green-sat-min", type=int, default=40,
                    help="Minimum saturation for a pixel to count as "
                         "leaf-green (0-255; default 40).")
    ap.add_argument("--flower-green-val-min", type=int, default=35,
                    help="Minimum value (V) for a pixel to count as "
                         "leaf-green (0-255; default 35).")
    ap.add_argument("--flower-green-blossom-override-frac", type=float,
                    default=0.20,
                    help="Skip --flower-max-mask-green-frac when the "
                         "mask's BLOSSOM-color fraction (white|pink) "
                         "is at least this much. A real flower "
                         "cluster on a branch picks up surrounding "
                         "leaves so its green frac can exceed the "
                         "cap, but its blossom frac is also high. "
                         "A leaf-with-sky has lots of green and only "
                         "a small white-from-sky fraction -- the "
                         "override doesn't fire and the cap rejects "
                         "it. Default 0.20 (20%%).")
    # Bright-peak density texture discriminator. Apple blossoms have
    # local V-channel maxima at petal scale (each blossom = a bright
    # petal with a darker anther center => one peak); leaves are
    # mostly uniform with at most 1-2 specular highlights. Counting
    # peaks per 1000 px of mask discriminates flower clusters from
    # leaf masks even when both have similar overall brightness.
    ap.add_argument("--flower-min-peaks-per-1000px", type=float,
                    default=0.0,
                    help="Reject any flower mask with fewer than this "
                         "many local brightness peaks per 1000 px of "
                         "mask area. Apple blossoms register as bright "
                         "V-channel local maxima at petal scale; "
                         "leaves do not. Default 0.0 = disabled. Try "
                         "3.0 (= ~3 peaks per 1000 px, or one per "
                         "single ~330 px blossom). Skipped on small "
                         "masks below --flower-peak-min-area-px.")
    ap.add_argument("--flower-peak-min-distance-px", type=int,
                    default=5,
                    help="Minimum spacing (in px) between counted "
                         "brightness peaks. Effective neighborhood = "
                         "(2*N+1)x(2*N+1). Default 5 -> 11x11 window. "
                         "Set to roughly half a typical blossom "
                         "diameter so two peaks inside one blossom "
                         "collapse to one.")
    ap.add_argument("--flower-peak-threshold-abs", type=int,
                    default=80,
                    help="Minimum V (HSV brightness, 0-255) for a "
                         "pixel to count as a candidate peak. Default "
                         "80 admits dimly-lit blossoms while excluding "
                         "shadow-noise local maxima.")
    ap.add_argument("--flower-peak-min-area-px", type=int,
                    default=60,
                    help="Skip the --flower-min-peaks-per-1000px gate "
                         "for masks smaller than this (in px). Tiny "
                         "masks naturally have 0 or 1 peaks due to "
                         "size alone; the gate would be unfair. "
                         "Default 60 px.")
    ap.add_argument("--flower-peak-min-distance-px2", type=int,
                    default=0,
                    help="Optional SECOND peak-detection scale, used "
                         "in addition to --flower-peak-min-distance-px. "
                         "Default 0 = single scale. Try 9 (= 19x19 "
                         "window with sigma=2.25) to also catch the "
                         "cluster-centroid peak of multi-blossom "
                         "clusters whose individual blossoms are "
                         "smoothed away at the narrow scale. Two "
                         "scales help with viewpoint robustness: "
                         "face-on blossoms peak strongly at the "
                         "narrow scale, edge-on or backlit clusters "
                         "often peak more clearly at the wider "
                         "scale. The peak counts at each scale are "
                         "OR'd, so a real flower can register at "
                         "either scale; leaves register at neither.")
    ap.add_argument("--flower-peak-prominence-min", type=float,
                    default=5.0,
                    help="Minimum prominence for a candidate peak: "
                         "V at the peak must be at least this many "
                         "units above the LOCAL MEAN of its window. "
                         "Suppresses uniform-bright plateaus (sky, "
                         "smooth leaves) where every pixel ties for "
                         "the window max. Default 5.0; lower for "
                         "dim conditions, higher to require sharper "
                         "peaks.")
    # Anther-hole density: a POSITIVE flower discriminator. An
    # anther hole is a small dark pixel cluster fully enclosed by
    # bright petal pixels. Leaves and leaf-with-sky masks lack this
    # pattern -- their dark pixels are adjacent to dark or to bright
    # sky, never enclosed. Implementation: global flood-fill the
    # bright (V >= petal_v_min) mask from a frame corner; pixels
    # that remain unreached and dark are anther candidates. Their
    # CCs (filtered by size) are the holes we count per mask.
    ap.add_argument("--flower-min-anther-holes-per-1000px", type=float,
                    default=0.0,
                    help="Reject any flower mask with fewer than this "
                         "many anther-hole CCs per 1000 px of mask "
                         "area. Apple blossoms have small dark anther "
                         "dots fully enclosed by bright petals; leaves "
                         "and sky-with-leaf don't. Default 0.0 = "
                         "disabled. Try 2.0 (= ~2 anthers per typical "
                         "1000 px cluster). Skipped on small masks "
                         "below --flower-anther-min-area-px.")
    ap.add_argument("--flower-anther-petal-v-min", type=int,
                    default=100,
                    help="Pixels with V (HSV brightness) at or above "
                         "this threshold are 'petal-bright' for the "
                         "anther-hole detector. Default 100 admits "
                         "dim petals while rejecting most foliage.")
    ap.add_argument("--flower-anther-hole-min-area-px", type=int,
                    default=2,
                    help="Minimum size (in px) for an interior dark "
                         "CC to count as an anther hole. Filters out "
                         "single-pixel speckle. Default 2.")
    ap.add_argument("--flower-anther-hole-max-area-px", type=int,
                    default=60,
                    help="Maximum size (in px) for an interior dark "
                         "CC to count as an anther hole. Filters out "
                         "branch gaps and large dark structures that "
                         "happen to be fully petal-enclosed. Default "
                         "60.")
    ap.add_argument("--flower-anther-min-area-px", type=int,
                    default=100,
                    help="Skip the --flower-min-anther-holes-per-1000px "
                         "gate for masks smaller than this (in px). "
                         "Anther dots need a few hundred bright petal "
                         "pixels around them to be detected at all; "
                         "small masks shouldn't be judged. Default 100 "
                         "px.")
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
    ap.add_argument("--split-seed-dilate-px", type=int, default=5,
                    help="Pixels to dilate each watershed marker seed before "
                         "running watershed. Mirrors the reference sprayer "
                         "pipeline's cv2.dilate(seed, MORPH_ELLIPSE 5x5). "
                         "Larger values give cleaner boundaries between "
                         "adjacent peaks; 0 disables. Default 5.")
    ap.add_argument("--split-area-cap", action="store_true",
                    help="Cap the number of sub-masks from a single cluster "
                         "to ceil(parent_area / area-per-flower-px). Watershed "
                         "occasionally over-splits a single blossom into 2-3 "
                         "sub-peaks (petal ridge irregularities, partial "
                         "anther holes). With this flag, only the LARGEST "
                         "sub-masks are kept up to the area cap.")
    # Ground-row rejection (filter, not splitter). Catches dandelions /
    # grass wildflowers whose detections sit in the bottom of the frame.
    ap.add_argument("--flower-max-ground-row", type=int, default=0,
                    help="Reject flower clusters whose centroid row is "
                         "below this Y (default 0 = disabled). When ON, "
                         "a cluster below this row is dropped UNLESS its "
                         "strict-color core covers at least "
                         "--flower-min-confirmed-pct-ground percent of "
                         "the cluster area (real low blossoms get "
                         "confirmed; grass wildflowers fail). The "
                         "sprayer pipeline reference uses 400.")
    ap.add_argument("--flower-min-confirmed-pct-ground", type=float,
                    default=10.0,
                    help="Strict-core percentage required for a below-"
                         "ground-row cluster to survive. Default 10%%.")
    args = ap.parse_args()

    sample = None if args.sample_per_session == 0 else args.sample_per_session
    prompts = list(args.prompts)
    # Auto-add the trunk prompt when --canopy-track-method=trunk is on
    # (we need SAM 3 to detect trunks per frame). Also auto-track the
    # trunk prompt so its bboxes get IoU-followed across frames if the
    # user wants per-tree analytics later. Avoid duplication if user
    # already passed it.
    if (args.track_canopy
            and args.canopy_track_method == "trunk"
            and args.canopy_trunk_prompt not in prompts):
        prompts.append(args.canopy_trunk_prompt)
        print(
            f"[init] --canopy-track-method=trunk: auto-added "
            f"prompt {args.canopy_trunk_prompt!r}",
            file=sys.stderr,
        )
    # Auto-add the SAM canopy prompt(s). SAM 3 will be queried with
    # the primary prompt + any --canopy-sam-multi-prompts each frame,
    # and the union of their (filtered) masks becomes the canopy.
    _canopy_sam_prompt_set: list[str] = []
    if args.canopy_sam_prompt:
        _canopy_sam_prompt_set.append(args.canopy_sam_prompt)
    if args.canopy_sam_multi_prompts:
        for p in args.canopy_sam_multi_prompts:
            if p and p not in _canopy_sam_prompt_set:
                _canopy_sam_prompt_set.append(p)
    for p in _canopy_sam_prompt_set:
        if p not in prompts:
            prompts.append(p)
            print(
                f"[init] auto-added canopy SAM prompt {p!r}",
                file=sys.stderr,
            )
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
    canopy_masks_dir = out_dir / "canopy_masks"
    canopy_overlays_dir = out_dir / "canopy_overlays"
    depth_overlays_dir = out_dir / "depth_fg_overlays"
    if args.save_masks:
        masks_dir.mkdir(exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(exist_ok=True)
    if args.save_canopy_masks:
        canopy_masks_dir.mkdir(exist_ok=True)
    if args.save_canopy_overlay:
        canopy_overlays_dir.mkdir(exist_ok=True)
    if args.save_depth_fg_overlay:
        depth_overlays_dir.mkdir(exist_ok=True)

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
                  "flower_density_sum",
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
    # Per-session canopy tracker. Reinstantiated per session so
    # tree IDs are scoped to one orchard-pass.
    current_canopy_tracker: CanopyTracker | None = None
    # Per-(session, tree_id) summary for the new trees CSV.
    tree_summaries: list[dict] = []
    # Per-frame, per-tree row for trees_per_frame CSV.
    tree_per_frame_rows: list[dict] = []
    # tracks_detail entries get a "tree_id" column populated
    # when canopy tracking is on; collect alongside flower
    # tracks so the per-track CSV can show which tree each
    # blossom track lives on.
    flower_track_tree_id: dict[tuple, int] = {}  # (session_key, prompt, track_id) -> tree_id

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
            require_info_modality=args.require_info_modality,
            sample_mode=args.sample_mode,
            sample_stride=args.sample_stride,
        ):
            if args.max_images is not None and total_imgs >= args.max_images:
                break

            # Detect session change → flush trackers for the previous session
            # and start fresh ones for the new session.
            session_key = (day, category, session)
            if args.track and session_key != current_session_key:
                if current_session_key is not None:
                    flush_session_trackers(current_session_key, current_trackers)
                    # Flush the CANOPY tracker too: aggregate one
                    # row per (session, tree_id) into tree_summaries.
                    if (args.track_canopy
                            and current_canopy_tracker is not None):
                        d_prev, c_prev, s_prev = current_session_key
                        for trk in current_canopy_tracker.summary():
                            tree_summaries.append({
                                "day": d_prev, "category": c_prev,
                                "session": s_prev,
                                "tree_id": trk["tree_id"],
                                "first_frame": trk["first_frame"],
                                "last_frame": trk["last_frame"],
                                "n_frames": trk["n_frames"],
                                "max_area_px": trk.get("max_area", 0),
                            })
                current_trackers = {p: IoUTracker(args.track_iou, args.track_max_age)
                                    for p in tracked_prompts}
                current_canopy_tracker = (
                    CanopyTracker(
                        iou_threshold=args.canopy_track_iou,
                        max_age=args.canopy_track_max_age,
                    ) if args.track_canopy else None
                )
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

                # SAM 3-based canopy. Iterate over the configured
                # canopy prompts (primary + multi-prompts), build a
                # filtered canopy from each, and union them. Each
                # prompt has different strengths: 'apple tree'
                # captures the prototypical case; 'tree branches'
                # catches sparse / dim trees; 'fruit tree' helps
                # with blossom-dominant trees; 'tree canopy' helps
                # with full-canopy trees. The union recovers cases
                # where any single prompt misses.
                _sct_canopy: np.ndarray | None = None
                if (args.tree_mask
                        and isinstance(infer, dict)
                        and _canopy_sam_prompt_set):
                    _rgb_for_sct = np.asarray(img)
                    _u = np.zeros(
                        (img.height, img.width), dtype=bool,
                    )
                    for _p in _canopy_sam_prompt_set:
                        if _p not in infer:
                            continue
                        _pi = infer[_p]
                        _u_p = build_canopy_from_sam_trees(
                            _pi.get("masks"),
                            _pi.get("scores"),
                            depth_mm,
                            (img.height, img.width),
                            min_score=args.canopy_sam_min_score,
                            depth_min_mm=args.depth_min_mm,
                            depth_max_mm=args.depth_max_mm,
                            min_pixels=args.canopy_sam_min_pixels,
                            position_min_lower_frac=(
                                args.canopy_sam_min_lower_frac
                            ),
                            require_depth_check=(
                                args.canopy_sam_depth_check
                            ),
                            min_valid_depth_frac=(
                                args.canopy_sam_min_valid_depth_frac
                            ),
                            rgb_arr=_rgb_for_sct,
                            rgb_fallback_min_vegetation_frac=(
                                args.canopy_sam_rgb_fallback_min_veg_frac
                            ),
                            max_top_row=args.canopy_sam_max_top_row,
                            min_aspect_ratio=(
                                args.canopy_sam_min_aspect_ratio
                            ),
                            max_depth_row_corr=(
                                args.canopy_sam_max_depth_row_corr
                            ),
                        )
                        if _u_p.any():
                            _u |= _u_p
                    if _u.any():
                        _sct_canopy = _u

                # Decide whether to skip the heuristic. When
                # --canopy-sam-only is on AND SAM found anything,
                # we trust SAM and skip the heuristic entirely
                # (avoids over-inclusion: heuristic adds grass /
                # ground noise where SAM has a clean tree mask).
                # Otherwise the heuristic runs and is unioned in.
                _skip_heuristic = (
                    bool(args.canopy_sam_only)
                    and _sct_canopy is not None
                )

                if (args.tree_mask and depth_mm is not None
                        and not _skip_heuristic):
                    if args.use_build_tree_mask:
                        # Robust canopy detection from tree_mask.py:
                        # depth band + HSV sky exclusion + ROI-anchored
                        # region growing + forward-branch recovery.
                        # Handles partial-frame trees much better than
                        # the simple depth-only compute_canopy_mask.
                        try:
                            from tree_mask import build_tree_mask as _btm
                            _rgb_for_canopy = np.asarray(img)
                            # Use the sprayer pipeline's DEFAULT center-
                            # column ROI (cols 285-354) for primary
                            # anchoring. This is critical: the algorithm
                            # computes center_depth from anchored-region
                            # pixels, then enforces a +/-500 mm band
                            # around that depth. Anchoring on the full
                            # frame width pulled in grass + sky pixels,
                            # which then biased center_depth and let
                            # the band include grass and dilation-leak
                            # into sky. With the tight default anchor
                            # the sprayer pipeline's logic locks onto
                            # the actual target tree's depth.
                            #
                            # Trees at the L/R frame edges that aren't
                            # in the center ROI are recovered by the
                            # SEPARATE tree-shape augmentation below.
                            _h, _w = depth_mm.shape
                            _btm_u8 = _btm(
                                depth_mm.astype(np.uint16),
                                rgb=_rgb_for_canopy,
                            )
                            canopy_mask_img = _btm_u8.astype(bool)
                        except Exception as _btm_err:
                            print(
                                f"[warn] build_tree_mask failed "
                                f"({_btm_err!r}); falling back to "
                                f"compute_canopy_mask",
                                file=sys.stderr,
                            )
                            canopy_mask_img = compute_canopy_mask(
                                depth_mm,
                                min_mm=args.depth_min_mm,
                                max_mm=args.depth_max_mm,
                                min_cc_area_px=args.canopy_min_cc_area_px,
                                max_row_width_frac=args.canopy_max_row_width_frac,
                            )
                    else:
                        canopy_mask_img = compute_canopy_mask(
                            depth_mm,
                            min_mm=args.depth_min_mm, max_mm=args.depth_max_mm,
                            min_cc_area_px=args.canopy_min_cc_area_px,
                            max_row_width_frac=args.canopy_max_row_width_frac,
                        )
                    # Optional dilation of the canopy mask. Catches
                    # flowers on lateral/forward branches whose mask
                    # extends slightly past what build_tree_mask
                    # captured as canopy. Without this, those flowers
                    # fail --tree-mask-min-overlap because the canopy
                    # CC filter dropped the thin branches.
                    if (canopy_mask_img is not None
                            and args.tree_mask_dilate_px > 0):
                        try:
                            import cv2 as _cv2_dil
                            _k = max(
                                1,
                                2 * int(args.tree_mask_dilate_px) + 1,
                            )
                            _kern = _cv2_dil.getStructuringElement(
                                _cv2_dil.MORPH_ELLIPSE, (_k, _k),
                            )
                            canopy_mask_img = _cv2_dil.dilate(
                                canopy_mask_img.astype(np.uint8),
                                _kern, iterations=1,
                            ).astype(bool)
                        except Exception as _dil_err:
                            print(
                                f"[warn] canopy mask dilation failed "
                                f"({_dil_err!r}); using undilated mask",
                                file=sys.stderr,
                            )

                    # Edge-tree augmentation. build_tree_mask uses a
                    # global center_depth band that excludes half-
                    # trees at the frame edges when multiple trees
                    # at different depths are in view. This wrapper
                    # adds foreground-depth components touching the
                    # left/right edges back to the canopy mask.
                    if (args.canopy_include_edge_trees
                            and canopy_mask_img is not None
                            and depth_mm is not None):
                        try:
                            _rgb_aug = np.asarray(img)
                            canopy_mask_img = augment_canopy_with_edge_trees(
                                canopy_mask_img,
                                depth_mm,
                                _rgb_aug,
                                depth_min_mm=args.depth_min_mm,
                                depth_max_mm=(
                                    args.canopy_edge_tree_max_depth_mm
                                ),
                                min_area_px=args.canopy_edge_tree_min_area_px,
                                min_height_px=(
                                    args.canopy_edge_tree_min_height_px
                                ),
                                max_top_row=(
                                    args.canopy_edge_tree_max_top_row
                                ),
                                min_aspect_ratio=(
                                    args.canopy_edge_tree_min_aspect_ratio
                                ),
                                max_depth_std_mm=(
                                    args.canopy_edge_tree_max_depth_std_mm
                                ),
                                min_green_frac=(
                                    args.canopy_edge_tree_min_green_frac
                                ),
                            )
                        except Exception as _edge_err:
                            print(
                                f"[warn] edge-tree canopy augmentation "
                                f"failed ({_edge_err!r}); using "
                                f"unaugmented mask",
                                file=sys.stderr,
                            )

                    # Post-processing tree-shape filter. Runs on the
                    # FINAL canopy mask (after build_tree_mask, after
                    # external dilation, after edge augmentation) to
                    # remove non-tree-shaped CCs. Catches grass bands
                    # that build_tree_mask pulled in via central
                    # anchoring (because the orchard floor extended
                    # up into cols 285-354) and any wide foreground-
                    # depth CC that snuck through. The aspect-ratio
                    # check is the workhorse here -- a tree is
                    # tall+narrow, grass is wide+short.
                    if (args.canopy_filter_by_tree_shape
                            and canopy_mask_img is not None):
                        try:
                            _rgb_filt = np.asarray(img)
                            canopy_mask_img = filter_canopy_by_tree_shape(
                                canopy_mask_img,
                                depth_mm,
                                _rgb_filt,
                                min_aspect_ratio=(
                                    args.canopy_filter_min_aspect_ratio
                                ),
                                max_depth_std_mm=(
                                    args.canopy_filter_max_depth_std_mm
                                ),
                                min_green_frac=(
                                    args.canopy_filter_min_green_frac
                                ),
                                min_area_px=(
                                    args.canopy_filter_min_area_px
                                ),
                            )
                        except Exception as _filt_err:
                            print(
                                f"[warn] tree-shape canopy filter "
                                f"failed ({_filt_err!r}); using "
                                f"unfiltered mask",
                                file=sys.stderr,
                            )

                # UNION SAM canopy with the heuristic canopy. SAM
                # gives pixel-accurate masks for trees it detects;
                # the heuristic catches half-trees, blossom-dominant
                # trees, and partially-segmented tree tops that SAM
                # missed. Either alone has gaps; the union covers
                # both.
                if _sct_canopy is not None:
                    if canopy_mask_img is None:
                        canopy_mask_img = _sct_canopy
                    else:
                        canopy_mask_img = (
                            canopy_mask_img.astype(bool)
                            | _sct_canopy.astype(bool)
                        )

                # Post-process canopy refinement (upward expansion
                # for tree tops + sky exclusion + morphological
                # closing for adjacent CC merging).
                if (args.canopy_refine
                        and canopy_mask_img is not None
                        and canopy_mask_img.any()):
                    try:
                        _rgb_ref = np.asarray(img)
                        canopy_mask_img = refine_canopy_mask(
                            canopy_mask_img,
                            depth_mm,
                            _rgb_ref,
                            upward_dilate_px=(
                                args.canopy_refine_upward_dilate_px
                            ),
                            close_px=args.canopy_refine_close_px,
                            depth_min_mm=args.depth_min_mm,
                            depth_max_mm=args.depth_max_mm,
                            exclude_sky=True,
                        )
                    except Exception as _ref_err:
                        print(
                            f"[warn] canopy refinement failed "
                            f"({_ref_err!r}); using unrefined mask",
                            file=sys.stderr,
                        )

                # PAINTED-STAKE EXCLUSION. Remove pixels that look
                # like a painted-green metal/wood stake (saturated
                # green at low brightness) from the canopy mask.
                # Stakes are visually distinct from leaf green
                # (which has higher V and lower S in the same hue
                # band). User can tune the HSV bounds via the
                # --canopy-stake-* flags.
                if (args.canopy_exclude_painted_stakes
                        and canopy_mask_img is not None
                        and canopy_mask_img.any()):
                    try:
                        import cv2 as _cv2_st
                        _rgb_st = np.asarray(img)
                        _hsv_st = _cv2_st.cvtColor(
                            _rgb_st.astype(np.uint8),
                            _cv2_st.COLOR_RGB2HSV,
                        )
                        _Hs = _hsv_st[:, :, 0]
                        _Ss = _hsv_st[:, :, 1]
                        _Vs = _hsv_st[:, :, 2]
                        _stake_pix = (
                            (_Hs >= int(args.canopy_stake_hue_min))
                            & (_Hs <= int(args.canopy_stake_hue_max))
                            & (_Ss >= int(args.canopy_stake_sat_min))
                            & (_Vs <= int(args.canopy_stake_val_max))
                        )
                        canopy_mask_img = (
                            canopy_mask_img.astype(bool) & ~_stake_pix
                        )
                    except Exception as _stake_err:
                        print(
                            f"[warn] painted-stake exclusion "
                            f"failed ({_stake_err!r})",
                            file=sys.stderr,
                        )

                # FILL SMALL CANOPY HOLES. After sky / bg-depth /
                # stake exclusion the canopy is often a
                # constellation of leafy blobs separated by tiny
                # sky-gaps that were correctly removed but which
                # belong to the tree's silhouette visually.
                # Filling small holes restores connectivity through
                # branches without re-adding large excluded regions
                # (which stay as holes because they exceed the size
                # cap).
                if (args.canopy_fill_small_holes
                        and canopy_mask_img is not None
                        and canopy_mask_img.any()):
                    try:
                        canopy_mask_img = fill_small_canopy_holes(
                            canopy_mask_img,
                            max_hole_area_px=(
                                args.canopy_max_hole_area_px
                            ),
                        )
                    except Exception as _fill_err:
                        print(
                            f"[warn] canopy hole-fill failed "
                            f"({_fill_err!r})",
                            file=sys.stderr,
                        )

                # FINAL tree-shape filter: catches CCs that became
                # wide+short AFTER closing bridged smaller CCs. The
                # earlier filter (inside the heuristic block) only
                # sees pre-union, pre-closing CCs. Closing can
                # bridge multiple foreground patches into a single
                # wide grass band; this final pass removes those.
                if (args.canopy_filter_by_tree_shape
                        and canopy_mask_img is not None
                        and canopy_mask_img.any()):
                    try:
                        _rgb_filt2 = np.asarray(img)
                        canopy_mask_img = filter_canopy_by_tree_shape(
                            canopy_mask_img,
                            depth_mm,
                            _rgb_filt2,
                            min_aspect_ratio=(
                                args.canopy_filter_min_aspect_ratio
                            ),
                            max_depth_std_mm=(
                                args.canopy_filter_max_depth_std_mm
                            ),
                            min_green_frac=(
                                args.canopy_filter_min_green_frac
                            ),
                            min_area_px=(
                                args.canopy_filter_min_area_px
                            ),
                            max_top_row=(
                                args.canopy_filter_max_top_row
                            ),
                        )
                    except Exception as _filt2_err:
                        print(
                            f"[warn] post-refine tree-shape filter "
                            f"failed ({_filt2_err!r}); using "
                            f"unfiltered mask",
                            file=sys.stderr,
                        )

                # Per-frame canopy tracking. Either:
                #   cc method:    extract connected components from
                #                 the canopy mask (cheap; fails on
                #                 physically-touching tree canopies).
                #   trunk method: SAM 3 detected trunks this frame
                #                 are used as Voronoi anchors to
                #                 partition the canopy. Distinguishes
                #                 overlapping trees because trunks
                #                 stay spatially distinct.
                # In both modes the session-scoped CanopyTracker
                # follows each region across frames. The tracker is
                # fed BBOXES from either the CC (cc mode) or the
                # TRUNK (trunk mode) so trunk-mode tracking is
                # extra-stable (trunks are slim and shift little
                # vs canopy bboxes that flop around).
                frame_canopy_components: list[dict] = []
                frame_tree_ids: list[int] = []
                if (args.track_canopy and canopy_mask_img is not None):
                    if current_canopy_tracker is None:
                        current_canopy_tracker = CanopyTracker(
                            iou_threshold=args.canopy_track_iou,
                            max_age=args.canopy_track_max_age,
                        )
                    if args.canopy_track_method == "trunk":
                        # Pull confident trunks from this frame's
                        # SAM 3 results.
                        _trunk_key = args.canopy_trunk_prompt
                        _trunk_infer = infer.get(_trunk_key) if isinstance(
                            infer, dict
                        ) else None
                        _trunk_boxes_all = (
                            _trunk_infer.get("boxes") if _trunk_infer else None
                        )
                        _trunk_scores_all = (
                            _trunk_infer.get("scores") if _trunk_infer else None
                        )
                        _trunk_masks_all = (
                            _trunk_infer.get("masks") if _trunk_infer else None
                        )
                        _kept_trunk_boxes: list = []
                        _kept_trunk_scores: list = []
                        _kept_trunk_masks: list = []
                        if (_trunk_boxes_all is not None
                                and len(_trunk_boxes_all)):
                            _scores = (
                                _trunk_scores_all
                                if _trunk_scores_all is not None
                                else [1.0] * len(_trunk_boxes_all)
                            )
                            for ti, (tb, ts) in enumerate(
                                zip(_trunk_boxes_all, _scores)
                            ):
                                if float(ts) >= args.canopy_trunk_min_score:
                                    _kept_trunk_boxes.append(
                                        [float(x) for x in tb]
                                    )
                                    _kept_trunk_scores.append(float(ts))
                                    if (_trunk_masks_all is not None
                                            and ti < len(_trunk_masks_all)):
                                        _kept_trunk_masks.append(
                                            _trunk_masks_all[ti]
                                        )
                            # Filter out painted-green support stakes
                            # before they get used as canopy anchors.
                            # Color stats run on the SAM mask (just
                            # trunk pixels) NOT the bbox, so leaves
                            # in front of a real trunk don't mis-flag
                            # it as a stake.
                            if (args.canopy_trunk_reject_green_stakes
                                    and _kept_trunk_boxes):
                                _rgb_arr_for_stake = np.asarray(img)
                                _masks_for_filter = (
                                    np.stack(_kept_trunk_masks, axis=0)
                                    if (_kept_trunk_masks
                                        and len(_kept_trunk_masks)
                                            == len(_kept_trunk_boxes))
                                    else None
                                )
                                _kept_trunk_boxes, _kept_trunk_scores = (
                                    filter_painted_stake_trunks(
                                        _kept_trunk_boxes,
                                        _kept_trunk_scores,
                                        _rgb_arr_for_stake,
                                        trunk_masks=_masks_for_filter,
                                        max_green_dominant_pct=args.canopy_trunk_max_green_pct,
                                        green_dominance_threshold=args.canopy_trunk_green_threshold,
                                        min_brown_pct=args.canopy_trunk_min_brown_pct,
                                    )
                                )
                                # ALSO drop the matching masks so the
                                # depth-filter sees the right ones.
                                if _masks_for_filter is not None:
                                    _kept_trunk_masks = [
                                        m for m, b in zip(
                                            _kept_trunk_masks,
                                            list(_kept_trunk_boxes)
                                            + [None] * (
                                                len(_kept_trunk_masks)
                                                - len(_kept_trunk_boxes)
                                            ),
                                        )
                                        if b is not None
                                    ][:len(_kept_trunk_boxes)]
                            # Reject background-row trunks by depth.
                            # A real foreground trunk sits in the
                            # canopy depth band (~1-3 m). A back-row
                            # trunk at 5+ m would otherwise become a
                            # Voronoi anchor and capture the
                            # background canopy region, leaking
                            # background flowers into the count.
                            if (args.canopy_trunk_max_depth_mm > 0
                                    and depth_mm is not None
                                    and _kept_trunk_boxes):
                                _masks_for_dfilter = (
                                    np.stack(_kept_trunk_masks, axis=0)
                                    if (_kept_trunk_masks
                                        and len(_kept_trunk_masks)
                                            == len(_kept_trunk_boxes))
                                    else None
                                )
                                _kept_trunk_boxes, _kept_trunk_scores = (
                                    filter_far_trunks(
                                        _kept_trunk_boxes,
                                        _kept_trunk_scores,
                                        depth_mm,
                                        trunk_masks=_masks_for_dfilter,
                                        max_depth_mm=args.canopy_trunk_max_depth_mm,
                                        min_valid_pixels=args.canopy_trunk_depth_min_pixels,
                                    )
                                )
                        if _kept_trunk_boxes:
                            # Trunk-based canopy refinements. Run
                            # BEFORE partition so the partition
                            # operates on the corrected mask.
                            #   1. Union trunk masks into canopy:
                            #      bridges two CCs that share a
                            #      trunk (one tree shown as two
                            #      leafy blobs because the visible
                            #      trunk between them was filtered
                            #      out). The trunk mask is narrow
                            #      so it doesn't add ground.
                            if (args.canopy_add_trunk_masks
                                    and _kept_trunk_masks
                                    and canopy_mask_img is not None):
                                _cm_b = canopy_mask_img.astype(bool)
                                _h_cm, _w_cm = _cm_b.shape
                                _v_ext = int(
                                    args.canopy_trunk_vertical_extension_px
                                )
                                for _tm, _tb in zip(
                                    _kept_trunk_masks, _kept_trunk_boxes,
                                ):
                                    _tmb = np.asarray(_tm).astype(bool)
                                    if _tmb.ndim == 3:
                                        _tmb = _tmb.any(axis=0)
                                    if _tmb.shape == _cm_b.shape:
                                        _cm_b = _cm_b | _tmb
                                    # Vertical extension: draw a 5px-
                                    # wide line from trunk top going
                                    # up by _v_ext rows. Bridges
                                    # upper canopy regions above the
                                    # visible trunk's segmented top.
                                    if _v_ext > 0:
                                        _x1 = float(_tb[0])
                                        _y1 = float(_tb[1])
                                        _x2 = float(_tb[2])
                                        _cx = int(round((_x1 + _x2) / 2))
                                        _y_top = int(round(_y1))
                                        _line_top = max(0, _y_top - _v_ext)
                                        _col_lo = max(0, _cx - 2)
                                        _col_hi = min(_w_cm, _cx + 3)
                                        if (_line_top < _y_top
                                                and _col_lo < _col_hi):
                                            _cm_b[_line_top:_y_top,
                                                  _col_lo:_col_hi] = True
                                canopy_mask_img = _cm_b
                            #   2. Crop canopy CCs below trunk
                            #      bottom: removes GROUND that the
                            #      SAM mask extended into below the
                            #      tree. Buffer accounts for low-
                            #      hanging branches.
                            if (args.canopy_crop_below_trunk
                                    and canopy_mask_img is not None):
                                try:
                                    canopy_mask_img = (
                                        crop_canopy_below_trunks(
                                            canopy_mask_img,
                                            _kept_trunk_boxes,
                                            buffer_below_px=(
                                                args.canopy_crop_below_trunk_buffer_px
                                            ),
                                        )
                                    )
                                except Exception as _crop_err:
                                    print(
                                        f"[warn] crop-below-trunk "
                                        f"failed ({_crop_err!r}); "
                                        f"using uncropped mask",
                                        file=sys.stderr,
                                    )
                            frame_canopy_components, _part_labels = (
                                partition_canopy_by_trunks(
                                    canopy_mask_img, _kept_trunk_boxes,
                                    min_trunk_area_px=args.canopy_track_min_cc_area,
                                )
                            )
                            # Track on TRUNK bboxes (most stable
                            # cross-frame anchor; canopy partition
                            # bbox can flop around as branches sway).
                            _track_input = [
                                {"bbox": c["trunk_box"], "area": c["area"]}
                                for c in frame_canopy_components
                            ]
                            frame_tree_ids = current_canopy_tracker.step(
                                _track_input,
                            )
                        else:
                            # No confident trunks this frame --
                            # fall back to CC partition for tree
                            # tracking continuity.
                            frame_canopy_components = extract_canopy_components(
                                canopy_mask_img,
                                min_area_px=args.canopy_track_min_cc_area,
                            )
                            frame_tree_ids = current_canopy_tracker.step(
                                frame_canopy_components,
                            )
                    else:
                        frame_canopy_components = extract_canopy_components(
                            canopy_mask_img,
                            min_area_px=args.canopy_track_min_cc_area,
                        )
                        frame_tree_ids = current_canopy_tracker.step(
                            frame_canopy_components,
                        )
                    # Per-frame log row for trees_per_frame.csv.
                    for cc, tid in zip(
                        frame_canopy_components, frame_tree_ids,
                    ):
                        bx0, by0, bx1, by1 = cc["bbox"]
                        tree_per_frame_rows.append({
                            "day": day, "category": category,
                            "session": session,
                            "image": str(img_path),
                            "tree_id": int(tid),
                            "bbox_x0": bx0, "bbox_y0": by0,
                            "bbox_x1": bx1, "bbox_y1": by1,
                            "area_px": int(cc["area"]),
                        })

                # Save the per-frame canopy mask for downstream use
                # (Label Studio polygon overlays, per-tree analytics,
                # debugging). One .npz per frame at
                # <out>/canopy_masks/<rel_path>.npz.
                if (args.save_canopy_masks
                        and canopy_mask_img is not None):
                    try:
                        rel_canopy = (
                            img_path.relative_to(args.root)
                            .with_suffix(".npz")
                        )
                        cp_path = canopy_masks_dir / rel_canopy
                        cp_path.parent.mkdir(parents=True, exist_ok=True)
                        # Build the partition labels image if we
                        # tracked trunks; otherwise we can rebuild
                        # it from the canopy CC labels at view time.
                        _save_extras: dict = {
                            "canopy": canopy_mask_img.astype(bool),
                        }
                        if frame_canopy_components:
                            _labels_img = frame_canopy_components[0].get(
                                "labels"
                            )
                            if _labels_img is not None:
                                _save_extras["partition_labels"] = (
                                    np.asarray(_labels_img, dtype=np.int32)
                                )
                            _trunk_bboxes_save = []
                            for cc in frame_canopy_components:
                                tb = cc.get("trunk_box")
                                if tb is not None:
                                    _trunk_bboxes_save.append(
                                        list(tb) + [int(cc.get("label_id", 0))]
                                    )
                            if _trunk_bboxes_save:
                                _save_extras["trunk_bboxes_with_label"] = (
                                    np.asarray(
                                        _trunk_bboxes_save, dtype=np.int32,
                                    )
                                )
                            if frame_tree_ids:
                                _save_extras["tree_ids"] = np.asarray(
                                    frame_tree_ids, dtype=np.int64,
                                )
                        np.savez_compressed(cp_path, **_save_extras)
                    except Exception as _cp_err:
                        print(
                            f"[warn] could not save canopy mask for "
                            f"{img_path.name}: {_cp_err!r}",
                            file=sys.stderr,
                        )

                # Save a HUMAN-VIEWABLE canopy overlay JPG. RGB +
                # red translucent fill on canopy pixels + per-tree
                # partition tints (when --track-canopy is on) + a
                # bright outline + trunk bboxes. Lets you visually
                # diagnose why a frame failed (e.g., edge tree
                # missing from the mask, grass added as canopy,
                # depth band exclusion, etc.) without having to
                # load and decode the .npz.
                if (args.save_canopy_overlay
                        and canopy_mask_img is not None):
                    try:
                        import cv2 as _cv2_co
                        rel_co = (
                            img_path.relative_to(args.root)
                            .with_suffix(".jpg")
                        )
                        co_path = canopy_overlays_dir / rel_co
                        co_path.parent.mkdir(parents=True, exist_ok=True)
                        rgb_co = np.asarray(img).copy()
                        if rgb_co.ndim == 2:
                            rgb_co = _cv2_co.cvtColor(
                                rgb_co, _cv2_co.COLOR_GRAY2RGB,
                            )
                        cm_bool = canopy_mask_img.astype(bool)
                        # Base layer: red tint inside canopy.
                        tint = np.zeros_like(rgb_co)
                        tint[cm_bool] = (255, 60, 60)
                        rgb_co = _cv2_co.addWeighted(
                            rgb_co, 0.7, tint, 0.3, 0,
                        )
                        # Per-tree partition tints (different hue
                        # per tree id) when canopy tracking is on.
                        if frame_canopy_components and frame_tree_ids:
                            _labels_img = (
                                frame_canopy_components[0].get("labels")
                            )
                            if _labels_img is not None:
                                # Color cycle for trees.
                                _tree_colors = [
                                    (60, 220, 60), (60, 60, 220),
                                    (220, 220, 60), (220, 60, 220),
                                    (60, 220, 220), (220, 140, 60),
                                    (140, 220, 60), (60, 140, 220),
                                ]
                                for cc, tid in zip(
                                    frame_canopy_components,
                                    frame_tree_ids,
                                ):
                                    lid = int(cc.get("label_id", 0))
                                    if lid <= 0:
                                        continue
                                    color = _tree_colors[
                                        int(tid) % len(_tree_colors)
                                    ]
                                    pix = (_labels_img == lid)
                                    if not pix.any():
                                        continue
                                    tt = np.zeros_like(rgb_co)
                                    tt[pix] = color
                                    rgb_co = _cv2_co.addWeighted(
                                        rgb_co, 0.85, tt, 0.15, 0,
                                    )
                                    # Tree id label at centroid.
                                    ys, xs = np.where(pix)
                                    if ys.size:
                                        cy_t = int(ys.mean())
                                        cx_t = int(xs.mean())
                                        _cv2_co.putText(
                                            rgb_co,
                                            f"T{int(tid)}",
                                            (cx_t, cy_t),
                                            _cv2_co.FONT_HERSHEY_SIMPLEX,
                                            0.6, (255, 255, 255), 2,
                                            _cv2_co.LINE_AA,
                                        )
                        # Bright outline on canopy boundary.
                        cm_u8 = cm_bool.astype(np.uint8) * 255
                        contours, _ = _cv2_co.findContours(
                            cm_u8, _cv2_co.RETR_EXTERNAL,
                            _cv2_co.CHAIN_APPROX_SIMPLE,
                        )
                        _cv2_co.drawContours(
                            rgb_co, contours, -1, (255, 255, 0), 2,
                        )
                        # Trunk bboxes in cyan.
                        if frame_canopy_components:
                            for cc in frame_canopy_components:
                                tb = cc.get("trunk_box")
                                if tb is not None:
                                    x1, y1, x2, y2 = (
                                        int(v) for v in tb
                                    )
                                    _cv2_co.rectangle(
                                        rgb_co, (x1, y1), (x2, y2),
                                        (0, 255, 255), 2,
                                    )
                        # Header text: # of canopy components,
                        # canopy fraction, # of trunks.
                        _cf = float(cm_bool.sum()) / float(cm_bool.size)
                        hdr = (
                            f"canopy_frac={_cf*100:.1f}%  "
                            f"trees={len(frame_canopy_components)}"
                        )
                        _cv2_co.putText(
                            rgb_co, hdr, (8, 22),
                            _cv2_co.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 4, _cv2_co.LINE_AA,
                        )
                        _cv2_co.putText(
                            rgb_co, hdr, (8, 22),
                            _cv2_co.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, _cv2_co.LINE_AA,
                        )
                        _cv2_co.imwrite(
                            str(co_path),
                            _cv2_co.cvtColor(rgb_co, _cv2_co.COLOR_RGB2BGR),
                            [int(_cv2_co.IMWRITE_JPEG_QUALITY), 88],
                        )
                    except Exception as _co_err:
                        print(
                            f"[warn] could not save canopy overlay "
                            f"for {img_path.name}: {_co_err!r}",
                            file=sys.stderr,
                        )

                # No-tree-in-frame detector: if neither a meaningful
                # FOREGROUND canopy NOR any close-foreground trunks
                # were found, this frame has no real subject tree --
                # ANY 'flower' detection is a false positive (grass
                # weeds, distant orchard rows, tiny far canopy, dust,
                # etc.). Used below to zero out per-prompt detections.
                #
                # 'Foreground' = canopy median depth within
                # [depth_min_mm, --flower-foreground-canopy-max-depth-mm].
                # A tiny background-tree canopy passes the size check
                # but its median depth is 3-5 m, so the depth check
                # rejects it. A partial-frame foreground tree with a
                # tiny but close canopy (median depth ~ 1-2 m) keeps
                # passing.
                _no_tree_in_frame = False
                if args.flower_require_tree_in_frame:
                    _canopy_present = False
                    if canopy_mask_img is not None:
                        _cf = (
                            float(canopy_mask_img.sum())
                            / float(canopy_mask_img.size)
                        )
                        if _cf >= args.tree_mask_min_canopy_frac:
                            _canopy_present = True
                            # Also require canopy median depth to be
                            # in the foreground band. Without this, a
                            # tiny background-row canopy patch passes
                            # the size check.
                            if (depth_mm is not None
                                    and args.flower_foreground_canopy_max_depth_mm > 0):
                                _cm_bool = canopy_mask_img.astype(bool)
                                _cm_depth = depth_mm[_cm_bool]
                                _cm_valid = _cm_depth[
                                    (_cm_depth >= args.flower_depth_min_mm)
                                    & (_cm_depth <= 60000)
                                ]
                                if _cm_valid.size >= 20:
                                    _cm_med = float(np.median(_cm_valid))
                                    if (_cm_med
                                            > args.flower_foreground_canopy_max_depth_mm):
                                        _canopy_present = False
                    _trunks_present = (
                        args.track_canopy
                        and bool(frame_canopy_components)
                    )
                    if (not _canopy_present) and (not _trunks_present):
                        _no_tree_in_frame = True

                # Depth-coverage fallback: if the frame has too little
                # valid depth (sparse D455 returns at frame edges,
                # distant scenes, sensor failures), skip every gate
                # that uses depth so a cascade of depth-based
                # rejections doesn't kill all the SAM detections.
                # Computed once per frame, consulted by every depth
                # gate downstream.
                _low_depth_coverage = False
                if (args.flower_depth_coverage_threshold > 0
                        and depth_mm is not None):
                    _vd_pix = (
                        (depth_mm >= args.flower_depth_min_mm)
                        & (depth_mm <= args.flower_depth_max_mm)
                    )
                    _coverage = float(_vd_pix.sum()) / float(depth_mm.size)
                    if _coverage < args.flower_depth_coverage_threshold:
                        _low_depth_coverage = True
                        print(
                            f"[depth-fallback] {img_path.name}: "
                            f"{_coverage*100:.1f}% < "
                            f"{args.flower_depth_coverage_threshold*100:.0f}% "
                            f"valid -- skipping depth gates",
                            file=sys.stderr,
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

                    # No-tree-in-frame zero-out: if the frame contains
                    # no foreground tree (no canopy mask, no detected
                    # trunks), every flower detection is a false
                    # positive. Force n=0 for flower prompts. Other
                    # prompts (e.g., trunk inference itself) are
                    # untouched -- we still want trunk results so
                    # tracking continuity works on subsequent frames.
                    is_flower_for_no_tree = (
                        "flower" in prompt.lower()
                        or "blossom" in prompt.lower()
                    )
                    if (_no_tree_in_frame and is_flower_for_no_tree
                            and n > 0):
                        # Drop all detections via an all-False keep.
                        keep_nt = np.zeros(n, dtype=bool)
                        # Audit before mutating arrays.
                        for _i in range(n):
                            orig_i = int(_i)  # surviving == identity here
                        audit_was_set = False
                        # Empty all detection arrays; preserve types.
                        masks_np = masks_np[keep_nt]
                        if scores_np is not None:
                            scores_np = scores_np[keep_nt]
                        if boxes_np is not None:
                            boxes_np = boxes_np[keep_nt]
                        n = 0
                        rt_key = (day, category, session, prompt)
                        rt = rejection_totals.setdefault(rt_key, {})
                        rt["no_tree_in_frame"] = (
                            rt.get("no_tree_in_frame", 0) + n_raw
                        )

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
                    if (args.depth and depth_mm is not None and n_raw > 0
                            and not _low_depth_coverage):
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
                        and not _low_depth_coverage
                    )
                    if (depth_mm is not None and n > 0
                            and not _low_depth_coverage
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
                    # Only skip when the CANOPY MASK ITSELF is unreliable
                    # (tiny / empty -- build_tree_mask couldn't capture
                    # the tree). Do NOT skip just because depth coverage
                    # is globally low: the canopy mask may still be
                    # fine, and disabling tree-mask in that case lets
                    # background grass-wildflowers leak through. The
                    # per-mask depth cap and ground-row gate still
                    # handle other failure modes; tree-mask is the
                    # primary canopy/ground separator and should
                    # remain active whenever it's usable.
                    canopy_overlap_mean: float | str = ""
                    _skip_tree_mask = False
                    if canopy_mask_img is not None:
                        _canopy_frac = (
                            float(canopy_mask_img.sum())
                            / float(canopy_mask_img.size)
                        )
                        if _canopy_frac < args.tree_mask_min_canopy_frac:
                            _skip_tree_mask = True
                    if (args.tree_mask and canopy_mask_img is not None
                            and n > 0 and not _skip_tree_mask):
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
                        # Phenology stage selector. When
                        # --flower-phenology is 'auto' or a specific
                        # stage name, override the white/pink
                        # thresholds with stage-tuned values from
                        # the sprayer pipeline reference. Otherwise
                        # use the args defaults as-is.
                        _stage_overrides: dict = {}
                        _stage_label = "off"
                        if args.flower_phenology != "off":
                            if args.flower_phenology == "auto":
                                _doy = doy_from_path(img_path)
                                if _doy is not None:
                                    _stage_label = phenol_stage_from_doy(
                                        _doy, args.flower_bloom_peak_doy,
                                    )
                            else:
                                _stage_label = args.flower_phenology
                            _stage_overrides = hsv_thresholds_for_stage(
                                _stage_label,
                            )
                        # Local thresholds: stage overrides win when
                        # provided; otherwise fall back to args.*.
                        _white_s_max = _stage_overrides.get(
                            "white_s_max", args.flower_white_s_max,
                        )
                        _white_v_min = _stage_overrides.get(
                            "white_v_min", args.flower_white_v_min,
                        )
                        _pink_v_min = _stage_overrides.get(
                            "pink_v_min", args.flower_pink_v_min,
                        )
                        _pink_s_min = _stage_overrides.get("pink_s_min", 20)
                        _pink_s_max = _stage_overrides.get("pink_s_max", 100)
                        _b_minus_r_max = _stage_overrides.get(
                            "b_minus_r_max", args.flower_b_minus_r_max,
                        )
                        _pink_bmr_max = _stage_overrides.get(
                            "pink_b_minus_r_max",
                            args.flower_pink_b_minus_r_max,
                        )
                        _pink_disabled = _stage_overrides.get(
                            "pink_disabled", False,
                        )
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
                                # B-R and G-R signed differences,
                                # ported from the sprayer pipeline's
                                # flower_detector. They give us two
                                # signals HSV alone can't:
                                #
                                #   b_minus_r <= 0 : pixel is COOL /
                                #     blue-shifted. Real apple petals
                                #     reflect cool sky onto their
                                #     undersides; specular leaf
                                #     glints, sun-bleached cuticles,
                                #     and most warm-tone false
                                #     positives have b_minus_r > 0.
                                #
                                #   g_minus_r > 12 : green dominates,
                                #     so this is foliage. Excludes
                                #     bright greenish leaves that
                                #     happen to pass the low-S gate.
                                R_chan = rgb_arr[..., 0].astype(np.int16)
                                G_chan = rgb_arr[..., 1].astype(np.int16)
                                B_chan = rgb_arr[..., 2].astype(np.int16)
                                b_minus_r = B_chan - R_chan
                                g_minus_r = G_chan - R_chan
                                green_dominant = (
                                    g_minus_r > args.flower_g_minus_r_max
                                )
                                # Leaf-bud exclusion: yellowish-green
                                # spring growth with H ∈ [20, 50] and
                                # S >= 15 -- bright but not a flower.
                                leaf_bud = (
                                    (Hc >= 20) & (Hc <= 50)
                                    & (Sc >= 15)
                                    & (g_minus_r >= -5)
                                )
                                # White petal mask. The b_minus_r
                                # check is the killer addition --
                                # without it, distant-tree foliage
                                # glints sneak past depth filters.
                                # Thresholds use _stage-adjusted
                                # locals when --flower-phenology is
                                # on, else args defaults.
                                white_mask = (
                                    (Sc <= _white_s_max)
                                    & (Vc >= _white_v_min)
                                    & (b_minus_r <= _b_minus_r_max)
                                    & ~green_dominant
                                    & ~leaf_bud
                                )
                                # Pink petal mask. Pink petals are
                                # also slightly cool (B > R is
                                # unusual but pink-shifted reds have
                                # b_minus_r near 0; -10 cutoff keeps
                                # them while excluding warm-leaf
                                # glints that test as "red hue").
                                # Disabled in the 'fruiting' stage
                                # where any pink-hue pixel is more
                                # likely a wildflower or fading leaf.
                                if _pink_disabled:
                                    pink_mask = np.zeros_like(white_mask)
                                else:
                                    pink_mask = (
                                        (((Hc >= 0) & (Hc <= 30))
                                         | ((Hc >= 150) & (Hc <= 179)))
                                        & ((Sc >= _pink_s_min) & (Sc <= _pink_s_max))
                                        & (Vc >= _pink_v_min)
                                        & (b_minus_r < _pink_bmr_max)
                                        & ~green_dominant
                                        & ~leaf_bud
                                    )
                                # Top-frame penalty: rows 0..N have
                                # the worst specular-leaf-glint
                                # density (sun strikes waxy cuticles
                                # at near-grazing angles). Use a
                                # stricter rule there.
                                if args.flower_top_frame_penalty_row > 0:
                                    top_n = int(
                                        args.flower_top_frame_penalty_row
                                    )
                                    if top_n > 0 and top_n < hsv_full.shape[0]:
                                        top_strict = (
                                            (Sc < 15)
                                            & (Vc > 200)
                                            & (b_minus_r <= -5)
                                            & ~green_dominant
                                        )
                                        white_mask[:top_n, :] = (
                                            white_mask[:top_n, :]
                                            & top_strict[:top_n, :]
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

                                # ============================================
                                # Reference-detector enhancements:
                                #   - has_texture (V_std + IR_std + Sobel)
                                #   - IR > N positive flower signal
                                #   - Multiple sky-type exclusions
                                #   - confirmed_real spatial gate
                                #   - Negative pixel masks (bark / dark / ground)
                                #   - Two-tier REGION + CORE
                                # All gated by their respective opt-in flags
                                # so back-compat is preserved.
                                # ============================================
                                _ir_8bit = None
                                if ir_arr is not None:
                                    _ir_norm = ir_arr.astype(np.float32)
                                    if _ir_norm.max() <= 1.5:
                                        _ir_8bit = np.clip(
                                            _ir_norm * 255.0, 0, 255,
                                        ).astype(np.uint8)
                                    else:
                                        _ir_8bit = np.clip(
                                            _ir_norm, 0, 255,
                                        ).astype(np.uint8)
                                # Texture map (only computed when used).
                                _has_texture = None
                                if (args.flower_use_texture
                                        or args.flower_confirmed_real
                                        or args.flower_exclude_sky_smooth
                                        or args.flower_exclude_sky_warm
                                        or args.flower_exclude_sky_upper):
                                    _tex = compute_texture_signals(
                                        rgb_arr, Vc, ir_arr,
                                        win=9,
                                        texture_threshold=args.flower_texture_threshold,
                                        edge_threshold=args.flower_edge_threshold,
                                    )
                                    _has_texture = _tex.get("has_texture")
                                # Depth helpers for confirmed_real and sky.
                                _valid_depth_pix = None
                                _no_depth_pix = None
                                _near_tree = None
                                _confirmed_real = None
                                # Skip depth-derived signals on low-
                                # coverage frames so confirmed_real
                                # doesn't reject every flower when
                                # depth is too sparse to be meaningful.
                                if (depth_mm is not None
                                        and not _low_depth_coverage):
                                    _vd = (
                                        (depth_mm >= args.flower_depth_min_mm)
                                        & (depth_mm <= args.flower_depth_max_mm)
                                    )
                                    _valid_depth_pix = _vd
                                    _no_depth_pix = ~_vd
                                    if (args.flower_confirmed_real
                                            and _has_texture is not None):
                                        _confirmed_real, _near_tree = (
                                            compute_confirmed_real(
                                                _vd, _has_texture,
                                                near_tree_radius_px=args.flower_near_tree_radius_px,
                                            )
                                        )
                                # IR positive flower signal (REGION tier 3).
                                # Captures dim shaded blossoms whose HSV
                                # alone is borderline -- petals reflect NIR
                                # strongly so high-IR pixels with V > 50 and
                                # low-S still qualify.
                                _ir_petal = None
                                if (_ir_8bit is not None
                                        and args.flower_ir_petal_min < 256):
                                    _ir_petal = (
                                        (_ir_8bit > args.flower_ir_petal_min)
                                        & (Vc > 50) & (Sc < 70)
                                        & ~green_dominant & ~leaf_bud
                                    )
                                    blossom_pix = blossom_pix | _ir_petal
                                # IR > N positive signal added to the gate
                                # so a pixel can pass on color OR IR alone.
                                if (_ir_8bit is not None
                                        and args.flower_ir_positive_min < 256):
                                    _ir_positive = (
                                        (_ir_8bit > args.flower_ir_positive_min)
                                        & ~green_dominant & ~leaf_bud
                                    )
                                    # Combined OR with current blossom_pix.
                                    blossom_pix = blossom_pix | _ir_positive
                                # Multiple sky-type exclusions.
                                if (_no_depth_pix is not None
                                        and (args.flower_exclude_sky_smooth
                                             or args.flower_exclude_sky_warm
                                             or args.flower_exclude_sky_upper
                                             or args.flower_exclude_sky_overcast
                                             or args.flower_exclude_sky_grey)):
                                    _sky_mask = compute_sky_exclusions(
                                        Hc, Sc, Vc, b_minus_r, _ir_8bit,
                                        _no_depth_pix,
                                        _has_texture if _has_texture is not None
                                            else np.zeros_like(Vc, dtype=bool),
                                        _near_tree,
                                        enable_smooth=args.flower_exclude_sky_smooth,
                                        enable_warm=args.flower_exclude_sky_warm,
                                        enable_upper=args.flower_exclude_sky_upper,
                                        enable_overcast=args.flower_exclude_sky_overcast,
                                        enable_grey=args.flower_exclude_sky_grey,
                                        enable_br=False,  # already covered by white_mask b_minus_r
                                        ir_sky_ceil=args.flower_ir_sky_ceil,
                                    )
                                    blossom_pix = blossom_pix & ~_sky_mask
                                # Negative pixel masks (bark / dark / ground).
                                if (args.flower_exclude_bark
                                        or args.flower_exclude_dark
                                        or args.flower_exclude_ground_grass):
                                    _neg = compute_negative_pixel_masks(
                                        Hc, Sc, Vc,
                                        enable_bark=args.flower_exclude_bark,
                                        enable_dark=args.flower_exclude_dark,
                                        enable_ground=args.flower_exclude_ground_grass,
                                    )
                                    blossom_pix = blossom_pix & ~_neg
                                # confirmed_real spatial gate.
                                if (args.flower_confirmed_real
                                        and _confirmed_real is not None):
                                    blossom_pix = blossom_pix & _confirmed_real
                                # has_texture gate (per-pixel).
                                if (args.flower_use_texture
                                        and _has_texture is not None):
                                    blossom_pix = blossom_pix & _has_texture

                                # ── Two-tier mask: CORE for counting ────
                                # CORE = strict white/pink-only intersection
                                # with the broad blossom_pix REGION.
                                # core_pct per cluster drives the confidence
                                # scaling and is also a downstream feature.
                                _flower_core = None
                                if args.flower_two_tier_mask:
                                    _white_core = (
                                        (Sc < 20) & (Vc > 150)
                                        & (b_minus_r <= 10)
                                        & ~green_dominant & ~leaf_bud
                                    )
                                    _pink_core = (
                                        (((Hc <= 12) | (Hc >= 160)))
                                        & (Sc >= 10) & (Sc <= 50)
                                        & (Vc > 130) & (b_minus_r < -10)
                                        & ~green_dominant & ~leaf_bud
                                    )
                                    if _valid_depth_pix is not None:
                                        _deep_pink_core = (
                                            (((Hc <= 20) | (Hc >= 150)))
                                            & (Sc >= 40) & (Sc <= 110)
                                            & (Vc > 80)
                                            & (b_minus_r < -20)
                                            & _valid_depth_pix
                                            & ~green_dominant & ~leaf_bud
                                        )
                                    else:
                                        _deep_pink_core = (
                                            (((Hc <= 20) | (Hc >= 150)))
                                            & (Sc >= 40) & (Sc <= 110)
                                            & (Vc > 80)
                                            & (b_minus_r < -20)
                                            & ~green_dominant & ~leaf_bud
                                        )
                                    _flower_core = (
                                        (_white_core | _pink_core | _deep_pink_core)
                                        & blossom_pix
                                    )
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
                                    # Per-mask CORE coverage (used by
                                    # the confidence-scaling step
                                    # below). _flower_core is the
                                    # strict-only mask from the
                                    # two-tier block; if disabled, we
                                    # skip core_pct entirely.
                                    if _flower_core is not None:
                                        _core_in_mask = int(
                                            (m_orig & _flower_core).sum()
                                        )
                                        _orig_area = int(m_orig.sum())
                                        _core_pct = (
                                            100.0 * _core_in_mask / _orig_area
                                            if _orig_area > 0 else 0.0
                                        )
                                        audit.set_meta(
                                            mi, core_pct=float(_core_pct),
                                        )
                                    # Optional anther hole-fill.
                                    # Closes 5-10 px gaps left by
                                    # yellow stamens at petal centers
                                    # (which fail the low-S white
                                    # rule). One blossom = one
                                    # connected mask after this.
                                    if (args.flower_fill_anther_holes
                                            and m_ref_raw.any()):
                                        m_filled_u8 = fill_anther_holes(
                                            m_ref_raw.astype(np.uint8) * 255
                                        )
                                        # Only keep filled pixels
                                        # that are inside the original
                                        # SAM mask -- the flood fill
                                        # can't accidentally claim
                                        # surrounding leaves because
                                        # the SAM mask bounds the
                                        # candidate region.
                                        m_ref_raw = (
                                            m_filled_u8 > 0
                                        ) & m_orig
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
                        # Skipped when --prgb-skip-centroid-check is
                        # set (lateral branches whose flowers extend
                        # slightly past the dilated ROI).
                        if (args.prgb and roi_mask_img is not None
                                and n > 0
                                and not args.prgb_skip_centroid_check):
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
                                and n > 0
                                and not _low_depth_coverage):
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
                        # Per-mask MAX-depth cap. Runs ALWAYS, even when
                        # --flower-depth-coverage-threshold has triggered
                        # the global depth-fallback. Catches the case
                        # where the global fallback bypassed all depth
                        # gates but a SPECIFIC mask still has enough
                        # valid-depth pixels to decide the mask is on
                        # a far background tree.
                        #
                        # Logic: if mask has >= min_pixels valid-depth
                        # pixels, compute their median; if median >
                        # cap, reject. If too few valid-depth pixels,
                        # don't decide (benefit of doubt).
                        if (args.flower_max_depth_cap_mm > 0
                                and depth_mm is not None and n > 0):
                            keep_cap = np.ones(n, dtype=bool)
                            cap_mm = float(args.flower_max_depth_cap_mm)
                            min_px = int(args.flower_max_depth_cap_min_pixels)
                            valid_pix_global = (depth_mm > 0) & (depth_mm < 60000)
                            for mi_cap in range(n):
                                mb_cap = masks_np[mi_cap].astype(bool)
                                if mb_cap.ndim == 3:
                                    mb_cap = mb_cap.any(axis=0)
                                valid_in_mask = mb_cap & valid_pix_global
                                if int(valid_in_mask.sum()) < min_px:
                                    continue  # not enough info
                                med = float(
                                    np.median(depth_mm[valid_in_mask])
                                )
                                if med > cap_mm:
                                    keep_cap[mi_cap] = False
                            audit.apply(keep_cap, "max_depth_cap")
                            if not keep_cap.all():
                                rt_key = (day, category, session, prompt)
                                rt = rejection_totals.setdefault(rt_key, {})
                                rt["max_depth_cap"] = (
                                    rt.get("max_depth_cap", 0)
                                    + int((~keep_cap).sum())
                                )
                                masks_np = masks_np[keep_cap]
                                if scores_np is not None:
                                    scores_np = scores_np[keep_cap]
                                if boxes_np is not None:
                                    boxes_np = boxes_np[keep_cap]
                                n = int(keep_cap.sum())
                        # Ground-row rejection. Catches dandelions /
                        # grass wildflowers in the bottom of the
                        # frame. A cluster whose centroid is below
                        # --flower-max-ground-row is dropped UNLESS
                        # its strict-core (blossom_pix) coverage is
                        # at least --flower-min-confirmed-pct-ground
                        # %. Real low-hanging blossoms confirm; grass
                        # wildflowers don't (their HSV mostly fails
                        # the strict bloom rule). Mirrors the
                        # reference flower_detector ground filter.
                        if (args.flower_max_ground_row > 0
                                and n > 0
                                and "blossom_pix" in dir()
                                and blossom_pix is not None):
                            keep_gr = np.ones(n, dtype=bool)
                            for gri in range(n):
                                mb_g = masks_np[gri].astype(bool)
                                if mb_g.ndim == 3:
                                    mb_g = mb_g.any(axis=0)
                                if not mb_g.any():
                                    continue
                                ys_g, _ = np.where(mb_g)
                                cy_g = float(ys_g.mean())
                                if cy_g <= args.flower_max_ground_row:
                                    continue
                                area_g = int(mb_g.sum())
                                # Tiny clusters skip the strict-core
                                # check (consistent with reference's
                                # `area > 200` guard).
                                if area_g <= 200:
                                    continue
                                core_g = int(
                                    (mb_g & blossom_pix).sum()
                                )
                                conf_pct = (
                                    100.0 * core_g / area_g
                                    if area_g > 0 else 0.0
                                )
                                if (conf_pct
                                        < args.flower_min_confirmed_pct_ground):
                                    keep_gr[gri] = False
                            audit.apply(keep_gr, "ground_row")
                            if not keep_gr.all():
                                rt_key = (day, category, session, prompt)
                                rt = rejection_totals.setdefault(rt_key, {})
                                rt["ground_row"] = (
                                    rt.get("ground_row", 0)
                                    + int((~keep_gr).sum())
                                )
                                masks_np = masks_np[keep_gr]
                                if scores_np is not None:
                                    scores_np = scores_np[keep_gr]
                                if boxes_np is not None:
                                    boxes_np = boxes_np[keep_gr]
                                n = int(keep_gr.sum())
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
                                # Match the in-loop refinement's
                                # color rules: HSV + b_minus_r cool-
                                # shadow + green-dominance + leaf-bud
                                # exclusion. See the refinement block
                                # for full rationale.
                                _R = rgb_arr[..., 0].astype(np.int16)
                                _G = rgb_arr[..., 1].astype(np.int16)
                                _B = rgb_arr[..., 2].astype(np.int16)
                                _bmr = _B - _R
                                _gmr = _G - _R
                                _green_dom = _gmr > args.flower_g_minus_r_max
                                _leaf_bud = (
                                    (_Hc >= 20) & (_Hc <= 50)
                                    & (_Sc >= 15)
                                    & (_gmr >= -5)
                                )
                                _white = (
                                    (_Sc <= args.flower_white_s_max)
                                    & (_Vc >= args.flower_white_v_min)
                                    & (_bmr <= args.flower_b_minus_r_max)
                                    & ~_green_dom
                                    & ~_leaf_bud
                                )
                                _pink = (
                                    (((_Hc >= 0) & (_Hc <= 30))
                                     | ((_Hc >= 150) & (_Hc <= 179)))
                                    & ((_Sc >= 20) & (_Sc <= 100))
                                    & (_Vc >= args.flower_pink_v_min)
                                    & (_bmr < args.flower_pink_b_minus_r_max)
                                    & ~_green_dom
                                    & ~_leaf_bud
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
                                    None if _low_depth_coverage else depth_mm,
                                    ndvi_arr,
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
                            edge_margin_sides=args.flower_edge_margin_sides_px,
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
                            max_mask_green_frac=args.flower_max_mask_green_frac,
                            green_h_lo=args.flower_green_hue_min,
                            green_h_hi=args.flower_green_hue_max,
                            green_s_min=args.flower_green_sat_min,
                            green_v_min=args.flower_green_val_min,
                            green_blossom_override_frac=(
                                args.flower_green_blossom_override_frac
                            ),
                            min_peaks_per_1000px=(
                                args.flower_min_peaks_per_1000px
                            ),
                            peak_min_distance_px=(
                                args.flower_peak_min_distance_px
                            ),
                            peak_threshold_abs=(
                                args.flower_peak_threshold_abs
                            ),
                            peak_min_area_px=(
                                args.flower_peak_min_area_px
                            ),
                            peak_min_distance_px2=(
                                args.flower_peak_min_distance_px2
                            ),
                            peak_prominence_min=(
                                args.flower_peak_prominence_min
                            ),
                            min_anther_holes_per_1000px=(
                                args.flower_min_anther_holes_per_1000px
                            ),
                            anther_petal_v_min=(
                                args.flower_anther_petal_v_min
                            ),
                            anther_hole_min_area_px=(
                                args.flower_anther_hole_min_area_px
                            ),
                            anther_hole_max_area_px=(
                                args.flower_anther_hole_max_area_px
                            ),
                            anther_min_area_px=(
                                args.flower_anther_min_area_px
                            ),
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
                        # Track which parent index each sub-mask
                        # came from so the audit's surviving-original-
                        # SAM-index map stays consistent. After
                        # splitting, audit.remap_after_split() rebuilds
                        # surviving so each sub-mask points back to
                        # the right original SAM detection.
                        parent_per_sub: list[int] = []
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
                                parent_per_sub.append(idx)
                                continue
                            subs = split_cluster_mask(
                                m, rgb_arr_for_split,
                                min_blossom_area_px=args.split_min_blossom_area_px,
                                min_marker_distance_px=args.split_min_marker_distance_px,
                                seed_dilate_px=args.split_seed_dilate_px,
                            )
                            # Per-CC area cap: a 250-px blossom that
                            # over-splits to 2-3 sub-peaks (petal ridge
                            # noise, partial anther holes) shouldn't
                            # be counted as 2-3 flowers. Cap at
                            # ceil(parent_area / area_per_flower) and
                            # keep the largest sub-masks.
                            if args.split_area_cap and len(subs) > 1:
                                per = max(1, int(args.flower_area_per_flower_px))
                                cap = max(1, (cluster_area + per - 1) // per)
                                if len(subs) > cap:
                                    subs = sorted(
                                        subs,
                                        key=lambda s: -int(np.asarray(s).sum()),
                                    )[:cap]
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
                                parent_per_sub.append(idx)
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
                            # Tell the audit each new sub-mask's
                            # original SAM index so per-mask rejection
                            # log + debug overlay stay coherent across
                            # the split.
                            try:
                                audit.remap_after_split(parent_per_sub)
                            except Exception:
                                pass

                    # Density-based individual flower estimate per cluster.
                    # With --flower-confidence-scale, scale each cluster's
                    # area-based estimate down when its CORE coverage is
                    # < 50% AND the cluster is large (>500 px). Reduces
                    # over-count on warm-petal / foliage-mixed clusters.
                    # Mirrors the reference detector's behaviour.
                    if is_flower_prompt and kept_areas:
                        per = args.flower_area_per_flower_px
                        if args.flower_confidence_scale and n > 0:
                            est_flowers = 0
                            kept_orig = audit.kept_originals()
                            for ki, a in enumerate(kept_areas):
                                count = max(1, round(a / per))
                                if a > 500 and ki < len(kept_orig):
                                    orig_i = int(kept_orig[ki])
                                    cp = audit.meta.get(
                                        orig_i, {}).get("core_pct", 100.0)
                                    if cp < 50.0:
                                        scale = max(0.5, cp / 100.0 + 0.3)
                                        count = max(1, round(count * scale))
                                est_flowers += int(count)
                        else:
                            est_flowers = int(sum(
                                max(1, round(a / per)) for a in kept_areas
                            ))
                    else:
                        est_flowers = n if not is_flower_prompt else 0
                    # Density score (Estrada-style Gaussian-blur sum).
                    # Optional alternative counting metric written to the
                    # CSV under 'flower_density_sum'. Sum over the union
                    # of all kept flower masks.
                    flower_density_sum: float | str = ""
                    if (args.flower_compute_density_score
                            and is_flower_prompt and n > 0):
                        flower_density_sum = round(
                            compute_density_score(
                                masks_np, (img.height, img.width),
                                sigma=4.0, kernel_size=15,
                            ),
                            3,
                        )

                    # Tracker step (after depth + flower-quality filters, so we
                    # only track real near-field, properly-sized flowers).
                    track_ids: list[int] = []
                    if (args.track and prompt in tracked_set
                            and current_trackers and n > 0 and boxes_np is not None):
                        track_ids = current_trackers[prompt].step(
                            boxes_np, scores_np,
                            areas=kept_areas if is_flower_prompt else None,
                        )

                    # Assign each kept flower mask to a tree_id via
                    # max-overlap on the canopy CC labels image.
                    flower_tree_ids: list[int] = []
                    if (args.track_canopy and is_flower_prompt and n > 0
                            and frame_canopy_components):
                        flower_tree_ids = assign_flowers_to_trees(
                            masks_np, frame_canopy_components,
                            frame_tree_ids,
                        )

                        # Per-tree foreground-relative depth gate.
                        # Reject flowers whose host canopy is
                        # significantly farther back than the
                        # closest canopy in the frame. Catches the
                        # "flower on background tree" case that the
                        # per-mask depth cap misses when the
                        # background tree is within the global
                        # depth_max limit (e.g., foreground at
                        # 1.5 m, background row at 2.5 m -- both
                        # pass --flower-max-depth-cap-mm 3500 but
                        # the background tree's flowers are wrong).
                        if (args.flower_max_behind_foreground_mm > 0
                                and depth_mm is not None
                                and flower_tree_ids
                                and n > 0):
                            comp_depths = canopy_component_depth_medians(
                                frame_canopy_components, depth_mm,
                            )
                            if comp_depths:
                                fg_depth = min(comp_depths.values())
                                max_behind = float(
                                    args.flower_max_behind_foreground_mm
                                )
                                tid_to_depth: dict[int, float] = {}
                                for c, _tid in zip(
                                    frame_canopy_components,
                                    frame_tree_ids,
                                ):
                                    lid = int(c.get("label_id", 0))
                                    if lid in comp_depths:
                                        tid_to_depth[int(_tid)] = (
                                            comp_depths[lid]
                                        )
                                keep_fd = np.ones(n, dtype=bool)
                                for fi, t_id in enumerate(flower_tree_ids):
                                    if t_id < 0:
                                        # Unassigned flower -- benefit
                                        # of doubt; the gate only
                                        # fires on confidently
                                        # assigned background-tree
                                        # flowers.
                                        continue
                                    d_tree = tid_to_depth.get(int(t_id))
                                    if d_tree is None:
                                        continue
                                    if d_tree > fg_depth + max_behind:
                                        keep_fd[fi] = False
                                if not keep_fd.all():
                                    audit.apply(
                                        keep_fd, "max_behind_fg",
                                    )
                                    rt_key = (
                                        day, category, session, prompt,
                                    )
                                    rt = rejection_totals.setdefault(
                                        rt_key, {},
                                    )
                                    rt["max_behind_fg"] = (
                                        rt.get("max_behind_fg", 0)
                                        + int((~keep_fd).sum())
                                    )
                                    masks_np = masks_np[keep_fd]
                                    if scores_np is not None:
                                        scores_np = scores_np[keep_fd]
                                    if boxes_np is not None:
                                        boxes_np = boxes_np[keep_fd]
                                    flower_tree_ids = [
                                        t for t, k in zip(
                                            flower_tree_ids, keep_fd,
                                        ) if k
                                    ]
                                    if (track_ids
                                            and len(track_ids) == len(keep_fd)):
                                        track_ids = [
                                            t for t, k in zip(
                                                track_ids, keep_fd,
                                            ) if k
                                        ]
                                    n = int(keep_fd.sum())

                        # Update flower-track -> tree-id mapping
                        # (use the most-common tree_id per track).
                        if track_ids and len(track_ids) == len(flower_tree_ids):
                            for det_i, (tid, tree_i) in enumerate(
                                zip(track_ids, flower_tree_ids)
                            ):
                                if tid < 0 or tree_i < 0:
                                    continue
                                key = (
                                    (day, category, session), prompt, tid,
                                )
                                # Simple "first non-negative wins";
                                # for max-vote replace this with a
                                # Counter per track. First-vote is
                                # adequate when canopy tracking is
                                # consistent.
                                flower_track_tree_id.setdefault(
                                    key, int(tree_i),
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
                    # When --save-empty-overlays is on, save EVERY frame's
                    # overlay even when n=0; useful for confirming SAM 3
                    # saw nothing vs the frame being skipped for some
                    # other reason.
                    save_this_overlay = args.save_overlays and (
                        n > 0 or args.save_empty_overlays
                    ) and (not args.track or prompt in tracked_set)
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
                        "flower_density_sum": flower_density_sum,
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
                # Print the full traceback (not just the exception
                # message) for the FIRST few errors so debugging
                # CUDA / dtype / tile-NMS issues doesn't require a
                # separate reproducer script. After 5 errors, fall
                # back to one-liners to keep the log readable.
                if not hasattr(main, "_err_count"):
                    main._err_count = 0  # type: ignore[attr-defined]
                main._err_count += 1  # type: ignore[attr-defined]
                if main._err_count <= 5:
                    import traceback as _tb
                    print(
                        f"[ERR] {img_path}: {e}\n"
                        f"      (full traceback below; later errors "
                        f"abbreviated)",
                        file=sys.stderr,
                    )
                    _tb.print_exc(file=sys.stderr)
                else:
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
        if (args.track_canopy and current_canopy_tracker is not None):
            d_last, c_last, s_last = current_session_key
            for trk in current_canopy_tracker.summary():
                tree_summaries.append({
                    "day": d_last, "category": c_last,
                    "session": s_last,
                    "tree_id": trk["tree_id"],
                    "first_frame": trk["first_frame"],
                    "last_frame": trk["last_frame"],
                    "n_frames": trk["n_frames"],
                    "max_area_px": trk.get("max_area", 0),
                })

    # ---- per-tree CSVs (when --track-canopy is on) -------------
    if args.track_canopy and tree_summaries:
        ts_path = out_dir / "trees_summary.csv"
        ts_fields = ["day", "category", "session", "tree_id",
                     "first_frame", "last_frame", "n_frames",
                     "max_area_px"]
        with open(ts_path, "w", newline="", encoding="utf-8") as tf_:
            tw_ = csv.DictWriter(tf_, fieldnames=ts_fields)
            tw_.writeheader()
            for r in tree_summaries:
                tw_.writerow(r)
        print(f"[done] per-session unique trees -> {ts_path}")
    if args.track_canopy and tree_per_frame_rows:
        tpf_path = out_dir / "trees_per_frame.csv"
        tpf_fields = ["day", "category", "session", "image", "tree_id",
                      "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
                      "area_px"]
        with open(tpf_path, "w", newline="", encoding="utf-8") as pf_:
            pw_ = csv.DictWriter(pf_, fieldnames=tpf_fields)
            pw_.writeheader()
            for r in tree_per_frame_rows:
                pw_.writerow(r)
        print(f"[done] per-frame tree appearances -> {tpf_path}")

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
