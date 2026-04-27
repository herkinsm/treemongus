"""Sprayer-pipeline tree-mask builder, ported verbatim into this repo.

Source: ``sprayer_pipeline/tree_mask.py`` -- the canonical Beer-Lambert
canopy-mask builder used by the sprayer.

Self-contained: only depends on numpy + cv2. Constants below are inlined
copies of ``sprayer_pipeline.config`` so this module ships with the
segmenter and works on the HPC without the full sprayer_pipeline install.

Behaviour intentionally identical to the upstream module so the segmenter's
tree masks line up with what the sprayer / flower-filter pipeline already
trusts.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Frame geometry / depth / ROI constants (copied from sprayer_pipeline.config) ──
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CANOPY_DEPTH_MIN_MM = 600
CANOPY_DEPTH_MAX_MM = 3000
ROI_COL_X = [(285, 319), (320, 354)]

# ── tree_mask.py constants ──
ROW_GROUND_START = 280
ROW_CUTOFF_BOTTOM = 480
ROW_UPPER_CANOPY = 280
ROW_CENTER_TOP = 100
ROW_CENTER_BOTTOM = 300
DEPTH_BAND_HALF_MM = 500
GROUND_BAND_MM = 100
ROW_LOWER_BAND_START = 385
GROUND_BAND_MM_ROW3 = 500
ROW_WIDTH_GROUND_FRACTION = 0.50
CANOPY_MAX_ROI_STRIP_ROW_CORRELATION = 0.70
CANOPY_MIN_ROI_STRIP_DEPTH_STD_MM = 200


def _estimate_center_depth(depth_f: np.ndarray, anchored_mask: np.ndarray) -> float:
    upper_mask = np.zeros_like(anchored_mask)
    upper_mask[:ROW_GROUND_START, :] = anchored_mask[:ROW_GROUND_START, :]
    upper_depths = depth_f[upper_mask > 0]
    if len(upper_depths) > 10:
        return float(np.median(upper_depths))
    main_depths = depth_f[anchored_mask > 0]
    if len(main_depths) > 10:
        return float(np.median(main_depths))
    h, w = depth_f.shape
    cx = w // 2
    center_strip = depth_f[
        ROW_CENTER_TOP:ROW_CENTER_BOTTOM,
        max(0, cx - 40):min(w, cx + 40),
    ]
    center_fg = center_strip[
        (center_strip >= CANOPY_DEPTH_MIN_MM)
        & (center_strip <= CANOPY_DEPTH_MAX_MM)
    ]
    if len(center_fg) > 10:
        return float(np.median(center_fg))
    return float((CANOPY_DEPTH_MIN_MM + CANOPY_DEPTH_MAX_MM) * 0.5)


def build_tree_mask(
    depth_mm: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    *,
    roi_cols: Optional[Tuple[int, int]] = None,
    apply_ground_filter: bool = True,
    hsv: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a (H, W) uint8 canopy mask (0 = background, 255 = canopy).

    Verbatim port of sprayer_pipeline.tree_mask.build_tree_mask. See the
    upstream module's docstring for the per-stage rationale; the algorithm
    is reproduced step-for-step here so the segmenter shares an identical
    canopy with the sprayer pipeline.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required for build_tree_mask")
    if depth_mm.ndim != 2:
        raise ValueError(f"depth_mm must be 2-D, got shape {depth_mm.shape}")
    h, w = depth_mm.shape

    # 0. Depth denoising
    depth_mm = cv2.medianBlur(depth_mm, 3)
    depth_f = depth_mm.astype(np.float32)

    # 1. Foreground depth band
    fg = (depth_f >= float(CANOPY_DEPTH_MIN_MM)) & (depth_f <= float(CANOPY_DEPTH_MAX_MM))

    # 2. Optional sky exclusion from RGB
    hsv_cached: Optional[np.ndarray] = None
    if rgb is not None:
        if rgb.ndim != 3 or rgb.shape[:2] != (h, w) or rgb.shape[2] != 3:
            raise ValueError(
                f"rgb must be ({h}, {w}, 3) to match depth; got {rgb.shape}"
            )
        if hsv is not None and hsv.shape == (h, w, 3):
            hsv_cached = hsv
        else:
            hsv_cached = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
        H_c, S_c, V_c = hsv_cached[:, :, 0], hsv_cached[:, :, 1], hsv_cached[:, :, 2]
        sky_blue = (H_c >= 85) & (H_c <= 145) & (S_c > 8)
        sky_bright = (V_c > 185) & (S_c < 15)
        fg = fg & ~(sky_blue | sky_bright)

    # 3. Small-component noise filter (>=4 px)
    mask_u8 = fg.astype(np.uint8) * 255
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_cc > 1:
        keep_lut = np.zeros(n_cc, dtype=np.uint8)
        keep_lut[1:] = np.where(
            stats[1:, cv2.CC_STAT_AREA] >= 4, 255, 0,
        ).astype(np.uint8)
        mask_u8 = keep_lut[labels]

    # 4. Anchor on the sprayer ROI columns via connected components
    if roi_cols is None:
        c0 = min(c for c, _ in ROI_COL_X)
        c1 = max(c1 for _, c1 in ROI_COL_X)
    else:
        c0, c1 = roi_cols
    n_cc, labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8,
    )
    search_c0 = max(0, c0 - 120)
    search_c1 = min(w - 1, c1 + 120)
    if n_cc > 1:
        band = labels[20:ROW_UPPER_CANOPY, search_c0:search_c1 + 1].ravel()
        counts = np.bincount(band, minlength=n_cc)
        counts[0] = 0
        lut = np.where(counts > 10, np.uint8(255), np.uint8(0))
        main_mask = lut[labels]
    else:
        main_mask = np.zeros_like(mask_u8)
        counts = np.zeros(n_cc, dtype=np.int64)

    center_depth = _estimate_center_depth(depth_f, main_mask)

    # 4b. Forward-branch recovery
    if n_cc > 1 and center_depth > CANOPY_DEPTH_MIN_MM:
        areas_all = cc_stats[:, cv2.CC_STAT_AREA]
        centroids_y_all = cc_centroids[:, 1]
        labels_flat = labels.ravel()
        depth_sums = np.bincount(
            labels_flat, weights=depth_f.ravel(), minlength=n_cc,
        )
        pixel_counts = np.bincount(labels_flat, minlength=n_cc)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_depths = depth_sums / np.maximum(pixel_counts, 1)
        unanchored = counts <= 10
        unanchored[0] = False
        candidate = (
            unanchored
            & (areas_all >= 100)
            & (mean_depths >= CANOPY_DEPTH_MIN_MM)
            & (mean_depths < (center_depth - 50.0))
            & (centroids_y_all < ROW_LOWER_BAND_START)
        )
        if candidate.any():
            lut2 = np.zeros(n_cc, dtype=np.uint8)
            lut2[candidate] = 255
            main_mask = np.maximum(main_mask, lut2[labels])

    # 4c. ROI-strip ground-plane detector
    R_STRIP_MAX = float(CANOPY_MAX_ROI_STRIP_ROW_CORRELATION)
    STRIP_STD_MIN = float(CANOPY_MIN_ROI_STRIP_DEPTH_STD_MM)
    if 0 < R_STRIP_MAX < 1.0 and STRIP_STD_MIN > 0:
        strip_c0 = min(c for c, _ in ROI_COL_X)
        strip_c1 = max(c1 for _, c1 in ROI_COL_X)
        lower_strip = main_mask[ROW_GROUND_START:, strip_c0:strip_c1 + 1] > 0
        if int(lower_strip.sum()) >= 300:
            ys_rel, xs_rel = np.where(lower_strip)
            ys_abs = ys_rel + ROW_GROUND_START
            ds = depth_f[ys_abs, xs_rel + strip_c0]
            d_std = float(ds.std())
            if d_std < STRIP_STD_MIN:
                if len(ys_abs) >= 2 and d_std > 1e-6:
                    r = float(np.corrcoef(
                        ys_abs.astype(np.float64),
                        ds.astype(np.float64),
                    )[0, 1])
                else:
                    r = 1.0
                if abs(r) > R_STRIP_MAX:
                    main_mask[ROW_GROUND_START:, :] = 0

    # 5. Fast dilation at half-res
    small = cv2.resize(main_mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    small = cv2.dilate(small, k_small, iterations=1)
    tree_zone = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    tree_mask = cv2.bitwise_and(mask_u8, tree_zone)

    # 6. Upper-canopy recovery
    upper_fg = mask_u8[:ROW_UPPER_CANOPY, :].copy()
    upper_depth = depth_f[:ROW_UPPER_CANOPY, :]
    too_far = upper_depth > (center_depth + DEPTH_BAND_HALF_MM)
    too_close = upper_depth < max(1, center_depth - DEPTH_BAND_HALF_MM)
    upper_fg[too_far | too_close] = 0
    tree_mask[:ROW_UPPER_CANOPY, :] = np.maximum(
        tree_mask[:ROW_UPPER_CANOPY, :], upper_fg,
    )

    # 7. Far-background depth cutoff
    far_bg = depth_f > (center_depth + DEPTH_BAND_HALF_MM)
    tree_mask[far_bg] = 0

    # 8. Row-banded ground filter
    ground_depth_r3 = depth_f > (center_depth + GROUND_BAND_MM_ROW3)
    tree_mask[ROW_GROUND_START:ROW_LOWER_BAND_START, :] = np.where(
        ground_depth_r3[ROW_GROUND_START:ROW_LOWER_BAND_START, :],
        0,
        tree_mask[ROW_GROUND_START:ROW_LOWER_BAND_START, :],
    )
    ground_depth_r4 = depth_f > (center_depth + GROUND_BAND_MM)
    tree_mask[ROW_LOWER_BAND_START:, :] = np.where(
        ground_depth_r4[ROW_LOWER_BAND_START:, :],
        0,
        tree_mask[ROW_LOWER_BAND_START:, :],
    )

    # 9. Background grass (HSV green + depth)
    if hsv_cached is not None:
        H_g, S_g, V_g = hsv_cached[:, :, 0], hsv_cached[:, :, 1], hsv_cached[:, :, 2]
        green_like = (H_g >= 25) & (H_g <= 85) & (S_g > 40) & (V_g > 50)
        far_green_r3 = green_like & (depth_f > (center_depth + GROUND_BAND_MM_ROW3))
        tree_mask[ROW_GROUND_START:ROW_LOWER_BAND_START, :][
            far_green_r3[ROW_GROUND_START:ROW_LOWER_BAND_START, :]
        ] = 0
        far_green_r4 = green_like & (depth_f > (center_depth + GROUND_BAND_MM))
        tree_mask[ROW_LOWER_BAND_START:, :][
            far_green_r4[ROW_LOWER_BAND_START:, :]
        ] = 0

    # 10. Hard bottom cutoff
    tree_mask[ROW_CUTOFF_BOTTOM:, :] = 0

    # 11. Per-component branch-vs-ground filter
    if apply_ground_filter:
        lower = tree_mask[ROW_GROUND_START:, :]
        lower_depth = depth_f[ROW_GROUND_START:, :]
        if lower.size and (lower > 0).any():
            n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
                lower, connectivity=8,
            )
            rejected_any = False
            lower_band_local = ROW_LOWER_BAND_START - ROW_GROUND_START
            _, W_l = lower.shape
            labels_flat = labels.ravel()
            order = np.argsort(labels_flat, kind="stable")
            sorted_labels = labels_flat[order]
            counts_per_label = np.bincount(sorted_labels, minlength=n_cc)
            cc_offsets = np.concatenate(
                ([0], np.cumsum(counts_per_label)),
            ).astype(np.int64)
            depth_flat = lower_depth.ravel()
            for i in range(1, n_cc):
                _x_cc, _y_cc, w_cc, h_cc, area = stats[i]
                if area < 40:
                    continue
                idx_flat = order[cc_offsets[i]:cc_offsets[i + 1]]
                cc_depths = depth_flat[idx_flat]
                if cc_depths.size < 10:
                    continue
                cc_med = float(np.median(cc_depths))
                ys_cc = (idx_flat // W_l).astype(np.int32, copy=False)
                cc_centroid_y = float(ys_cc.mean())
                in_row4_band = cc_centroid_y >= lower_band_local
                threshold_mm = (
                    GROUND_BAND_MM if in_row4_band else GROUND_BAND_MM_ROW3
                )
                beyond_trunk = cc_med > center_depth + threshold_mm
                if beyond_trunk:
                    lower.flat[idx_flat] = 0
                    rejected_any = True
                    continue
                if (w_cc >= 40 and h_cc < 60
                        and h_cc * 2.5 < w_cc
                        and beyond_trunk):
                    lower.flat[idx_flat] = 0
                    rejected_any = True
                    continue
                corr_gate = 0.85 if in_row4_band else 0.90
                if area >= 100:
                    yf = ys_cc.astype(np.float32)
                    yf -= yf.mean()
                    zf = cc_depths.astype(np.float32, copy=True)
                    zf -= zf.mean()
                    denom = float(
                        np.sqrt((yf * yf).sum()) * np.sqrt((zf * zf).sum())
                    )
                    if denom > 0:
                        corr = float((yf * zf).sum() / denom)
                        if corr > corr_gate:
                            _, row_idx = np.unique(ys_cc, return_inverse=True)
                            cnt = np.bincount(row_idx)
                            s = np.bincount(row_idx, weights=cc_depths)
                            s2 = np.bincount(
                                row_idx,
                                weights=(cc_depths.astype(np.float64)
                                         * cc_depths.astype(np.float64)),
                            )
                            with np.errstate(invalid="ignore", divide="ignore"):
                                mean_z = s / cnt
                                var = s2 / cnt - mean_z * mean_z
                            np.maximum(var, 0.0, out=var)
                            std = np.sqrt(var)
                            mask3 = cnt >= 3
                            mean_row_std = (
                                float(std[mask3].mean()) if mask3.any() else 0.0
                            )
                            if mean_row_std < 70.0:
                                lower.flat[idx_flat] = 0
                                rejected_any = True
                                continue
            if rejected_any:
                tree_mask[ROW_GROUND_START:, :] = lower

    # 12. Depth-gated row-width ground filter
    if apply_ground_filter and ROW_WIDTH_GROUND_FRACTION < 1.0:
        lower = tree_mask[ROW_GROUND_START:, :]
        lower_depth = depth_f[ROW_GROUND_START:, :]
        lower_band_local = ROW_LOWER_BAND_START - ROW_GROUND_START
        if lower.size:
            per_row_frac = (lower > 0).sum(axis=1) / float(lower.shape[1])
            for ri in np.where(per_row_frac >= ROW_WIDTH_GROUND_FRACTION)[0]:
                row_pixels = lower[ri] > 0
                row_depths = lower_depth[ri][row_pixels]
                if row_depths.size == 0:
                    continue
                row_med = float(np.median(row_depths))
                threshold_mm = (
                    GROUND_BAND_MM if ri >= lower_band_local
                    else GROUND_BAND_MM_ROW3
                )
                if row_med > center_depth + threshold_mm:
                    lower[ri, :] = 0
            tree_mask[ROW_GROUND_START:, :] = lower

    return tree_mask


# ── Sprayer ROI grid (10 zones, 2 cols x 5 rows) ─────────────────
# Verbatim from sprayer_pipeline/config.py:ROI_RECTS at reference
# layout (640x480 frame). Each tuple is (x0, y0, x1, y1).
ROI_ROW_Y = [
    (1, 94),
    (97, 190),
    (193, 286),
    (289, 382),
    (385, 478),
]
ROI_RECTS = [
    (ROI_COL_X[side][0], ROI_ROW_Y[row][0],
     ROI_COL_X[side][1], ROI_ROW_Y[row][1])
    for side in range(2)
    for row in range(5)
]
BEER_LAMBERT_K = 0.5


def zone_canopy_fractions(
    tree_mask: np.ndarray,
    rects=ROI_RECTS,
):
    """Canopy fraction (0..1) for each rectangle in ``rects``."""
    out = []
    mask_bool = tree_mask > 0
    for (x0, y0, x1, y1) in rects:
        sub = mask_bool[y0:y1 + 1, x0:x1 + 1]
        if sub.size == 0:
            out.append(0.0)
            continue
        out.append(float(sub.sum()) / float(sub.size))
    return out


def beer_lambert_lai(canopy_fraction: float, k: float = BEER_LAMBERT_K) -> float:
    """LAI = -ln(1 - cf) / k, clamped to avoid blow-up at cf in {0, 1}."""
    gap = 1.0 - float(canopy_fraction)
    gap = max(0.01, min(0.99, gap))
    return float(-np.log(gap) / max(1e-6, k))


# ── Wide tree mask (verbatim port of sprayer_pipeline.tree_aggregate
# .build_wide_tree_mask, the validated R²=0.564 LAI feature extractor)
# ─────────────────────────────────────────────────────────────────────
_BOTTOM_CUTOFF = 420


def build_wide_tree_mask(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    *,
    isolate_target_tree: bool = False,
    lateral_tolerance_frac: Optional[float] = None,
    bottom_cutoff_row: int = _BOTTOM_CUTOFF,
    apply_gradient_ground_filter: bool = True,
    gradient_threshold_mm_per_row: float = 3.0,
    apply_blue_cv_sky_filter: bool = True,
) -> np.ndarray:
    """Full-canopy mask, including the ground-depth-gradient filter that
    works on horizontal-facing cameras. Verbatim port of
    sprayer_pipeline.tree_aggregate.build_wide_tree_mask.

    Pipeline:
      1. Depth band [600, 3000] mm.
      2. HSV sky exclusion + blue-channel CV sky filter (top rows).
      3. Optional depth-gradient ground filter: pixel is rejected if
         dDepth/dRow > threshold over a 5-row window. Real ground
         on a horizontal camera has dDepth/dRow > 3-5 mm/row;
         canopy is depth-flat. KEY for our use-case.
      4. Morphology open 3x3, CC keep (largest if
         isolate_target_tree else all CCs > 2000 px), dilate 11x11.
      5. Optional lateral-tolerance band (central fraction of width).
      6. Hard bottom cutoff at row 420.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) required for build_wide_tree_mask")
    depth_f = depth_mm.astype(np.float32)
    h, w = depth_f.shape

    fg = (
        (depth_f >= float(CANOPY_DEPTH_MIN_MM))
        & (depth_f <= float(CANOPY_DEPTH_MAX_MM))
    )

    # HSV sky exclusion
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]
    sky_blue = (H >= 85) & (H <= 145) & (S > 8)
    sky_bright = (V > 185) & (S < 15)
    fg = fg & ~(sky_blue | sky_bright)

    # Blue-channel CV sky filter (top rows only).
    if apply_blue_cv_sky_filter:
        blk = 48
        max_row = 240
        blue = rgb[:, :, 2].astype(np.float32)
        red = rgb[:, :, 0].astype(np.float32)
        k = (blk, blk)
        mean_b = cv2.blur(blue, k)
        mean_r = cv2.blur(red, k)
        mean_b2 = cv2.blur(blue * blue, k)
        var_b = np.clip(mean_b2 - mean_b * mean_b, 0.0, None)
        std_b = np.sqrt(var_b)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_pct = np.where(mean_b > 1.0, 100.0 * std_b / mean_b, 0.0)
        blue_dominates = mean_b > (mean_r + 3.0)
        sky_blue_cv = (
            (mean_b > 171.0)
            & (cv_pct > 1.0) & (cv_pct < 10.0)
            & blue_dominates
        )
        if 0 < max_row < h:
            sky_blue_cv[max_row:, :] = False
        fg = fg & ~sky_blue_cv

    # Depth-gradient ground filter -- crucial for horizontal cameras.
    # On a horizontal camera, ground depth INCREASES with image row
    # (rows farther down see ground farther away). dDepth/dRow > 3 mm
    # per row is ground; canopy is roughly depth-flat (gradient ≈ 0).
    if apply_gradient_ground_filter:
        gap = 5
        grad = np.zeros_like(depth_f)
        grad[:-gap, :] = depth_f[gap:, :] - depth_f[:-gap, :]
        valid = (depth_f > 0) & (np.roll(depth_f, -gap, axis=0) > 0)
        thr = float(gradient_threshold_mm_per_row)
        ground_like = valid & (grad > thr * gap)
        fg = fg & ~ground_like

    m = fg.astype(np.uint8) * 255
    m = cv2.morphologyEx(
        m, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    # CC keep
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    if isolate_target_tree:
        best_i, best_area = -1, 0
        for i in range(1, n_cc):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a > best_area:
                best_area = a
                best_i = i
        if best_i > 0 and best_area > 2000:
            keep[labels == best_i] = 255
    else:
        for i in range(1, n_cc):
            if stats[i, cv2.CC_STAT_AREA] > 2000:
                keep[labels == i] = 255

    keep = cv2.dilate(
        keep,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    keep = cv2.bitwise_and(m, keep)

    # Lateral tolerance band
    if (lateral_tolerance_frac is not None
            and 0 < float(lateral_tolerance_frac) < 1.0):
        half_band = int(0.5 * float(lateral_tolerance_frac) * w)
        xc = w // 2
        left_edge = max(0, xc - half_band)
        right_edge = min(w, xc + half_band)
        keep[:, :left_edge] = 0
        keep[:, right_edge:] = 0

    # Hard bottom cutoff
    if 0 < bottom_cutoff_row < h:
        keep[bottom_cutoff_row:, :] = 0

    return keep


__all__ = [
    "build_tree_mask",
    "build_wide_tree_mask",
    "zone_canopy_fractions",
    "beer_lambert_lai",
    "ROI_RECTS",
    "BEER_LAMBERT_K",
]
