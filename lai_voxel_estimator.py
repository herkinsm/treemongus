"""lai_voxel_estimator.py — High-accuracy per-tree LAI estimation
via hierarchical SAM2 segmentation, multi-view 3D voxel fusion, and
Beer-Lambert gap fraction, with LAI-2200C calibration.

Designed for ground-truth-quality LAI estimation as a research
deliverable AND as a label-generation pipeline for downstream YOLO
training. Speed is not a concern (~1-3 minutes per tree on a 4090);
accuracy is the dominant objective.

Three independent LAI estimates per tree, fused at the end
-----------------------------------------------------------
1. **Voxel LAI** — primary estimate. Multi-view canopy points are
   accumulated into a per-tree voxel grid; total leaf area = surface
   voxel count × per-voxel area, calibrated empirically against
   LAI-2200C ground truth. This is closest to what the 2200C
   physically measures (3D leaf area per ground area) and is robust
   to clumping, view angle, and partial occlusion.

2. **Beer-Lambert LAI** — per-frame gap-fraction-based, averaged.
   LAI = -ln(P_gap) / k, where P_gap = gap_pixels / canopy_pixels
   inside the tree silhouette. Useful as an independent cross-check
   and as a fast online proxy. Needs a clumping correction for
   trellised apple (Ω ≈ 0.7-0.85).

3. **Calibrated fusion** — linear blend of (1) + (2), coefficients
   fit against 2200C ground truth on a calibration subset. This is
   the value to use for downstream analysis / publication.

The hierarchical SAM2 pass that drives all three estimates also
produces fine-grained per-pixel labels (trunk, leaf, branch, gap,
fruit) that export directly to YOLO segmentation format for later
training of a real-time deployable model.

Pipeline
--------
1. ``segment_tree_subregions``     — SAM2 auto-mask within tree silhouette
2. ``classify_subregions``         — leaf/branch/gap/fruit classifier
3. ``backproject_to_world``        — 2D leaf pixels -> 3D world points
4. ``register_frames_with_icp``    — refine GPS pose using trunk points
5. ``aggregate_tree_pointcloud``   — fuse multi-view per-tree clouds
6. ``voxelize_and_estimate``       — voxel surface area -> LAI
7. ``gap_fraction_lai_per_frame``  — Beer-Lambert per frame
8. ``calibrate_against_2200c``     — fit fusion coefficients
9. ``export_yolo_labels``          — write training labels per frame

Setup (additional to sam2_orchard_segmenter.py)
-----------------------------------------------
    pip install open3d scikit-image
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .sam2_orchard_segmenter import (
        All2023FrameLoader,
        FrameLoader,
        PNGSequenceLoader,
        SegmenterConfig,
        TreeCluster,
        TrunkDetection,
        camera_to_world,
        haversine_m,
    )
except ImportError:
    from sam2_orchard_segmenter import (   # type: ignore[no-redef]
        All2023FrameLoader,
        FrameLoader,
        PNGSequenceLoader,
        SegmenterConfig,
        TreeCluster,
        TrunkDetection,
        camera_to_world,
        haversine_m,
    )


log = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class CameraIntrinsics:
    """RealSense D455 RGB intrinsics at 1280x720 (defaults).

    Pull these from your specific camera's calibration via
    ``rs.video_stream_profile.get_intrinsics()`` for best accuracy.
    A 5% intrinsics error becomes ~5% LAI error in the voxel branch.
    """
    fx: float = 644.0
    fy: float = 644.0
    cx: float = 644.0
    cy: float = 360.0
    width: int = 1280
    height: int = 720


@dataclass
class LAIConfig:
    """All tuneable parameters for the LAI estimation pipeline."""

    # ── Camera ────────────────────────────────────────────────────
    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    # Camera mount geometry: position relative to GPS antenna (metres),
    # in tractor frame (X forward, Y left, Z up). Used to transform
    # camera-frame points into world coords.
    cam_offset_forward_m: float = 0.0
    cam_offset_left_m: float = 0.0
    cam_offset_up_m: float = 1.5    # typical sprayer mast height
    # Camera tilt relative to horizontal (positive = looking up).
    cam_pitch_deg: float = 0.0

    # ── Hierarchical SAM2 (auto-mask within tree silhouette) ─────
    sam2_model_id: str = "facebook/sam2.1-hiera-large"
    sam2_points_per_side: int = 32     # 32x32 = 1024 prompt points
    sam2_pred_iou_thresh: float = 0.70
    sam2_stability_thresh: float = 0.80
    sam2_min_mask_area: int = 100      # pixels

    # ── Sub-region classification ────────────────────────────────
    # Each SAM2 sub-mask gets classified into one of:
    #   0=trunk, 1=leaf, 2=branch, 3=gap, 4=fruit
    # Default classifier uses HSV + depth + NDVI rules. For higher
    # accuracy, train a small CNN on hand-labeled samples and pass
    # it via ``classifier_fn``.
    classifier_fn: Optional[Callable] = None
    # NIR channel availability. D455 exposes IR streams through the
    # SDK; without NIR we use ExG (excess green) instead of NDVI.
    has_nir: bool = True
    # Depth thresholds for gap detection: pixels with depth > this
    # are sky/distant background = gap.
    gap_depth_min_m: float = 5.0       # treat as gap
    # Tree-bounding-volume depth: pixels beyond this (relative to
    # tree centre depth) are background, not part of tree.
    tree_depth_envelope_m: float = 0.8

    # ── Frame-to-frame registration ──────────────────────────────
    # ICP refinement using trunk surface points. Locks tractor-frame
    # camera pose better than GPS+heading alone (GPS heading drifts
    # ~1-2 deg, which becomes ~3-5 cm pose error at 2 m distance --
    # significant at 2 cm voxel resolution).
    use_icp: bool = True
    icp_voxel_size_m: float = 0.02
    icp_max_iter: int = 50
    icp_max_correspondence_m: float = 0.10

    # ── Voxelization ──────────────────────────────────────────────
    voxel_size_m: float = 0.02         # 2 cm voxels
    # A surface voxel = an occupied voxel that has at least one
    # empty neighbour. Counting only surface voxels approximates leaf
    # area better than counting all occupied voxels (which would
    # double-count thick foliage layers).
    surface_neighbour_threshold: int = 1
    # Empirical leaf-area-per-surface-voxel calibration. Default of
    # voxel_size² is the upper bound (assumes each surface voxel = one
    # flat leaf face); typical apple foliage is ~0.6-0.8 of that
    # because of leaf curl / overlap. Calibrate against 2200C.
    leaf_area_per_voxel_factor: float = 0.7

    # ── Beer-Lambert gap-fraction LAI ────────────────────────────
    # Extinction coefficient k. For tractor-side viewing through
    # trellised apple (canopy depth ~1 m, view nearly horizontal),
    # k is closer to 0.4 than the canonical 0.5 used for nadir LAI.
    # CALIBRATE this against 2200C — it's the dominant uncertainty
    # in the Beer-Lambert branch.
    extinction_coef_k: float = 0.4
    # Clumping index Ω: trellised apple is non-random foliage,
    # so true LAI = naive_LAI / Ω. Typical 0.7-0.85 for apple.
    clumping_index: float = 0.78

    # ── Ground projection area (denominator of LAI) ──────────────
    # LAI = leaf_area / ground_area. For a single tree the ground
    # area is ``tree_spacing_m × row_spacing_m`` (the allotted plot
    # per tree). 2200C reports LAI on the same convention.
    tree_spacing_m: float = 3.0
    row_spacing_m: float = 4.0

    # ── Pose / fusion calibration ────────────────────────────────
    # Coefficients for calibrated_LAI = a*voxel + b*beer + d*leafcount + c.
    # Filled in by ``calibrate_against_2200c``; defaults to identity
    # on voxel only.
    fusion_a: float = 1.0
    fusion_b: float = 0.0
    fusion_c: float = 0.0
    fusion_d: float = 0.0

    # ── YOLO label export ────────────────────────────────────────
    yolo_class_names: Tuple[str, ...] = (
        "trunk", "leaf", "branch", "gap", "fruit",
    )
    yolo_polygon_simplify_tol: float = 1.5    # pixels

    # ── Individual leaf counting (third LAI branch) ──────────────
    # Single-leaf area range. Apple leaves are typically 15-80 cm²
    # at maturity; default bounds are wider to capture juvenile and
    # senescent leaves. Tighten per-cultivar from your own data.
    min_leaf_area_m2: float = 5e-4       # 5 cm²
    max_leaf_area_m2: float = 150e-4     # 150 cm²
    # Aspect ratio bounds (long axis / short axis) for single-leaf
    # masks. Multi-leaf clumps and mask fragments fall outside.
    min_leaf_aspect: float = 1.2
    max_leaf_aspect: float = 3.5
    # Depth coherence: single leaves have tight depth distribution
    # (most pixels within this range of the median).
    leaf_depth_std_max_m: float = 0.025
    # Foreshortening correction. When True, estimate leaf normal
    # from the local depth gradient and correct projected area by
    # 1/cos(angle). Adds ~10% compute cost; recovers ~20-30% of
    # systematically-underestimated tilted leaves.
    correct_foreshortening: bool = True
    # Multi-view leaf dedup: same physical leaf seen in N frames
    # collapses to one detection if its centroid falls within this
    # 3D radius. 3 cm catches the same leaf across the tractor pass
    # while keeping adjacent leaves distinct.
    leaf_dedup_radius_m: float = 0.03
    # Visibility correction factor: true_LAI ≈ visible_LAI × this.
    # 2.0 is typical for one-sided viewing; 1.1-1.2 if you fuse
    # opposite-side passes. CALIBRATE against 2200C; this is the
    # dominant uncertainty in the leaf-count branch.
    leaf_visibility_factor: float = 2.0


# ============================================================================
# Data classes
# ============================================================================
@dataclass
class SubMask:
    """One SAM2 sub-mask within a tree silhouette, plus its class."""
    frame_idx: int
    tree_id: int
    mask: np.ndarray           # bool, frame-shape
    class_id: int              # index into yolo_class_names
    depth_median_m: float
    confidence: float


@dataclass
class LeafDetection:
    """One individual-leaf detection from the leaf-count branch."""
    frame_idx: int
    tree_id: int
    mask: np.ndarray
    pixel_area: int
    depth_median_m: float
    depth_std_m: float
    aspect_ratio: float
    # 3D centroid in tree-local ENU coords (anchored at cluster centroid).
    world_e_m: float = 0.0
    world_n_m: float = 0.0
    world_u_m: float = 0.0
    # Projected area in m² from camera intrinsics + depth.
    projected_area_m2: float = 0.0
    # Foreshortening-corrected (true) area in m².
    true_area_m2: float = 0.0
    # Estimated angle between leaf normal and viewing axis (radians).
    view_angle_rad: float = 0.0
    confidence: float = 0.0


@dataclass
class TreeLAIResult:
    """All LAI estimates for one tree, plus the underlying evidence."""
    tree_id: int
    n_frames: int
    n_leaf_points: int
    voxel_lai: float
    beer_lambert_lai_per_frame: List[float]
    beer_lambert_lai_mean: float
    beer_lambert_lai_median: float
    # Leaf-count branch.
    leaf_count_lai: float = float("nan")
    n_leaves_detected: int = 0
    mean_leaf_area_m2: float = float("nan")
    leaf_area_m2_p25: float = float("nan")
    leaf_area_m2_p75: float = float("nan")
    visible_leaf_area_m2: float = float("nan")
    calibrated_lai: float = float("nan")
    # Per-tree leaf surface area in m² (from voxel branch,
    # pre-division by ground area).
    leaf_area_m2: float = 0.0
    # Quality flags.
    notes: List[str] = field(default_factory=list)


# ============================================================================
# Stage 1: Hierarchical SAM2 sub-segmentation within tree silhouette
# ============================================================================
def segment_tree_subregions(
    cluster: TreeCluster,
    loader: FrameLoader,
    cfg: LAIConfig,
    device: str = "cuda",
) -> Dict[int, List[np.ndarray]]:
    """SAM2 automatic mask generation, restricted to each tree's silhouette.

    For every frame the tree is visible, take its silhouette mask
    (from the trunk segmenter's output), bound it tight to a crop,
    and run SAM2 auto-mask generation inside that crop. Returns a
    dict ``{frame_idx: [mask, mask, ...]}`` of all sub-masks found
    for this tree.

    Each sub-mask is a bool array in *full-frame* coordinates so it
    composes cleanly with the trunk segmenter outputs.
    """
    import torch
    from PIL import Image as _PILImage
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from tqdm.auto import tqdm

    # SAM 3 doesn't ship an automatic mask generator. Equivalent
    # behaviour for our use case (sub-mask coverage of a tree's
    # silhouette into leaf-/branch-/etc. instances) is obtained by
    # text-prompting "leaf" + "branch" + "apple" inside the same
    # crop -- SAM 3 returns one mask per detected instance, which is
    # what classify_subregions wants.
    log.info("Loading SAM 3 image model for hierarchical sub-segmentation")
    sam3_model = build_sam3_image_model()
    processor = Sam3Processor(
        sam3_model, device=device, confidence_threshold=0.10,
    )

    submasks_by_frame: Dict[int, List[np.ndarray]] = {}

    autocast_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    # Prompts cover the things classify_subregions cares about plus
    # generic "leaf" so dense canopy gets sliced into per-leaf chunks
    # for the leaf-count branch. "twig" / "stem" are deliberately
    # left out -- they overlap with branch and add noise.
    sub_prompts = ("leaf", "leaves", "branch", "apple", "fruit")

    # Get the per-frame silhouette masks for this tree from its tracks.
    # Prefer the whole-tree (canopy) mask when available -- the trunk
    # mask alone is too narrow for hierarchical sub-segmentation and
    # produces nonsense gap fractions (all-gap or all-foliage in the
    # tiny strip).
    silhouette_by_frame: Dict[int, np.ndarray] = {}
    for track in cluster.tracks:
        for det in track.detections:
            if det.frame_idx not in cluster.frame_pixels:
                continue
            sil = (det.tree_mask
                   if (det.tree_mask is not None
                       and det.tree_mask.any())
                   else det.mask)
            if sil is None:
                continue
            m = silhouette_by_frame.get(det.frame_idx)
            silhouette_by_frame[det.frame_idx] = (
                sil if m is None else (m | sil)
            )

    for frame_idx, silhouette in tqdm(
        silhouette_by_frame.items(),
        desc=f"Tree {cluster.tree_id} sub-seg",
        leave=False,
    ):
        if not silhouette.any():
            continue
        ys, xs = np.where(silhouette)
        # Pad bbox by 20 px so SAM2 auto-mask sees a bit of context
        # at the silhouette edge (helps it segment edge gaps).
        pad = 20
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(silhouette.shape[0], int(ys.max()) + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(silhouette.shape[1], int(xs.max()) + pad)

        rgb = loader.load_rgb(frame_idx)
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        full = []
        with torch.autocast(
            "cuda" if device.startswith("cuda") else "cpu",
            dtype=autocast_dtype,
        ):
            state = processor.set_image(_PILImage.fromarray(crop))
            for prompt in sub_prompts:
                processor.reset_all_prompts(state)
                try:
                    state = processor.set_text_prompt(
                        prompt=prompt, state=state,
                    )
                except Exception as exc:
                    log.debug("SAM 3 text %r failed on frame %d: %s",
                              prompt, frame_idx, exc)
                    continue
                masks = state.get("masks")
                if masks is None or len(masks) == 0:
                    continue
                arr = masks.detach().cpu()
                if arr.dtype != torch.bool:
                    arr = arr.to(torch.bool)
                arr = arr.numpy()
                if arr.ndim == 4:
                    arr = arr.squeeze(1)
                for i in range(arr.shape[0]):
                    crop_mask = arr[i].astype(bool)
                    full_mask = np.zeros_like(silhouette, dtype=bool)
                    full_mask[y0:y1, x0:x1] = crop_mask
                    inside = full_mask & silhouette
                    if inside.sum() < cfg.sam2_min_mask_area:
                        continue
                    full.append(inside)
        submasks_by_frame[frame_idx] = full

    n_total = sum(len(v) for v in submasks_by_frame.values())
    log.info("Tree %d: %d sub-masks across %d frames",
             cluster.tree_id, n_total, len(submasks_by_frame))
    return submasks_by_frame


# ============================================================================
# Stage 2: Sub-region classification (trunk / leaf / branch / gap / fruit)
# ============================================================================
def _default_classifier(
    submask: np.ndarray,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    nir: Optional[np.ndarray],
    cfg: LAIConfig,
    tree_centre_depth_m: float,
) -> Tuple[int, float]:
    """Rule-based foliage classifier.

    Inputs
    ------
    submask : bool, frame-shape — the SAM2 sub-mask
    rgb     : HxWx3 uint8
    depth_m : HxW float, NaN for invalid
    nir     : HxW float (0-1), or None
    cfg     : LAIConfig
    tree_centre_depth_m : reference depth of this tree's trunk

    Decision tree (in priority order, first match wins):
      1. Depth >> tree_centre_depth or invalid -> GAP
      2. Very low saturation + medium value + brown hue -> BRANCH
      3. NDVI > 0.4 (or ExG > threshold) + saturated green -> LEAF
      4. Red hue + high saturation -> FRUIT
      5. Brown hue + low position in frame + thin shape -> TRUNK
      6. Default -> LEAF (most pixels in canopy are leaves)

    Returns (class_id, confidence in [0, 1]).
    """
    import cv2

    if submask.sum() == 0:
        return 3, 0.0   # GAP, no confidence

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.float32)     # 0-179 in OpenCV
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0

    valid_depth = np.isfinite(depth_m) & (depth_m > 0.1) & (depth_m < 20.0)
    inside = submask & valid_depth

    # Step 1: depth-based gap detection.
    far_or_invalid = submask & (
        ~valid_depth
        | (depth_m > (tree_centre_depth_m + cfg.tree_depth_envelope_m))
    )
    if far_or_invalid.sum() > 0.7 * submask.sum():
        return 3, 0.95   # high-confidence GAP

    if inside.sum() == 0:
        return 3, 0.5

    h_mean = float(np.mean(h[inside]))
    s_mean = float(np.mean(s[inside]))
    v_mean = float(np.mean(v[inside]))

    # Step 2: NDVI (or ExG fallback) over the same pixels.
    if nir is not None:
        red = rgb[..., 0].astype(np.float32) / 255.0
        n = nir.astype(np.float32)
        if n.max() > 1.5:                  # rescale 16-bit
            n = n / 65535.0 if n.max() > 255 else n / 255.0
        ndvi = (n - red) / np.maximum(n + red, 1e-6)
        veg_idx = float(np.mean(ndvi[inside]))
        veg_thresh = 0.40
    else:
        # Excess Green: 2G - R - B, normalised
        rgbf = rgb.astype(np.float32) / 255.0
        exg = 2.0 * rgbf[..., 1] - rgbf[..., 0] - rgbf[..., 2]
        veg_idx = float(np.mean(exg[inside]))
        veg_thresh = 0.05

    # Step 3: branch — low saturation, brown hue (10-25 in OpenCV HSV),
    # depth consistent with tree.
    if s_mean < 0.25 and 5.0 < h_mean < 25.0 and v_mean < 0.55:
        return 2, 0.85   # BRANCH

    # Step 4: leaf — vegetation index high.
    if veg_idx > veg_thresh:
        # Confidence rises with vegetation strength.
        conf = float(np.clip((veg_idx - veg_thresh) / 0.3 + 0.5, 0.5, 0.99))
        return 1, conf   # LEAF

    # Step 5: fruit — red/yellow hue, high saturation. Apple skin
    # ranges from green to red depending on cultivar / ripeness.
    # Conservative: only call fruit on clearly red regions.
    if (h_mean < 12.0 or h_mean > 165.0) and s_mean > 0.45 and v_mean > 0.35:
        return 4, 0.7    # FRUIT

    # Step 6: trunk — thin vertical shape, brown hue, low in frame.
    ys, xs = np.where(submask)
    if ys.size > 0:
        y_centre = float(np.mean(ys)) / submask.shape[0]
        bbox_h = float(ys.max() - ys.min())
        bbox_w = float(xs.max() - xs.min()) if xs.size > 0 else 1.0
        aspect = bbox_h / max(bbox_w, 1.0)
        if (aspect > 2.0 and y_centre > 0.4 and
                s_mean < 0.35 and 5.0 < h_mean < 30.0):
            return 0, 0.80   # TRUNK

    # Default: weak leaf (probably leaf in shadow / glare).
    return 1, 0.40


def classify_subregions(
    submasks_by_frame: Dict[int, List[np.ndarray]],
    cluster: TreeCluster,
    loader: FrameLoader,
    cfg: LAIConfig,
) -> Dict[int, List[SubMask]]:
    """Apply the foliage classifier to every sub-mask.

    Returns ``{frame_idx: [SubMask, SubMask, ...]}``. The classifier
    used is ``cfg.classifier_fn`` if set, else the rule-based default.
    For maximum accuracy, train a small CNN on hand-labeled data
    from the contact sheets (the SubMask outputs of this function
    are exactly the right input format for that training step).
    """
    from tqdm.auto import tqdm

    # Per-frame tree-centre depth, for the gap test.
    tree_centre_depth: Dict[int, float] = {}
    for track in cluster.tracks:
        for det in track.detections:
            if det.depth_m > 0:
                tree_centre_depth[det.frame_idx] = det.depth_m

    classifier = cfg.classifier_fn or _default_classifier
    out: Dict[int, List[SubMask]] = {}

    for frame_idx, masks in tqdm(
        submasks_by_frame.items(),
        desc=f"Tree {cluster.tree_id} classify",
        leave=False,
    ):
        rgb = loader.load_rgb(frame_idx)
        depth = loader.load_depth_m(frame_idx)
        nir = None
        if cfg.has_nir:
            # Preferred path: loader exposes a load_nir() method
            # (All2023FrameLoader reads IR/<base>-IR.bmp). Fallback:
            # an explicit "nir_path" entry in the per-frame metadata.
            load_nir = getattr(loader, "load_nir", None)
            if callable(load_nir):
                try:
                    nir = load_nir(frame_idx)
                except Exception:                                  # noqa: BLE001
                    nir = None
            if nir is None:
                try:
                    meta = loader.load_meta(frame_idx)
                    nir_path = meta.get("nir_path")
                    if nir_path and os.path.exists(nir_path):
                        nir = np.load(nir_path) if nir_path.endswith(".npy") \
                            else None
                except Exception:                                  # noqa: BLE001
                    nir = None

        ref_depth = tree_centre_depth.get(frame_idx, 2.5)
        per_frame: List[SubMask] = []
        for m in masks:
            class_id, conf = classifier(m, rgb, depth, nir, cfg, ref_depth)
            valid = m & np.isfinite(depth) & (depth > 0.1) & (depth < 20.0)
            d_med = float(np.median(depth[valid])) if valid.any() else float("nan")
            per_frame.append(SubMask(
                frame_idx=frame_idx,
                tree_id=cluster.tree_id,
                mask=m,
                class_id=class_id,
                depth_median_m=d_med,
                confidence=conf,
            ))
        out[frame_idx] = per_frame
    return out


# ============================================================================
# Stage 3: 2D leaf pixels -> 3D world points
# ============================================================================
def _camera_to_world_se3(
    gps_lat: float,
    gps_lon: float,
    heading_deg: float,
    cfg: LAIConfig,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Build a 4x4 transform from camera frame to local world ENU.

    Returns (T_4x4, (lat0, lon0)) where the matrix maps a point in
    *camera coordinates* (X right, Y down, Z forward — OpenCV
    convention) into a *local ENU frame* anchored at (lat0, lon0).
    Caller turns ENU back into (lat, lon) at the end of the run.

    ENU axes here are (East, North, Up), so the camera's optical
    axis Z points along the heading direction in the horizontal
    plane after we apply the heading rotation and pitch.
    """
    # Heading: 0 = North, 90 = East. Rotate camera so its +Z aligns
    # with that heading direction in ENU.
    h = math.radians(heading_deg)
    p = math.radians(cfg.cam_pitch_deg)

    # Camera-to-ENU rotation: first un-pitch (rotate about camera X),
    # then yaw to heading (rotate about ENU up axis).
    # Camera convention: X right, Y down, Z forward.
    # ENU convention: X east, Y north, Z up.
    # So before yaw: (cam +X) -> ENU east-ish, (cam -Y) -> ENU up-ish,
    # (cam +Z) -> ENU forward-ish (depends on heading).
    R_pitch = np.array([
        [1, 0, 0],
        [0, math.cos(p), math.sin(p)],
        [0, -math.sin(p), math.cos(p)],
    ])
    # Map camera (X right, -Y up, Z forward) into a tractor frame
    # with (forward, left, up):
    R_cam_to_tractor = np.array([
        [0, 0, 1],     # tractor forward = camera Z
        [-1, 0, 0],    # tractor left = -camera X
        [0, -1, 0],    # tractor up = -camera Y
    ])
    # Tractor (forward, left, up) -> ENU (east, north, up) by heading.
    cos_h, sin_h = math.cos(h), math.sin(h)
    R_tractor_to_enu = np.array([
        [sin_h, cos_h, 0],    # east
        [cos_h, -sin_h, 0],   # north (heading 0 = +north)
        [0, 0, 1],            # up
    ])
    R = R_tractor_to_enu @ R_cam_to_tractor @ R_pitch

    # Translation: camera origin in ENU. Camera mounts forward of GPS
    # antenna by cam_offset_forward, left by cam_offset_left, up by
    # cam_offset_up_m. Convert that tractor offset to ENU.
    t_tractor = np.array([
        cfg.cam_offset_forward_m,
        cfg.cam_offset_left_m,
        cfg.cam_offset_up_m,
    ])
    t_enu = R_tractor_to_enu @ t_tractor

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t_enu
    return T, (gps_lat, gps_lon)


def backproject_to_world(
    submasks: List[SubMask],
    loader: FrameLoader,
    cfg: LAIConfig,
    target_classes: Tuple[int, ...] = (1,),       # default: leaves only
    lat0: float = 0.0,
    lon0: float = 0.0,
    icp_corrections: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    """Convert classified pixels into a world-coordinate point cloud.

    Returns an (N, 4) array of (east_m, north_m, up_m, class_id),
    coordinates in the local ENU frame anchored at (lat0, lon0).

    By default only leaf pixels are projected (target_classes=(1,)).
    Pass (0, 1, 2, 4) to also export trunk/branch/fruit -- useful for
    QA visualization in CloudCompare or Open3D.
    """
    if not submasks:
        return np.zeros((0, 4), dtype=np.float32)

    intr = cfg.intrinsics
    # Lazily build a pixel-grid (u, v) of camera-frame ray directions
    # for the full frame; we slice by mask.
    u, v = np.meshgrid(
        np.arange(intr.width, dtype=np.float32),
        np.arange(intr.height, dtype=np.float32),
    )
    x_norm = (u - intr.cx) / intr.fx
    y_norm = (v - intr.cy) / intr.fy

    cos_lat0 = math.cos(math.radians(lat0)) or 1e-6

    points: List[np.ndarray] = []
    classes: List[np.ndarray] = []

    # Group submasks by frame to amortize meta + depth load.
    by_frame: Dict[int, List[SubMask]] = {}
    for sm in submasks:
        if sm.class_id in target_classes:
            by_frame.setdefault(sm.frame_idx, []).append(sm)

    for frame_idx, sms in by_frame.items():
        depth = loader.load_depth_m(frame_idx)
        meta = loader.load_meta(frame_idx)
        # GPS antenna position in ENU (relative to lat0, lon0).
        gps_lat = float(meta["gps_lat"])
        gps_lon = float(meta["gps_lon"])
        heading = float(meta["heading_deg"])
        ant_e = (gps_lon - lon0) * 111_320.0 * cos_lat0
        ant_n = (gps_lat - lat0) * 111_320.0
        ant_u = 0.0
        T, _ = _camera_to_world_se3(gps_lat, gps_lon, heading, cfg)
        T[:3, 3] += np.array([ant_e, ant_n, ant_u])
        # Apply ICP refinement to this frame's pose if available.
        if icp_corrections is not None:
            corr = icp_corrections.get(frame_idx)
            if corr is not None:
                T = corr @ T

        for sm in sms:
            sel = sm.mask & np.isfinite(depth) & (depth > 0.1) & (depth < 20.0)
            if not sel.any():
                continue
            z = depth[sel]
            xn = x_norm[sel]
            yn = y_norm[sel]
            # Camera-frame 3D points.
            X = xn * z
            Y = yn * z
            Z = z
            cam_pts = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)   # (N, 4)
            world_pts = cam_pts @ T.T                                # (N, 4)
            points.append(world_pts[:, :3].astype(np.float32))
            classes.append(np.full(z.size, sm.class_id, dtype=np.int32))

    if not points:
        return np.zeros((0, 4), dtype=np.float32)
    pts = np.concatenate(points, axis=0)
    cls = np.concatenate(classes, axis=0)
    return np.concatenate([pts, cls[:, None].astype(np.float32)], axis=1)


# ============================================================================
# Stage 4: ICP registration refinement using trunk surface points
# ============================================================================
def register_frames_with_icp(
    cluster: TreeCluster,
    loader: FrameLoader,
    cfg: LAIConfig,
) -> Dict[int, np.ndarray]:
    """Per-frame 4x4 correction transform, anchored on trunk geometry.

    GPS heading drift (~1-2 deg) and integration error in tractor
    speed produce ~3-5 cm pose error frame-to-frame. At 2 cm voxel
    resolution that smears leaf surfaces noticeably. ICP between
    consecutive trunk point clouds (which are well-defined cylinders,
    so highly informative for registration) refines the pose.

    Returns ``{frame_idx: T_correction_4x4}``. Apply by left-
    multiplying the GPS-derived camera->world transform with this
    correction.

    Falls back to identity transforms if Open3D is unavailable.
    """
    if not cfg.use_icp:
        return {fid: np.eye(4) for fid in cluster.frame_pixels}

    try:
        import open3d as o3d
    except ImportError:
        log.warning("Open3D not available -- skipping ICP refinement")
        return {fid: np.eye(4) for fid in cluster.frame_pixels}

    from tqdm.auto import tqdm

    # Build per-frame trunk point clouds.
    intr = cfg.intrinsics
    u, v = np.meshgrid(
        np.arange(intr.width, dtype=np.float32),
        np.arange(intr.height, dtype=np.float32),
    )
    x_norm = (u - intr.cx) / intr.fx
    y_norm = (v - intr.cy) / intr.fy

    sorted_frames = sorted(cluster.frame_pixels.keys())
    trunk_clouds: Dict[int, np.ndarray] = {}
    for track in cluster.tracks:
        for det in track.detections:
            if det.mask is None:
                continue
            depth = loader.load_depth_m(det.frame_idx)
            sel = det.mask & np.isfinite(depth) & (depth > 0.1) & (depth < 20.0)
            if sel.sum() < 50:
                continue
            z = depth[sel]
            X = x_norm[sel] * z
            Y = y_norm[sel] * z
            cloud = np.stack([X, Y, z], axis=1).astype(np.float32)
            trunk_clouds[det.frame_idx] = cloud

    transforms: Dict[int, np.ndarray] = {sorted_frames[0]: np.eye(4)}
    prev_T = np.eye(4)

    for i in tqdm(range(1, len(sorted_frames)),
                  desc=f"Tree {cluster.tree_id} ICP", leave=False):
        fid = sorted_frames[i]
        prev_fid = sorted_frames[i - 1]
        if fid not in trunk_clouds or prev_fid not in trunk_clouds:
            transforms[fid] = prev_T
            continue
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(trunk_clouds[fid])
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(trunk_clouds[prev_fid])

        src_d = src.voxel_down_sample(cfg.icp_voxel_size_m)
        tgt_d = tgt.voxel_down_sample(cfg.icp_voxel_size_m)
        result = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d, cfg.icp_max_correspondence_m,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=cfg.icp_max_iter,
            ),
        )
        T_step = np.asarray(result.transformation, dtype=np.float64)
        # Compose with previous to get cumulative correction.
        prev_T = prev_T @ T_step
        transforms[fid] = prev_T

    return transforms


# ============================================================================
# Stage 5: Aggregate per-tree multi-view point cloud
# ============================================================================
def aggregate_tree_pointcloud(
    cluster: TreeCluster,
    submasks_by_frame: Dict[int, List[SubMask]],
    loader: FrameLoader,
    cfg: LAIConfig,
    icp_corrections: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    """Build the unified leaf point cloud for one tree.

    Returns an (N, 3) float32 array in *tree-local* metric coordinates
    (origin at the cluster centroid, axes ENU).

    Multi-view fusion is what gives the voxel-LAI branch its accuracy:
    a leaf observed from one frame is a single 3D point with depth
    noise; the same leaf observed across 8-15 frames as the tractor
    passes is a tight 3D cluster that voxelizes correctly.
    """
    # Prefer projecting the whole canopy silhouette per frame --
    # SAM 3 with text "leaf" returns coarse all-leaves masks rather
    # than per-leaf instances, so the leaf-class sub-mask path
    # produces 0 points on sparse-canopy data.
    #
    # If the segmenter already populated det.tree_mask, use it. If
    # not, compute a per-detection canopy mask here using
    # build_tree_mask + horizontal-column restriction around the
    # trunk centroid -- guarantees a real canopy mask reaches
    # backproject_to_world even when the segmenter's tree-mask
    # association failed for this detection.
    _build_tree_mask = None
    try:
        from tree_mask import build_tree_mask as _build_tree_mask
    except ImportError as exc:
        log.warning(
            "tree_mask import failed for tree %d (%s); on-demand canopy "
            "fallback disabled", cluster.tree_id, exc,
        )

    n_used_existing = n_used_built = n_used_trunk = n_skipped = 0
    silhouette_subs: List[SubMask] = []
    canopy_column_half_px = 60
    for track in cluster.tracks:
        for det in track.detections:
            if det.frame_idx not in cluster.frame_pixels:
                continue
            if det.tree_mask is not None and det.tree_mask.any():
                sil = det.tree_mask
                n_used_existing += 1
            else:
                sil = None
            if sil is None and _build_tree_mask is not None:
                # Build canopy on demand from depth + RGB.
                try:
                    depth = loader.load_depth_m(det.frame_idx)
                    rgb = loader.load_rgb(det.frame_idx)
                    if (depth is not None and rgb is not None
                            and np.isfinite(depth).any()):
                        depth_mm = np.where(
                            np.isfinite(depth) & (depth > 0),
                            (depth * 1000.0).astype(np.uint16),
                            np.zeros_like(depth, dtype=np.uint16),
                        )
                        canopy_u8 = _build_tree_mask(
                            depth_mm, rgb=rgb,
                            roi_cols=(0, depth_mm.shape[1] - 1),
                        )
                        canopy = (canopy_u8 > 0)
                        if canopy.any():
                            # Restrict to a column band centred on the
                            # trunk so adjacent trees don't pollute
                            # this cluster's canopy point cloud.
                            x1, _, x2, _ = det.bbox_xyxy
                            cx = int(round((x1 + x2) * 0.5))
                            x_lo = max(0, cx - canopy_column_half_px)
                            x_hi = min(canopy.shape[1],
                                       cx + canopy_column_half_px)
                            band = np.zeros_like(canopy)
                            band[:, x_lo:x_hi] = True
                            sil = canopy & band
                            if sil.any():
                                n_used_built += 1
                            else:
                                sil = None
                except Exception as exc:
                    log.debug("Canopy compute failed for tree %d frame %d: %s",
                              cluster.tree_id, det.frame_idx, exc)
            if sil is None and det.mask is not None and det.mask.any():
                sil = det.mask
                n_used_trunk += 1
            if sil is None or not sil.any():
                n_skipped += 1
                continue
            silhouette_subs.append(SubMask(
                frame_idx=det.frame_idx,
                tree_id=cluster.tree_id,
                mask=sil,
                class_id=1,                # treat as foliage
                depth_median_m=float(det.depth_m or 0.0),
                confidence=1.0,
            ))

    log.info(
        "Tree %d aggregator: silhouettes built — existing=%d, "
        "computed=%d, trunk-fallback=%d, skipped=%d",
        cluster.tree_id, n_used_existing, n_used_built,
        n_used_trunk, n_skipped,
    )
    if silhouette_subs:
        pts4 = backproject_to_world(
            silhouette_subs, loader, cfg,
            target_classes=(1,),
            lat0=cluster.world_lat,
            lon0=cluster.world_lon,
            icp_corrections=icp_corrections,
        )
        log.info(
            "Tree %d aggregator: %d silhouette sub-masks -> %d 3D points",
            cluster.tree_id, len(silhouette_subs), pts4.shape[0],
        )
        if pts4.size > 0:
            return pts4[:, :3].astype(np.float32)

    # Fallback: classified leaf sub-masks (original path).
    flat_subs: List[SubMask] = []
    for fid, sms in submasks_by_frame.items():
        flat_subs.extend(sm for sm in sms if sm.class_id == 1)
    if not flat_subs:
        return np.zeros((0, 3), dtype=np.float32)
    pts4 = backproject_to_world(
        flat_subs, loader, cfg,
        target_classes=(1,),
        lat0=cluster.world_lat,
        lon0=cluster.world_lon,
        icp_corrections=icp_corrections,
    )
    if pts4.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return pts4[:, :3].astype(np.float32)


# ============================================================================
# Stage 6: Voxelization -> leaf area -> LAI
# ============================================================================
def voxelize_and_estimate(
    points: np.ndarray,
    cfg: LAIConfig,
) -> Tuple[float, float, int, int]:
    """Voxelize the leaf cloud and estimate leaf area and LAI.

    Returns (lai, leaf_area_m2, n_occupied_voxels, n_surface_voxels).

    Surface voxels (occupied voxels with >= 1 empty neighbour) are
    used as the leaf-area proxy because counting all occupied voxels
    double-counts thick foliage layers — a single broad leaf only
    occupies ~1-2 voxels in thickness, but a clump of stacked leaves
    can occupy many voxels in depth even though the projected leaf
    area is similar to the broad-leaf case.

    Calibration: leaf_area = n_surface_voxels * voxel_size² * factor.
    The factor is empirical; calibrate against LAI-2200C.
    """
    if points.size == 0:
        return 0.0, 0.0, 0, 0

    v = cfg.voxel_size_m
    # Quantize points to voxel indices.
    keys = np.round(points / v).astype(np.int32)
    occupied = set(map(tuple, keys.tolist()))
    n_occ = len(occupied)

    # Surface voxel test: an occupied voxel with >= threshold empty
    # 6-neighbours.
    n_surface = 0
    neighbours = ((1, 0, 0), (-1, 0, 0),
                  (0, 1, 0), (0, -1, 0),
                  (0, 0, 1), (0, 0, -1))
    for vox in occupied:
        empty = 0
        for dx, dy, dz in neighbours:
            if (vox[0] + dx, vox[1] + dy, vox[2] + dz) not in occupied:
                empty += 1
                if empty >= cfg.surface_neighbour_threshold:
                    n_surface += 1
                    break

    leaf_area_m2 = n_surface * (v * v) * cfg.leaf_area_per_voxel_factor
    ground_area_m2 = cfg.tree_spacing_m * cfg.row_spacing_m
    lai = leaf_area_m2 / max(ground_area_m2, 1e-6)
    return lai, leaf_area_m2, n_occ, n_surface


# ============================================================================
# Stage 7: Beer-Lambert per-frame gap-fraction LAI
# ============================================================================
def gap_fraction_lai_per_frame(
    submasks_by_frame: Dict[int, List[SubMask]],
    cfg: LAIConfig,
    silhouette_by_frame: Optional[Dict[int, np.ndarray]] = None,
) -> List[float]:
    """Compute Beer-Lambert LAI per frame from gap fraction.

        LAI = -ln(P_gap) / k   (then divided by Ω for clumping correction)

    Two modes, picked automatically:

    (A) If ``silhouette_by_frame`` is provided, gap fraction is the
        fraction of the silhouette's bounding column that is NOT
        canopy. This is robust because the silhouette itself comes
        from build_tree_mask + leaf-colour augmentation, both of
        which are depth-validated. No reliance on SAM-sub-mask
        classification accuracy.

            p_gap = 1 - (canopy_pixels / bbox_pixels)
                  = 1 - canopy_fraction

        This is the right path when sub-mask classification is
        unreliable (e.g. thin saplings where most pixels have no
        depth, so classify_subregions tags everything as GAP).

    (B) Otherwise fall back to the original sub-mask-classifier
        path: ``p_gap = n_gap / (n_leaf + n_branch + n_gap)``.
        Kept for backwards compatibility and for users with a well-
        trained leaf classifier.

    Returns a list of per-frame LAI estimates (one per frame the
    tree was visible).
    """
    out: List[float] = []

    if silhouette_by_frame:
        for fid, sil in silhouette_by_frame.items():
            if sil is None or not sil.any():
                continue
            ys, xs = np.where(sil)
            x_lo = int(xs.min()); x_hi = int(xs.max()) + 1
            y_lo = int(ys.min()); y_hi = int(ys.max()) + 1
            bbox_pixels = (y_hi - y_lo) * (x_hi - x_lo)
            if bbox_pixels < 100:
                continue
            canopy_pixels = int(
                sil[y_lo:y_hi, x_lo:x_hi].sum()
            )
            cf = canopy_pixels / float(bbox_pixels)
            p_gap = max(1e-3, min(0.99, 1.0 - cf))
            lai_naive = -math.log(p_gap) / cfg.extinction_coef_k
            lai = lai_naive / max(cfg.clumping_index, 1e-3)
            out.append(float(lai))
        return out

    # Sub-mask-classifier mode (original).
    for fid, sms in submasks_by_frame.items():
        n_leaf = sum(int(sm.mask.sum()) for sm in sms if sm.class_id == 1)
        n_branch = sum(int(sm.mask.sum()) for sm in sms if sm.class_id == 2)
        n_gap = sum(int(sm.mask.sum()) for sm in sms if sm.class_id == 3)
        n_canopy = n_leaf + n_branch + n_gap
        if n_canopy < 100:
            continue
        p_gap = max(1e-3, min(0.99, n_gap / n_canopy))
        lai_naive = -math.log(p_gap) / cfg.extinction_coef_k
        lai = lai_naive / max(cfg.clumping_index, 1e-3)
        out.append(float(lai))
    return out


# ============================================================================
# Stage 8: Calibration against LAI-2200C ground truth
# ============================================================================
def calibrate_against_2200c(
    voxel_lai: Sequence[float],
    beer_lai: Sequence[float],
    leaf_lai: Sequence[float],
    truth_lai: Sequence[float],
) -> Tuple[float, float, float, float, Dict[str, float]]:
    """Fit ``calibrated = a*voxel + b*beer + d*leafcount + c`` against 2200C.

    Returns (a, b, c, d, metrics). ``metrics`` reports RMSE, R²,
    and per-component contributions for the paper.

    The calibration auto-selects model complexity by sample size:
      * n >= 12 -> full 4-parameter fit (a, b, c, d)
      * n >=  8 -> 3-parameter fit on voxel + beer (d=0, leaf-count
                   ignored -- it's the noisiest predictor and needs
                   the most data to fit reliably)
      * n >=  3 -> 50/50 voxel/beer blend, fit offset only
      * n <  3 -> identity on voxel
    """
    v = np.asarray(voxel_lai, dtype=np.float64)
    b = np.asarray(beer_lai, dtype=np.float64)
    l = np.asarray(leaf_lai, dtype=np.float64)
    y = np.asarray(truth_lai, dtype=np.float64)
    n = len(y)

    if n >= 12:
        X = np.stack([v, b, l, np.ones_like(v)], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, bcoef, dcoef, c = (float(coef[0]), float(coef[1]),
                              float(coef[2]), float(coef[3]))
    elif n >= 8:
        X = np.stack([v, b, np.ones_like(v)], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, bcoef, c = float(coef[0]), float(coef[1]), float(coef[2])
        dcoef = 0.0
    elif n >= 3:
        a, bcoef, dcoef = 0.5, 0.5, 0.0
        c = float(np.mean(y - 0.5 * v - 0.5 * b))
    else:
        a, bcoef, dcoef, c = 1.0, 0.0, 0.0, 0.0

    pred = a * v + bcoef * b + dcoef * l + c
    res = y - pred
    rmse = float(np.sqrt(np.mean(res ** 2)))
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    return a, bcoef, c, dcoef, {
        "rmse": rmse,
        "r2": r2,
        "n_calibration": float(n),
        "voxel_lai_mean": float(v.mean()),
        "beer_lai_mean": float(b.mean()) if np.isfinite(b).all() else float("nan"),
        "leaf_lai_mean": float(l.mean()) if np.isfinite(l).all() else float("nan"),
        "truth_lai_mean": float(y.mean()),
    }


# ============================================================================
# Stage 9: YOLO-format label export
# ============================================================================
def _mask_to_polygon(
    mask: np.ndarray,
    simplify_tol: float,
) -> List[Tuple[float, float]]:
    """Convert a binary mask to a simplified polygon (CCW, normalised).

    Uses OpenCV findContours then Douglas-Peucker simplification.
    Returns a list of (x_norm, y_norm) tuples in [0, 1] for direct
    YOLO format. Picks the largest external contour if mask has
    multiple components.
    """
    import cv2

    h, w = mask.shape
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return []
    simp = cv2.approxPolyDP(contour, simplify_tol, True)
    return [(float(p[0][0]) / w, float(p[0][1]) / h) for p in simp]


def export_yolo_labels(
    submasks_per_tree: Dict[int, Dict[int, List[SubMask]]],
    output_dir: str,
    cfg: LAIConfig,
    image_extension: str = ".png",
) -> None:
    """Write YOLO-format segmentation labels, one .txt per frame.

    Inputs
    ------
    submasks_per_tree :
        ``{tree_id: {frame_idx: [SubMask, ...]}}`` from the
        classification stage, accumulated across all trees.
    output_dir :
        Destination root. Creates ``labels/`` and writes
        ``labels/{frame_idx:06d}.txt``. Also writes ``classes.txt``
        and a ``dataset.yaml`` ready for Ultralytics YOLO training.

    Format per line (Ultralytics segmentation):
        class_id x1 y1 x2 y2 ... xn yn
    """
    out_root = Path(output_dir)
    label_dir = out_root / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    # Invert: frame_idx -> list of (class_id, mask).
    by_frame: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for tid, frames in submasks_per_tree.items():
        for fid, sms in frames.items():
            for sm in sms:
                by_frame.setdefault(fid, []).append((sm.class_id, sm.mask))

    for fid, items in by_frame.items():
        lines: List[str] = []
        for class_id, mask in items:
            poly = _mask_to_polygon(mask, cfg.yolo_polygon_simplify_tol)
            if len(poly) < 3:
                continue
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            lines.append(f"{class_id} {coords}")
        label_path = label_dir / f"{fid:06d}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    # classes.txt
    (out_root / "classes.txt").write_text(
        "\n".join(cfg.yolo_class_names) + "\n",
    )

    # dataset.yaml
    yaml = (
        f"# YOLO segmentation dataset auto-generated by lai_voxel_estimator\n"
        f"path: {out_root.resolve()}\n"
        f"train: images\n"
        f"val: images\n"
        f"names:\n"
    )
    for i, name in enumerate(cfg.yolo_class_names):
        yaml += f"  {i}: {name}\n"
    (out_root / "dataset.yaml").write_text(yaml)
    log.info("Wrote %d YOLO labels + classes.txt + dataset.yaml to %s",
             len(by_frame), out_root)


# ============================================================================
# Top-level orchestrator
# ============================================================================
def compute_tree_lai(
    cluster: TreeCluster,
    loader: FrameLoader,
    cfg: LAIConfig,
    device: str = "cuda",
) -> Tuple[TreeLAIResult, Dict[int, List[SubMask]]]:
    """Full LAI estimation for a single tree.

    Returns the LAI result + the per-frame classified sub-masks
    (so the caller can accumulate them for YOLO export across
    multiple trees).
    """
    notes: List[str] = []

    # Hierarchical SAM2.
    submasks_by_frame_raw = segment_tree_subregions(cluster, loader, cfg, device)
    if not submasks_by_frame_raw:
        notes.append("no_submasks")

    # Classification.
    classified = classify_subregions(submasks_by_frame_raw, cluster, loader, cfg)

    # ICP (optional, GPS-only fallback if open3d missing).
    icp = register_frames_with_icp(cluster, loader, cfg)

    # 3D leaf cloud.
    points = aggregate_tree_pointcloud(cluster, classified, loader, cfg, icp)

    # Voxel LAI.
    lai_voxel, leaf_area_m2, n_occ, n_surf = voxelize_and_estimate(points, cfg)

    # Build per-frame silhouette dict from the cluster's tree masks
    # (or trunk masks if no tree mask was produced for that frame).
    # The Beer-Lambert path uses this directly when available --
    # gap_fraction is computed from the silhouette vs its bounding
    # column instead of relying on sub-mask classification.
    silhouette_by_frame: Dict[int, np.ndarray] = {}
    for track in cluster.tracks:
        for det in track.detections:
            if det.frame_idx not in cluster.frame_pixels:
                continue
            sil = (det.tree_mask
                   if (det.tree_mask is not None and det.tree_mask.any())
                   else det.mask)
            if sil is None or not sil.any():
                continue
            existing = silhouette_by_frame.get(det.frame_idx)
            silhouette_by_frame[det.frame_idx] = (
                sil if existing is None else (existing | sil)
            )

    # Beer-Lambert per-frame, silhouette-based when available.
    beer_per_frame = gap_fraction_lai_per_frame(
        classified, cfg, silhouette_by_frame=silhouette_by_frame,
    )
    if beer_per_frame:
        beer_mean = float(np.mean(beer_per_frame))
        beer_median = float(np.median(beer_per_frame))
    else:
        beer_mean = beer_median = float("nan")
        notes.append("no_beer_lambert_estimate")

    # Leaf-count branch (third independent estimator).
    leaf_lai, leaf_diag = lai_from_leaf_count(cluster, classified, loader, cfg)
    if not math.isfinite(leaf_lai):
        notes.append("no_leaf_count_estimate")

    # Calibrated fusion using the cfg-supplied coefficients (default
    # = identity on voxel). After calibration the user re-runs with
    # fitted (a, b, c, d).
    components: List[Tuple[float, float]] = [(cfg.fusion_a, lai_voxel)]
    if math.isfinite(beer_mean):
        components.append((cfg.fusion_b, beer_mean))
    if math.isfinite(leaf_lai):
        components.append((cfg.fusion_d, leaf_lai))
    calibrated = sum(c * v for c, v in components) + cfg.fusion_c

    # Sanity flags.
    if lai_voxel <= 0.05:
        notes.append("voxel_lai_implausibly_low")
    if lai_voxel > 8.0:
        notes.append("voxel_lai_implausibly_high")
    if math.isfinite(beer_mean) and abs(lai_voxel - beer_mean) > 1.5:
        notes.append(f"voxel_beer_disagreement_{lai_voxel - beer_mean:+.2f}")
    if math.isfinite(leaf_lai) and abs(lai_voxel - leaf_lai) > 2.0:
        notes.append(f"voxel_leafcount_disagreement_{lai_voxel - leaf_lai:+.2f}")

    result = TreeLAIResult(
        tree_id=cluster.tree_id,
        n_frames=len(classified),
        n_leaf_points=int(points.shape[0]),
        voxel_lai=float(lai_voxel),
        beer_lambert_lai_per_frame=beer_per_frame,
        beer_lambert_lai_mean=beer_mean,
        beer_lambert_lai_median=beer_median,
        leaf_count_lai=float(leaf_lai),
        n_leaves_detected=int(leaf_diag["n_leaves"]),
        mean_leaf_area_m2=float(leaf_diag["mean_leaf_area_m2"]),
        leaf_area_m2_p25=float(leaf_diag["p25_leaf_area_m2"]),
        leaf_area_m2_p75=float(leaf_diag["p75_leaf_area_m2"]),
        visible_leaf_area_m2=float(leaf_diag["visible_leaf_area_m2"]),
        calibrated_lai=float(calibrated),
        leaf_area_m2=float(leaf_area_m2),
        notes=notes,
    )
    return result, classified


def process_clusters_for_lai(
    clusters: List[TreeCluster],
    loader: FrameLoader,
    cfg: Optional[LAIConfig] = None,
    output_dir: str = "lai_out",
    device: str = "cuda",
    write_yolo_labels: bool = True,
    truth_lai_by_tree: Optional[Dict[int, float]] = None,
) -> List[TreeLAIResult]:
    """End-to-end LAI estimation for every tree in the run.

    If ``truth_lai_by_tree`` is provided (a dict of tree_id -> 2200C
    LAI for some trees), runs the calibration step and updates each
    result's ``calibrated_lai`` accordingly. Coefficients (a, b, c)
    and calibration metrics are written to ``calibration.json``.

    YOLO labels are written to ``output_dir/yolo/`` if
    ``write_yolo_labels=True``.
    """
    cfg = cfg or LAIConfig()
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: List[TreeLAIResult] = []
    submasks_per_tree: Dict[int, Dict[int, List[SubMask]]] = {}

    for cluster in clusters:
        result, classified = compute_tree_lai(cluster, loader, cfg, device)
        results.append(result)
        submasks_per_tree[cluster.tree_id] = classified
        log.info(
            "Tree %d: voxel_LAI=%.3f, beer_LAI=%.3f (n=%d frames)",
            cluster.tree_id, result.voxel_lai,
            result.beer_lambert_lai_mean, result.n_frames,
        )

    # Calibration pass.
    if truth_lai_by_tree:
        calib_v: List[float] = []
        calib_b: List[float] = []
        calib_l: List[float] = []
        calib_y: List[float] = []
        for r in results:
            if r.tree_id not in truth_lai_by_tree:
                continue
            if not math.isfinite(r.beer_lambert_lai_mean):
                continue
            calib_v.append(r.voxel_lai)
            calib_b.append(r.beer_lambert_lai_mean)
            calib_l.append(r.leaf_count_lai
                           if math.isfinite(r.leaf_count_lai) else 0.0)
            calib_y.append(truth_lai_by_tree[r.tree_id])
        if calib_v:
            a, b, c, d, metrics = calibrate_against_2200c(
                calib_v, calib_b, calib_l, calib_y,
            )
            cfg.fusion_a, cfg.fusion_b = a, b
            cfg.fusion_c, cfg.fusion_d = c, d
            # Re-apply to all trees with the fitted coefficients.
            for r in results:
                pred = a * r.voxel_lai + c
                if math.isfinite(r.beer_lambert_lai_mean):
                    pred += b * r.beer_lambert_lai_mean
                if math.isfinite(r.leaf_count_lai):
                    pred += d * r.leaf_count_lai
                r.calibrated_lai = pred
            (out_root / "calibration.json").write_text(json.dumps({
                "fusion_a": a, "fusion_b": b,
                "fusion_c": c, "fusion_d": d,
                "metrics": metrics,
            }, indent=2))
            log.info(
                "Calibration: a=%.3f b=%.3f d=%.3f c=%.3f  RMSE=%.3f  R²=%.3f",
                a, b, d, c, metrics["rmse"], metrics["r2"],
            )

    # Per-tree results JSON.
    (out_root / "lai_per_tree.json").write_text(json.dumps([
        {
            "tree_id": r.tree_id,
            "n_frames": r.n_frames,
            "n_leaf_points": r.n_leaf_points,
            "voxel_lai": round(r.voxel_lai, 4),
            "beer_lambert_lai_mean": round(r.beer_lambert_lai_mean, 4)
                if math.isfinite(r.beer_lambert_lai_mean) else None,
            "beer_lambert_lai_median": round(r.beer_lambert_lai_median, 4)
                if math.isfinite(r.beer_lambert_lai_median) else None,
            "leaf_count_lai": round(r.leaf_count_lai, 4)
                if math.isfinite(r.leaf_count_lai) else None,
            "n_leaves_detected": r.n_leaves_detected,
            "mean_leaf_area_m2": round(r.mean_leaf_area_m2, 6)
                if math.isfinite(r.mean_leaf_area_m2) else None,
            "leaf_area_p25_m2": round(r.leaf_area_m2_p25, 6)
                if math.isfinite(r.leaf_area_m2_p25) else None,
            "leaf_area_p75_m2": round(r.leaf_area_m2_p75, 6)
                if math.isfinite(r.leaf_area_m2_p75) else None,
            "visible_leaf_area_m2": round(r.visible_leaf_area_m2, 4)
                if math.isfinite(r.visible_leaf_area_m2) else None,
            "calibrated_lai": round(r.calibrated_lai, 4),
            "leaf_area_m2": round(r.leaf_area_m2, 3),
            "notes": r.notes,
        }
        for r in results
    ], indent=2))

    # YOLO labels.
    if write_yolo_labels:
        export_yolo_labels(submasks_per_tree, str(out_root / "yolo"), cfg)

    return results


# ============================================================================
# Leaf-counting branch (third LAI estimator)
# ============================================================================
def _measure_leaf_area_m2(
    mask: np.ndarray,
    depth_m: np.ndarray,
    cfg: LAIConfig,
) -> Tuple[float, float, float, float]:
    """Convert a leaf mask to a true area in m².

    Returns (projected_area_m2, true_area_m2, view_angle_rad, depth_std).

    Pipeline:
      1. Pixel area × (z/fx)(z/fy) gives projected area (m²) at the
         leaf's median depth -- this is what a flat leaf facing the
         camera would have.
      2. Foreshortening correction: estimate the leaf's normal from
         the local depth gradient inside the mask. A leaf tilted at
         angle θ from the viewing axis has projected_area = true_area
         × cos(θ); we recover true_area by dividing by cos(θ),
         clipped to a 4× ceiling so degenerate (near-edge-on) views
         don't blow up.
    """
    valid = mask & np.isfinite(depth_m) & (depth_m > 0.1) & (depth_m < 20.0)
    if valid.sum() < 10:
        return 0.0, 0.0, 0.0, 0.0

    z_med = float(np.median(depth_m[valid]))
    z_std = float(np.std(depth_m[valid]))
    intr = cfg.intrinsics
    px_per_m_x = intr.fx / z_med
    px_per_m_y = intr.fy / z_med
    n_pix = int(valid.sum())
    proj_area = n_pix / (px_per_m_x * px_per_m_y)

    if not cfg.correct_foreshortening:
        return proj_area, proj_area, 0.0, z_std

    # Estimate leaf normal from depth gradients within the mask.
    # A flat leaf has depth that varies linearly across its surface;
    # the gradient direction + magnitude encodes tilt.
    ys, xs = np.where(valid)
    if ys.size < 30:
        return proj_area, proj_area, 0.0, z_std
    z_vals = depth_m[ys, xs]
    # Centre coordinates in the local image frame.
    xc = xs - xs.mean()
    yc = ys - ys.mean()
    # Fit z = a*x + b*y + c with least squares.
    A = np.stack([xc, yc, np.ones_like(xc)], axis=1).astype(np.float32)
    try:
        coef, *_ = np.linalg.lstsq(A, z_vals.astype(np.float32), rcond=None)
        a, b, _ = coef
    except np.linalg.LinAlgError:
        return proj_area, proj_area, 0.0, z_std

    # Convert pixel-space gradient to metric: dz/dx_m = a × fx / z_med.
    dz_dx = float(a) * intr.fx / z_med
    dz_dy = float(b) * intr.fy / z_med
    # Leaf surface normal (camera frame): (-dz_dx, -dz_dy, 1) normalized.
    nz_norm = 1.0 / math.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
    cos_angle = nz_norm                            # dot with view axis (0,0,1)
    cos_angle = max(0.25, cos_angle)               # clip ceiling at ~75°
    angle = math.acos(cos_angle)
    true_area = proj_area / cos_angle
    return proj_area, true_area, angle, z_std


def detect_individual_leaves(
    cluster: TreeCluster,
    submasks_by_frame: Dict[int, List[SubMask]],
    loader: FrameLoader,
    cfg: LAIConfig,
) -> List[LeafDetection]:
    """Filter SAM2 sub-masks to single-leaf masks and measure each.

    Inputs are the classified sub-masks from ``classify_subregions``;
    we keep only those classified as leaf (class_id=1) and additionally
    filter to single-leaf shapes via area, aspect-ratio, and depth-
    coherence tests. Multi-leaf clumps and mask fragments fail
    these tests and get discarded.
    """
    import cv2

    detections: List[LeafDetection] = []
    cos_lat0 = math.cos(math.radians(cluster.world_lat)) or 1e-6
    intr = cfg.intrinsics
    u_grid, v_grid = np.meshgrid(
        np.arange(intr.width, dtype=np.float32),
        np.arange(intr.height, dtype=np.float32),
    )

    for frame_idx, sms in submasks_by_frame.items():
        depth = loader.load_depth_m(frame_idx)
        meta = loader.load_meta(frame_idx)

        # Camera-to-ENU transform anchored at cluster centroid.
        T, _ = _camera_to_world_se3(
            float(meta["gps_lat"]), float(meta["gps_lon"]),
            float(meta["heading_deg"]), cfg,
        )
        ant_e = (float(meta["gps_lon"]) - cluster.world_lon) * 111_320.0 * cos_lat0
        ant_n = (float(meta["gps_lat"]) - cluster.world_lat) * 111_320.0
        T = T.copy()
        T[:3, 3] += np.array([ant_e, ant_n, 0.0])

        for sm in sms:
            if sm.class_id != 1:                              # leaves only
                continue
            mask = sm.mask
            n_pix = int(mask.sum())
            if n_pix < 20:
                continue

            proj_a, true_a, view_ang, z_std = _measure_leaf_area_m2(
                mask, depth, cfg,
            )
            if true_a < cfg.min_leaf_area_m2 or true_a > cfg.max_leaf_area_m2:
                continue
            if z_std > cfg.leaf_depth_std_max_m:
                continue

            # Aspect ratio from the rotated bounding rectangle.
            ys, xs = np.where(mask)
            if xs.size < 5:
                continue
            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            (_, (w_rect, h_rect), _) = cv2.minAreaRect(pts)
            if min(w_rect, h_rect) < 1e-3:
                continue
            aspect = max(w_rect, h_rect) / max(min(w_rect, h_rect), 1e-3)
            if aspect < cfg.min_leaf_aspect or aspect > cfg.max_leaf_aspect:
                continue

            # 3D centroid in cluster-local ENU.
            valid = mask & np.isfinite(depth) & (depth > 0.1) & (depth < 20.0)
            if not valid.any():
                continue
            z_med = float(np.median(depth[valid]))
            cy_pix = float(np.mean(v_grid[valid]))
            cx_pix = float(np.mean(u_grid[valid]))
            X = (cx_pix - intr.cx) / intr.fx * z_med
            Y = (cy_pix - intr.cy) / intr.fy * z_med
            cam_pt = np.array([X, Y, z_med, 1.0])
            world_pt = T @ cam_pt

            detections.append(LeafDetection(
                frame_idx=frame_idx,
                tree_id=cluster.tree_id,
                mask=mask,
                pixel_area=n_pix,
                depth_median_m=z_med,
                depth_std_m=z_std,
                aspect_ratio=float(aspect),
                world_e_m=float(world_pt[0]),
                world_n_m=float(world_pt[1]),
                world_u_m=float(world_pt[2]),
                projected_area_m2=float(proj_a),
                true_area_m2=float(true_a),
                view_angle_rad=float(view_ang),
                confidence=float(sm.confidence),
            ))

    log.debug("Tree %d: %d candidate leaves passed filters",
              cluster.tree_id, len(detections))
    return detections


def deduplicate_leaves_3d(
    detections: List[LeafDetection],
    cfg: LAIConfig,
) -> List[LeafDetection]:
    """Collapse multi-frame views of the same physical leaf.

    DBSCAN at ``leaf_dedup_radius_m`` over (E, N, U) centroids. Each
    cluster = one physical leaf. We keep the highest-confidence
    detection per cluster, but use the *median* true_area_m2 across
    the cluster's members (multi-view averaging reduces single-frame
    area-estimation noise).

    Singletons (DBSCAN noise) are kept -- they're real leaves seen
    in only one frame, common at canopy edges.
    """
    if len(detections) < 2:
        return list(detections)
    from sklearn.cluster import DBSCAN

    pts = np.array([(d.world_e_m, d.world_n_m, d.world_u_m)
                    for d in detections], dtype=np.float32)
    db = DBSCAN(eps=cfg.leaf_dedup_radius_m, min_samples=1).fit(pts)
    labels = db.labels_

    deduped: List[LeafDetection] = []
    for lbl in set(labels):
        members = [d for d, l in zip(detections, labels) if l == lbl]
        if not members:
            continue
        # Pick the highest-confidence member as the representative.
        best = max(members, key=lambda d: d.confidence)
        # Median area across multi-frame views.
        median_area = float(np.median([m.true_area_m2 for m in members]))
        # Replace area with the dedup-averaged value.
        best = LeafDetection(
            **{**best.__dict__, "true_area_m2": median_area},
        )
        deduped.append(best)

    log.debug("Leaf dedup: %d -> %d (%.1f×)",
              len(detections), len(deduped),
              len(detections) / max(len(deduped), 1))
    return deduped


def lai_from_leaf_count(
    cluster: TreeCluster,
    submasks_by_frame: Dict[int, List[SubMask]],
    loader: FrameLoader,
    cfg: LAIConfig,
) -> Tuple[float, Dict]:
    """Estimate LAI by counting + measuring individual leaves.

    Pipeline:
      1. Filter classified leaves to single-leaf masks (size, aspect,
         depth coherence).
      2. Measure each leaf's true area in m² with foreshortening
         correction.
      3. 3D dedup across frames so each physical leaf is counted once.
      4. Sum visible leaf areas; multiply by ``leaf_visibility_factor``
         to estimate true total leaf area (corrects for occluded /
         non-imaged leaves).
      5. Divide by ground area for LAI.

    Returns (lai, diagnostics) where diagnostics includes leaf count,
    mean / quartile areas, and the visible (uncorrected) leaf area.
    """
    detections = detect_individual_leaves(
        cluster, submasks_by_frame, loader, cfg,
    )
    detections = deduplicate_leaves_3d(detections, cfg)

    if not detections:
        return float("nan"), {
            "n_leaves": 0,
            "mean_leaf_area_m2": float("nan"),
            "p25_leaf_area_m2": float("nan"),
            "p75_leaf_area_m2": float("nan"),
            "visible_leaf_area_m2": 0.0,
        }

    areas = np.array([d.true_area_m2 for d in detections])
    visible_total = float(areas.sum())
    corrected_total = visible_total * cfg.leaf_visibility_factor
    ground_area = cfg.tree_spacing_m * cfg.row_spacing_m
    lai = corrected_total / max(ground_area, 1e-6)

    diagnostics = {
        "n_leaves": int(len(detections)),
        "mean_leaf_area_m2": float(areas.mean()),
        "p25_leaf_area_m2": float(np.percentile(areas, 25)),
        "p75_leaf_area_m2": float(np.percentile(areas, 75)),
        "visible_leaf_area_m2": visible_total,
    }
    return lai, diagnostics


# ============================================================================
# ROI-driven entry point (skip trunk segmentation entirely)
# ============================================================================
@dataclass
class TreeROI:
    """One tree's region-of-interest in one frame.

    Provide either ``bbox_xyxy`` (pixel coords, will be refined to a
    silhouette mask via SAM2 image mode) or ``mask`` (bool array,
    frame-shape, used directly). All ROIs sharing the same
    ``tree_id`` are treated as multi-view observations of the same
    physical tree -- the voxel branch fuses them into one point
    cloud, the Beer-Lambert branch averages their per-frame LAI.
    """
    frame_idx: int
    tree_id: int
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.bbox_xyxy is None and self.mask is None:
            raise ValueError(
                "TreeROI requires either bbox_xyxy or mask"
            )


def _refine_bboxes_to_masks(
    rois: List[TreeROI],
    loader: FrameLoader,
    cfg: LAIConfig,
    device: str = "cuda",
) -> List[TreeROI]:
    """SAM2 image-mode refinement of bbox ROIs into silhouette masks.

    A tight tree silhouette is essential for the Beer-Lambert branch
    (gap_pixels / canopy_pixels needs the denominator to be the
    actual tree extent, not a bbox that includes background).

    Mask-already ROIs pass through unchanged.
    """
    needs_refine = [r for r in rois if r.mask is None]
    if not needs_refine:
        return rois

    import torch
    from PIL import Image as _PILImage
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from tqdm.auto import tqdm

    log.info("Loading SAM 3 (geometric box prompt) to refine %d bbox ROIs",
             len(needs_refine))
    sam3_model = build_sam3_image_model()
    processor = Sam3Processor(
        sam3_model, device=device, confidence_threshold=0.05,
    )
    autocast_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    by_frame: Dict[int, List[TreeROI]] = {}
    for r in needs_refine:
        by_frame.setdefault(r.frame_idx, []).append(r)

    out_lookup: Dict[int, np.ndarray] = {}
    for fid, rs in tqdm(by_frame.items(), desc="Refine bboxes"):
        rgb = loader.load_rgb(fid)
        h, w = rgb.shape[:2]
        with torch.autocast(
            "cuda" if device.startswith("cuda") else "cpu",
            dtype=autocast_dtype,
        ):
            state = processor.set_image(_PILImage.fromarray(rgb))
            for r in rs:
                processor.reset_all_prompts(state)
                x1, y1, x2, y2 = r.bbox_xyxy
                cx = (x1 + x2) * 0.5 / w
                cy = (y1 + y2) * 0.5 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                state = processor.add_geometric_prompt(
                    box=[cx, cy, bw, bh], label=True, state=state,
                )
                masks = state.get("masks")
                boxes = state.get("boxes")
                if masks is None or len(masks) == 0:
                    continue
                m_arr = masks.detach().cpu()
                if m_arr.dtype != torch.bool:
                    m_arr = m_arr.to(torch.bool)
                m_arr = m_arr.numpy()
                if m_arr.ndim == 4:
                    m_arr = m_arr.squeeze(1)
                # Pick the mask whose box best overlaps the input bbox.
                if boxes is not None and len(boxes) > 0:
                    b_arr = boxes.detach().cpu()
                    if b_arr.dtype in (torch.bfloat16, torch.float16):
                        b_arr = b_arr.to(torch.float32)
                    b_arr = b_arr.numpy()
                    best_iou, best_i = -1.0, -1
                    a_area = max(0.0, (x2 - x1) * (y2 - y1))
                    for i, b in enumerate(b_arr):
                        bx1, by1, bx2, by2 = b
                        ix1 = max(x1, bx1); iy1 = max(y1, by1)
                        ix2 = min(x2, bx2); iy2 = min(y2, by2)
                        iw = max(0.0, ix2 - ix1); ih_ = max(0.0, iy2 - iy1)
                        inter = iw * ih_
                        b_area = max(0.0, (bx2 - bx1) * (by2 - by1))
                        union = a_area + b_area - inter
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_i = i
                else:
                    best_i = 0
                if best_i >= 0 and best_i < len(m_arr):
                    m = m_arr[best_i].astype(bool)
                    if m.ndim == 2 and m.any():
                        out_lookup[id(r)] = m

    refined: List[TreeROI] = []
    for r in rois:
        if r.mask is not None:
            refined.append(r)
            continue
        m = out_lookup.get(id(r))
        if m is None:
            log.warning("SAM2 produced no mask for ROI tree=%d frame=%d; skipping",
                        r.tree_id, r.frame_idx)
            continue
        refined.append(TreeROI(
            frame_idx=r.frame_idx, tree_id=r.tree_id,
            bbox_xyxy=r.bbox_xyxy, mask=m,
        ))
    return refined


def _build_clusters_from_rois(
    rois: List[TreeROI],
    loader: FrameLoader,
    cfg: LAIConfig,
) -> List[TreeCluster]:
    """Synthesise minimal :class:`TreeCluster` objects from ROIs.

    The downstream pipeline operates on TreeCluster, so we wrap the
    ROIs in the cluster structure: one TrunkTrack per cluster
    holding one TrunkDetection per frame, with the ROI mask as the
    detection's silhouette mask. World coords come from the median
    GPS+depth across the cluster's frames.

    The "trunk" detections built here are silhouette-shaped, not
    actual trunks -- but the LAI pipeline uses them only as
    silhouette masks for sub-segmentation, so the naming is the
    only awkwardness.
    """
    # Sam2OrchardSegmenter dataclasses, imported at top of module.
    from sam2_orchard_segmenter import TrunkTrack, TrunkDetection  # type: ignore

    by_tree: Dict[int, List[TreeROI]] = {}
    for r in rois:
        if r.mask is None:
            continue
        by_tree.setdefault(r.tree_id, []).append(r)

    clusters: List[TreeCluster] = []
    for tree_id, tree_rois in by_tree.items():
        track = TrunkTrack(track_id=tree_id)

        gps_lats: List[float] = []
        gps_lons: List[float] = []
        for r in tree_rois:
            meta = loader.load_meta(r.frame_idx)
            depth = loader.load_depth_m(r.frame_idx)
            valid = r.mask & np.isfinite(depth) & (depth > 0.1) & (depth < 20.0)
            d_med = float(np.median(depth[valid])) if valid.any() else 2.5

            wlat, wlon = camera_to_world(
                gps_lat=float(meta["gps_lat"]),
                gps_lon=float(meta["gps_lon"]),
                heading_deg=float(meta["heading_deg"]),
                depth_m=d_med,
                lateral_offset_m=cfg.cam_offset_left_m,
            )
            gps_lats.append(wlat)
            gps_lons.append(wlon)

            ys, xs = np.where(r.mask)
            if xs.size == 0:
                continue
            bbox = (float(xs.min()), float(ys.min()),
                    float(xs.max()), float(ys.max()))
            det = TrunkDetection(
                frame_idx=r.frame_idx,
                bbox_xyxy=bbox,
                score=1.0,
                mask=r.mask,
                depth_m=d_med,
                world_lat=wlat,
                world_lon=wlon,
                track_id=tree_id,
            )
            track.detections.append(det)

        if not track.detections:
            continue

        cluster = TreeCluster(
            tree_id=tree_id,
            world_lat=float(np.median(gps_lats)),
            world_lon=float(np.median(gps_lons)),
            tracks=[track],
            frame_pixels={
                d.frame_idx: int(d.mask.sum())
                for d in track.detections if d.mask is not None
            },
        )
        clusters.append(cluster)

    log.info("Built %d clusters from %d ROIs",
             len(clusters), len(rois))
    return sorted(clusters, key=lambda c: c.tree_id)


def lai_from_rois(
    rois: List[TreeROI],
    loader: FrameLoader,
    cfg: Optional[LAIConfig] = None,
    output_dir: str = "lai_roi_out",
    device: str = "cuda",
    write_yolo_labels: bool = True,
    truth_lai_by_tree: Optional[Dict[int, float]] = None,
    refine_bbox_with_sam2: bool = True,
) -> List[TreeLAIResult]:
    """End-to-end LAI from pre-defined ROIs (the typical "I have data" path).

    Use this when you already have tree ROIs -- bounding boxes or
    masks per frame, with a tree_id label -- from manual annotation,
    your existing tracker output, or imported labels. Skips the
    trunk-anchored segmenter entirely.

    Multi-view fusion is automatic: ROIs sharing a ``tree_id`` are
    fused into one per-tree point cloud for the voxel branch and
    averaged in the Beer-Lambert branch.

    For SINGLE-VIEW per tree (one ROI per tree across the dataset),
    voxel LAI is degraded -- only one viewpoint = no occlusion
    averaging -- but Beer-Lambert is unaffected and remains accurate
    given a calibrated extinction coefficient. The fusion step (if
    you have 2200C truth) will down-weight voxel automatically.
    """
    cfg = cfg or LAIConfig()

    if refine_bbox_with_sam2:
        rois = _refine_bboxes_to_masks(rois, loader, cfg, device)
    else:
        # Without SAM2 refinement, fall back to filling bbox interior
        # as a rectangular "mask". Crude but lets you skip the SAM2
        # load if your bboxes are already tight to the canopy.
        for r in rois:
            if r.mask is None and r.bbox_xyxy is not None:
                w_img = cfg.intrinsics.width
                h_img = cfg.intrinsics.height
                m = np.zeros((h_img, w_img), dtype=bool)
                x1, y1, x2, y2 = (int(round(v)) for v in r.bbox_xyxy)
                m[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)] = True
                r.mask = m

    clusters = _build_clusters_from_rois(rois, loader, cfg)

    return process_clusters_for_lai(
        clusters, loader, cfg=cfg, output_dir=output_dir, device=device,
        write_yolo_labels=write_yolo_labels,
        truth_lai_by_tree=truth_lai_by_tree,
    )


def load_rois_from_csv(
    path: str,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> List[TreeROI]:
    """Load ROIs from a CSV with columns ``frame_idx,tree_id,x1,y1,x2,y2``.

    The header row is required. Pixel coordinates can be either
    absolute or normalised [0, 1]; if any value > 1 the row is
    treated as absolute, else it's scaled by image_width/height
    (which then must be supplied).

    For mask-based ROIs, supply a ``mask_path`` column instead of
    bbox columns -- that path is loaded with ``np.load``.
    """
    import csv as _csv

    rois: List[TreeROI] = []
    with open(path, "r", newline="") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            frame_idx = int(row["frame_idx"])
            tree_id = int(row["tree_id"])
            mask_path = row.get("mask_path", "").strip()
            if mask_path:
                mask = np.load(mask_path).astype(bool)
                rois.append(TreeROI(frame_idx, tree_id, mask=mask))
                continue
            x1 = float(row["x1"]); y1 = float(row["y1"])
            x2 = float(row["x2"]); y2 = float(row["y2"])
            if max(x1, y1, x2, y2) <= 1.0:
                if image_width is None or image_height is None:
                    raise ValueError(
                        "Normalised bbox coords found; supply image_width/height"
                    )
                x1 *= image_width;  x2 *= image_width
                y1 *= image_height; y2 *= image_height
            rois.append(TreeROI(
                frame_idx, tree_id, bbox_xyxy=(x1, y1, x2, y2),
            ))
    log.info("Loaded %d ROIs from %s", len(rois), path)
    return rois


# ============================================================================
# CLI
# ============================================================================
def _main() -> None:
    import argparse
    import csv as _csv

    parser = argparse.ArgumentParser(
        description="Per-tree LAI from ROIs — HPC/supercomputer-ready",
    )
    parser.add_argument("run_root", type=str,
                        help="Either an All2023 session dir (containing RGB/, depth/, "
                             "PRGB/, IR/, Info/) — autodetected — or a PNG-sequence "
                             "root (rgb/000000.png, depth/000000.npy + meta JSON).")
    parser.add_argument("--loader", choices=("auto", "all2023", "png"),
                        default="auto",
                        help="Force a specific FrameLoader; default 'auto' picks "
                             "All2023FrameLoader when run_root contains RGB/ + Info/, "
                             "else PNGSequenceLoader.")
    parser.add_argument("--row-heading-deg", type=float, default=None,
                        help="Compass bearing of the orchard row "
                             "(0=N, 90=E). All2023 only; defaults to GPS-trail "
                             "estimate from the session's first → last GPS fix.")
    parser.add_argument("--roi-csv", type=str, required=True,
                        help="CSV with columns: frame_idx,tree_id,x1,y1,x2,y2  "
                             "(or mask_path instead of bbox columns). "
                             "Use scripts/prgb_to_roi_csv.py to generate from PRGB.")
    parser.add_argument("--output-dir", type=str, default="lai_out")
    parser.add_argument("--truth-csv", type=str, default=None,
                        help="Optional CSV: tree_id,lai  for LAI-2200C calibration")
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--no-icp", action="store_true")
    parser.add_argument("--no-sam2-refine", action="store_true",
                        help="Skip SAM2 bbox→mask refinement; use bbox rectangle directly. "
                             "Saves GPU memory on nodes without a large VRAM budget.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device string: 'cuda', 'cuda:1', 'cpu', etc.")
    # ── HPC chunking ─────────────────────────────────────────────────────────
    # Set --n-chunks to the SLURM array size and --chunk-idx to
    # $SLURM_ARRAY_TASK_ID.  Each job processes every n-th tree so the
    # workload is spread evenly without any coordination file.
    parser.add_argument("--n-chunks", type=int, default=1,
                        help="Total SLURM array size (number of parallel jobs)")
    parser.add_argument("--chunk-idx", type=int, default=0,
                        help="Zero-based index of this job (0 .. n-chunks-1); "
                             "set to $SLURM_ARRAY_TASK_ID in your job script")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = LAIConfig()
    if args.no_icp:
        cfg.use_icp = False

    # Pick the loader. All2023 sessions are recognised by the
    # presence of RGB/ (capital) and Info/ siblings; PNG sequences
    # use rgb/ (lowercase) + depth/ + a meta JSON.
    run_root = Path(args.run_root)
    if args.loader == "all2023" or (
        args.loader == "auto"
        and (run_root / "RGB").is_dir()
        and (run_root / "Info").is_dir()
    ):
        log.info("Using All2023FrameLoader on %s", run_root)
        loader = All2023FrameLoader(
            str(run_root), row_heading_deg=args.row_heading_deg,
        )
    else:
        log.info("Using PNGSequenceLoader on %s", run_root)
        loader = PNGSequenceLoader(str(run_root))

    rois = load_rois_from_csv(
        args.roi_csv,
        image_height=cfg.intrinsics.height,
        image_width=cfg.intrinsics.width,
    )
    if not rois:
        log.error("No ROIs loaded from %s — check the CSV format.", args.roi_csv)
        raise SystemExit(1)

    # ── HPC chunk selection ───────────────────────────────────────────────────
    if args.n_chunks > 1:
        all_tree_ids = sorted({r.tree_id for r in rois})
        # Stride assignment so each chunk gets a contiguous-ish spread of IDs.
        chunk_ids = set(all_tree_ids[args.chunk_idx::args.n_chunks])
        rois = [r for r in rois if r.tree_id in chunk_ids]
        output_dir = str(Path(args.output_dir) / f"chunk_{args.chunk_idx:04d}")
        log.info(
            "Chunk %d/%d: %d trees → %s",
            args.chunk_idx, args.n_chunks, len(chunk_ids), output_dir,
        )
    else:
        output_dir = args.output_dir

    if not rois:
        log.warning("Chunk %d has no trees; nothing to do.", args.chunk_idx)
        raise SystemExit(0)

    # ── Optional ground-truth LAI for calibration ─────────────────────────────
    truth_lai: Optional[Dict[int, float]] = None
    if args.truth_csv:
        truth_lai = {}
        with open(args.truth_csv, newline="") as fh:
            for row in _csv.DictReader(fh):
                truth_lai[int(row["tree_id"])] = float(row["lai"])
        log.info("Loaded %d ground-truth LAI values from %s",
                 len(truth_lai), args.truth_csv)

    results = lai_from_rois(
        rois,
        loader,
        cfg=cfg,
        output_dir=output_dir,
        device=args.device,
        write_yolo_labels=not args.no_yolo,
        truth_lai_by_tree=truth_lai,
        refine_bbox_with_sam2=not args.no_sam2_refine,
    )

    log.info(
        "Done. %d trees processed. Results → %s/lai_per_tree.json",
        len(results), output_dir,
    )


if __name__ == "__main__":
    _main()


__all__ = [
    "CameraIntrinsics",
    "LAIConfig",
    "SubMask",
    "LeafDetection",
    "TreeLAIResult",
    "TreeROI",
    "segment_tree_subregions",
    "classify_subregions",
    "backproject_to_world",
    "register_frames_with_icp",
    "aggregate_tree_pointcloud",
    "voxelize_and_estimate",
    "gap_fraction_lai_per_frame",
    "detect_individual_leaves",
    "deduplicate_leaves_3d",
    "lai_from_leaf_count",
    "calibrate_against_2200c",
    "export_yolo_labels",
    "compute_tree_lai",
    "process_clusters_for_lai",
    "lai_from_rois",
    "load_rois_from_csv",
]
