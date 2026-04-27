"""sam2_orchard_segmenter.py — Trunk-anchored post-processing tree
segmentation using Grounding DINO + SAM2 + DBSCAN.

Drop-in replacement for ``TreeTracker.refine_with_signal``. Trades
real-time capability (this runs at ~1-5 fps on a single GPU) for
ground-truth-quality tree count and exact frame-to-tree attribution.

The fundamental insight: canopy pixel sums blur identity at touching
canopies; trunks don't. One trunk = one tree, always. We detect trunks
zero-shot with Grounding DINO, refine each detection to a precise
mask with SAM2 (image mode, one frame at a time), project each
trunk's median depth into world coordinates, and DBSCAN the resulting
cloud. Each cluster is one physical tree.

**Frame rate note.** Your input is a 5 fps frame stream — individual
RGB+depth frames captured chronologically. There is no video file
involved. SAM2's "video predictor" is a misnomer: it operates on a
directory of sequential JPEGs. That said, at 5 fps with tractor
motion the inter-frame displacement (~10-40 cm of scene shift) is
large enough that SAM2's temporal propagation gives diminishing
returns over running SAM2 image-mode independently per frame and
letting DBSCAN do the temporal-identity work via world coordinates.
Image mode is the default and recommended path. Video mode is kept
available behind ``cfg.propagation_mode = "video"`` for higher-fps
collection (>= 15 fps) where temporal continuity is strong.

Pipeline
--------
1. ``detect_trunks_grounding_dino`` — per-frame zero-shot trunk bboxes
2. ``propagate_with_sam2``         — per-detection precise mask via
                                     SAM2 image mode (default) or
                                     temporal propagation (video mode)
3. ``project_tracks_to_world``     — trunk centroid -> (lat, lon) via
                                     existing ``camera_to_world``
4. ``cluster_to_trees``            — DBSCAN on trunk world positions;
                                     this is where temporal identity
                                     is established in image mode
5. ``attribute_frames``            — frame_idx -> [(tree_id, weight)]
6. ``sanity_check_spacing``        — flag clusters that violate row geom
7. ``build_contact_sheets``        — verification artifact for review
8. ``populate_tree_tracker``       — write results back to TreeTracker

Setup
-----
    pip install torch torchvision transformers scikit-learn
    pip install opencv-python pillow numpy tqdm
    # SAM2 from Meta:
    pip install git+https://github.com/facebookresearch/sam2.git

The Grounding DINO + SAM2 model weights download automatically on
first run (~1.5 GB total). Both models live on GPU; one frame at a
time fits comfortably in 16 GB VRAM at 1280x720.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Geometry helpers — implemented locally so the script runs standalone
# without needing tree_tracker on the Python path.
# populate_tree_tracker() still imports tree_tracker lazily if you use it.

def camera_to_world(
    gps_lat: float,
    gps_lon: float,
    heading_deg: float,
    depth_m: float,
    lateral_offset_m: float = 0.0,
) -> Tuple[float, float]:
    """Project a trunk depth observation to world (lat, lon).

    Assumes the camera faces the direction of travel (heading_deg).
    depth_m is the distance to the trunk. lateral_offset_m shifts the
    effective origin laterally (positive = right of travel direction).
    """
    h = math.radians(heading_deg)
    # Forward component (into the row).
    dx = math.sin(h) * depth_m        # east, metres
    dy = math.cos(h) * depth_m        # north, metres
    # Lateral offset: camera mounted right/left of GPS antenna.
    h_right = h + math.pi / 2
    dx += math.sin(h_right) * lateral_offset_m
    dy += math.cos(h_right) * lateral_offset_m
    # Metres → degrees.
    cos_lat = math.cos(math.radians(gps_lat)) or 1e-9
    return gps_lat + dy / 111_320.0, gps_lon + dx / (111_320.0 * cos_lat)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


log = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class SegmenterConfig:
    """All tuneable parameters for the post-processing pipeline.

    Defaults are calibrated for tractor-mounted RealSense D455 at 5 fps
    in central-leader / spindle apple orchards (Gala, Golden Delicious,
    Sun Fuji, Old Block). Override per-orchard as needed.
    """

    # ── Grounding DINO ─────────────────────────────────────────────
    gdino_model_id: str = "IDEA-Research/grounding-dino-base"
    # Two prompts joined by "." — the first catches clean trunks against
    # row middles, the second catches partially-occluded trunks. SAM2
    # only needs ONE good detection per trunk to seed the propagation,
    # so prompt redundancy + high recall is more useful than precision.
    text_prompt: str = "a tree trunk. a wooden trunk."
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    # Reject detections whose bbox is implausibly small/large for a
    # trunk at typical RealSense distances (1-4 m).
    min_box_area_frac: float = 0.001    # 0.1% of frame area
    max_box_area_frac: float = 0.20     # 20% of frame area
    # Trunks are tall and narrow — drop fat blobs (likely canopy).
    min_aspect_ratio: float = 1.5       # height / width
    # Reject detections whose median bbox depth is out of plausible
    # range. Flowers from analyze_days.py use 0.6-3.0 m for canopy;
    # trunks can sit a bit further back as the camera approaches but
    # anything > 6 m is a background object (other rows, barns).
    trunk_min_depth_m: float = 0.3
    trunk_max_depth_m: float = 15.0
    # When mask depth is unmeasurable (RealSense fails on thin trunks
    # against bright sky / sparse foliage), fall back to this nominal
    # perpendicular distance from camera to the sprayed row. The
    # tree's projected world position then uses pixel-x angle *
    # nominal_distance to compute the along-row offset, instead of
    # actual depth. Set to 0 to disable the fallback (strict depth
    # only, will discard most thin-trunk detections).
    nominal_tree_distance_m: float = 1.5
    # Canopy depth band is computed *per-frame* from the PRGB ROI's
    # actual depth distribution, not a hardcoded number. These are
    # the safety bounds applied around the data-driven target depth:
    #
    #   target_depth_m = median of valid depths inside PRGB ROI
    #   half_band_m = max(canopy_band_min_half_m,
    #                     iqr_of_roi_depths * canopy_band_iqr_mult)
    #   half_band_m = min(half_band_m, canopy_band_max_half_m)
    #
    # so the band is wide enough for the actual canopy thickness
    # (IQR captures depth spread of leaves at varying distances)
    # but capped to keep ground / background out.
    canopy_band_min_half_m: float = 0.30   # ≥ 0.6 m total band
    canopy_band_max_half_m: float = 1.20   # ≤ 2.4 m total band
    canopy_band_iqr_mult: float = 2.0      # band ~= 4·IQR around median
    # Text prompt for the SAM 3 second pass that segments the whole
    # tree (canopy + trunk) containing the GDINO-detected trunk box.
    # The combined text + box prompt lets SAM 3 separate touching
    # trees by trunk identity instead of by canopy boundary.
    tree_text_prompt: str = "apple tree"
    # Require the GDINO bbox to overlap the PRGB ROI by at least this
    # fraction of the bbox. The ROI marks the tree currently being
    # sprayed; trunks outside it are background. Set to 0 to disable.
    trunk_min_roi_overlap: float = 0.10

    # ── SAM2 (image mode = default, video mode = optional) ───────
    # Image mode runs SAM2 on each Grounding DINO bbox independently
    # and lets DBSCAN establish temporal identity via world coords.
    # This is the right choice for 5 fps capture: inter-frame
    # displacement is too large for video propagation to add much,
    # and image mode is simpler / more robust to detection gaps.
    propagation_mode: str = "image"     # {"image", "video"}
    sam2_model_id: str = "facebook/sam2.1-hiera-large"   # HF hub
    # Used only when propagation_mode == "video" (>=15 fps capture):
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    bidirectional: bool = True
    # If a trunk has fewer than this many world-projected detections
    # within DBSCAN's eps, it gets dropped as DBSCAN noise. Tune via
    # dbscan_min_samples below; this is a separate per-track filter
    # used only in video mode.
    min_track_length: int = 4

    # ── World projection ──────────────────────────────────────────
    # Camera mounting offset from GPS antenna (metres). Lateral offset
    # is positive to the right of travel direction.
    camera_lateral_offset_m: float = 0.0
    # Camera horizontal field of view (degrees). Used to convert a
    # detection's pixel-x to a true lateral offset in world coords:
    # a trunk seen at the left edge of a 60°-HFOV frame at depth 4 m
    # has lateral_offset ≈ -2.3 m, not 0. Without this every trunk
    # in a frame projects to the same point, and a single tree's
    # detections smear several metres along the row as the camera
    # passes by — DBSCAN can't reassemble them. RealSense D435 RGB
    # is ~69°; default 60° is a conservative pick that under-shoots
    # rather than over-shoots lateral spread.
    # RealSense D455F datasheet HFOV (sprayer_pipeline.config.CAMERA_HFOV_DEG).
    camera_hfov_deg: float = 87.0
    # Use the median depth inside the trunk mask rather than the mean —
    # robust against the few aberrant depth pixels every D455 frame has.
    depth_estimator: str = "median"     # {"median", "mean", "p25"}

    # ── DBSCAN clustering ─────────────────────────────────────────
    # eps_m: maximum distance (metres) between trunk detections of the
    # same physical tree. When None (default) the eps is auto-estimated
    # from the data using the k-distance graph elbow — it finds the gap
    # between within-tree jitter and between-tree spacing automatically,
    # so it adapts to whatever spacing the orchard actually has without
    # needing a manual tree_spacing_m parameter.
    # Set to a float to override (useful if auto-estimation misbehaves on
    # very sparse or very dense detection sets).
    dbscan_eps_m: Optional[float] = None   # None = auto from data
    dbscan_min_samples: int = 2

    # tree_spacing_m is kept for the spacing sanity check only.
    tree_spacing_m: float = 3.0

    # ── Spacing sanity check ──────────────────────────────────────
    # Disabled by default because tree spacing varies across spray-variable
    # orchards. Enable with spacing_check=True to flag gaps that deviate
    # from tree_spacing_m by more than spacing_tolerance.
    spacing_check: bool = False
    spacing_tolerance: float = 0.50     # ±50%

    # ── Frame attribution ────────────────────────────────────────
    # A frame is "attributed" to a tree if the tree's mask covers at
    # least this fraction of the *largest* visible mask in that frame.
    # This handles the common case where 2-3 trees are partially
    # visible: the dominant one wins primary attribution, others get
    # listed as secondary.
    attribution_dominance_threshold: float = 0.30

    # ── Verification artifacts ───────────────────────────────────
    contact_sheet_thumb_size: int = 256
    contact_sheet_max_thumbs: int = 24


# ============================================================================
# Data classes
# ============================================================================
@dataclass
class TrunkDetection:
    """One trunk observation in one frame.

    Populated incrementally: detection stage fills bbox/score/mask;
    projection stage adds (world_lat, world_lon, depth_m).
    """

    frame_idx: int
    bbox_xyxy: Tuple[float, float, float, float]   # pixels, image coords
    score: float
    mask: Optional[np.ndarray] = None              # bool array, frame-shape
    # Whole-tree (canopy + trunk) mask from a second SAM 3 pass with
    # text prompt "apple tree". Falls back to None if SAM 3 is
    # unavailable or returns nothing usable.
    tree_mask: Optional[np.ndarray] = None
    depth_m: float = 0.0
    world_lat: float = 0.0
    world_lon: float = 0.0
    # Attached after SAM2 propagation:
    track_id: int = -1


@dataclass
class TrunkTrack:
    """One trunk identity across a contiguous span of frames.

    Output of SAM2 video propagation. Each track will be projected to
    world coordinates and clustered; a single tree may span 1+ tracks
    if SAM2 lost the mask mid-traversal (the DBSCAN step re-merges).
    """

    track_id: int
    detections: List[TrunkDetection] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.detections)

    @property
    def median_world(self) -> Tuple[float, float]:
        if not self.detections:
            return (0.0, 0.0)
        lats = np.array([d.world_lat for d in self.detections])
        lons = np.array([d.world_lon for d in self.detections])
        return float(np.median(lats)), float(np.median(lons))

    @property
    def frame_range(self) -> Tuple[int, int]:
        if not self.detections:
            return (-1, -1)
        idx = [d.frame_idx for d in self.detections]
        return min(idx), max(idx)


@dataclass
class TreeCluster:
    """Persistent tree identity after DBSCAN clustering.

    Replaces ``TrackedTree`` for the post-processing output, with
    direct conversion via :func:`to_tracked_tree`.
    """

    tree_id: int
    world_lat: float
    world_lon: float
    tracks: List[TrunkTrack] = field(default_factory=list)
    # Frames where this tree's mask was visible, with the per-frame
    # mask pixel count (used as the "dominance weight" downstream).
    frame_pixels: Dict[int, int] = field(default_factory=dict)
    # Diagnostic flag set by sanity_check_spacing.
    flagged_reason: Optional[str] = None
    # Filled by attribute_flowers_via_roi(); total flowers attributed to this tree.
    flower_count: float = 0.0
    # Beer-Lambert LAI per cluster: -ln(gap_fraction)/k where gap is
    # the fraction of pixels in the tree's bounding column NOT inside
    # the per-frame tree_mask. Averaged over the frames where this
    # cluster has a tree_mask. NaN if no tree masks were available.
    lai_beer_lambert: float = 0.0
    # Per-ROI Beer-Lambert LAI: list of 10 LAI values, one per
    # sprayer ROI zone (2 cols x 5 rows = 10 zones, indexed
    # zone = side * 5 + row, side 0=LEFT, side 1=RIGHT, row 0=top).
    # Averaged across the cluster's frames; NaN entries where no
    # tree mask covered that zone in any frame.
    lai_per_roi: List[float] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frame_pixels)

    @property
    def first_frame(self) -> int:
        return min(self.frame_pixels) if self.frame_pixels else -1

    @property
    def last_frame(self) -> int:
        return max(self.frame_pixels) if self.frame_pixels else -1


# ============================================================================
# Frame-loading abstraction
# ============================================================================
class FrameLoader:
    """Plug-in interface for pulling RGB + depth + metadata per frame.

    Your existing pipeline almost certainly has its own data layout
    (BAG file replay, PNG sequence + JSON sidecars, NPY arrays, etc.).
    Subclass this and implement the four methods, or use the included
    :class:`PNGSequenceLoader` for the standard "frames as files" case.
    """

    def __len__(self) -> int:                                  # noqa: D401
        raise NotImplementedError

    def frame_indices(self) -> List[int]:
        """Frame indices in chronological order."""
        raise NotImplementedError

    def load_rgb(self, frame_idx: int) -> np.ndarray:
        """Return an HxWx3 uint8 RGB image."""
        raise NotImplementedError

    def load_depth_m(self, frame_idx: int) -> np.ndarray:
        """Return an HxW float32 depth-in-metres image (NaN for invalid)."""
        raise NotImplementedError

    def load_meta(self, frame_idx: int) -> Dict:
        """Return the per-frame metadata dict.

        Required keys: ``timestamp``, ``gps_lat``, ``gps_lon``,
        ``heading_deg``, ``estimated_lai``. Anything else is forwarded
        verbatim into the resulting TreeDetection.meta.
        """
        raise NotImplementedError

    def load_roi_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return a bool (H, W) PRGB ROI mask, or None if unavailable.

        True pixels are inside the red-box region drawn by the sprayer
        pipeline for this frame. Override in your FrameLoader subclass
        to enable ROI-based flower attribution. The default always
        returns None (no ROI available).
        """
        return None


class All2023FrameLoader(FrameLoader):
    """FrameLoader for the All2023 orchard dataset layout.

    Reads the native session directory produced by the sprayer pipeline —
    no frame extraction needed. Directory structure expected::

        <session_dir>/
            RGB/<timestamp>-RGB-BP.bmp     uint8 RGB
            depth/<timestamp>-Depth.bmp    16-bit depth in mm
            PRGB/<timestamp>-RGB-PP.bmp    sprayer ROI overlay
            <timestamp>.txt                GPS + metadata sidecar

    GPS is parsed from the NMEA line in each sidecar .txt file.
    Row heading is estimated from the GPS trail of the full session
    (bearing from first to last valid GPS fix) unless you override it
    with ``row_heading_deg``. For a straight orchard row the GPS-trail
    estimate is reliable to ±5°, which is enough for DBSCAN clustering.

    Parameters
    ----------
    session_dir
        Path to one session folder (the one that contains RGB/, depth/, PRGB/).
    row_heading_deg
        Compass bearing (0 = north, 90 = east) of the orchard row. When
        None (default) the heading is estimated from the GPS trail.
    frame_range
        Optional (start, stop) frame indices (0-based, stop exclusive) to
        match the ``--frame-range`` used in analyze_days.py.
    """

    _DEPTH_MM_TO_M: float = 1e-3   # RealSense stores depth as uint16 mm

    def __init__(
        self,
        session_dir: str,
        row_heading_deg: Optional[float] = None,
        frame_range: Optional[Tuple[int, int]] = None,
        require_all_modalities: bool = True,
    ):
        import re as _re
        self._session_dir = Path(session_dir)
        self._rgb_dir = self._session_dir / "RGB"
        self._depth_dir = self._session_dir / "depth"
        self._prgb_dir = self._session_dir / "PRGB"
        self._info_dir = self._session_dir / "Info"

        all_imgs = sorted(
            p for p in self._rgb_dir.iterdir()
            if p.suffix.lower() in {".bmp", ".jpg", ".png"}
        )

        if require_all_modalities:
            kept: List[Path] = []
            n_drop_depth = n_drop_prgb = n_drop_info = 0
            for p in all_imgs:
                base = self._base_stem(p)
                has_depth = (
                    (self._depth_dir / f"{base}-Depth.txt").is_file()
                    or (self._depth_dir / f"{base}-Depth.bmp").is_file()
                )
                has_prgb = (
                    self._prgb_dir / f"{base}-RGB-PP.bmp"
                ).is_file()
                has_info = (
                    self._info_dir.is_dir()
                    and any(self._info_dir.glob(f"{base}*.txt"))
                )
                if not has_depth:
                    n_drop_depth += 1; continue
                if not has_prgb:
                    n_drop_prgb += 1; continue
                if not has_info:
                    n_drop_info += 1; continue
                kept.append(p)
            n_dropped = len(all_imgs) - len(kept)
            if n_dropped:
                log.info(
                    "All2023FrameLoader: dropped %d/%d frames lacking "
                    "modalities (no-depth=%d no-prgb=%d no-info=%d)",
                    n_dropped, len(all_imgs),
                    n_drop_depth, n_drop_prgb, n_drop_info,
                )
            all_imgs = kept

        # Slice AFTER modality filtering so --frame-range 1 100 means
        # "the first 100 fully-modal frames in timestamp order", not
        # "frames 1-100 of which some may be dropped".
        if frame_range is not None:
            a, b = frame_range
            all_imgs = all_imgs[max(0, a): b]

        self._imgs = all_imgs

        # Parse every sidecar up front (fast — small text files).
        self._sidecars: List[Dict] = [
            self._parse_sidecar(p) for p in self._imgs
        ]

        if row_heading_deg is not None:
            self._heading = float(row_heading_deg)
        else:
            self._heading = self._estimate_heading()

        log.info(
            "All2023FrameLoader: %d frames in '%s', heading=%.1f°",
            len(self._imgs), self._session_dir.name, self._heading,
        )

    # ── FrameLoader interface ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._imgs)

    def frame_indices(self) -> List[int]:
        return list(range(len(self._imgs)))

    def load_rgb(self, frame_idx: int) -> np.ndarray:
        import cv2
        path = self._imgs[frame_idx]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Missing RGB: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def load_depth_m(self, frame_idx: int) -> np.ndarray:
        import cv2
        base = self._base_stem(self._imgs[frame_idx])
        h, w = self.load_rgb(frame_idx).shape[:2]

        def _resize_to_rgb(d: np.ndarray) -> np.ndarray:
            # All2023 depth .txt files are stored decimated horizontally
            # (480 rows × ~50 cols for a 640×480 RGB frame). Bilinearly
            # upsample to the RGB resolution so masks line up.
            if d.shape == (h, w):
                return d
            from PIL import Image as _PILImage
            pim = _PILImage.fromarray(
                d.astype(np.float32), mode="F",
            ).resize((w, h), _PILImage.BILINEAR)
            return np.asarray(pim, dtype=np.float32)

        # Prefer the .txt sidecar — that's the raw uint16 mm depth.
        # The .bmp may be a colormap visualization (0-255 8-bit) that
        # looks like depth but isn't, mirroring analyze_days.py's
        # load_depth_mm guard.
        txt = self._depth_dir / f"{base}-Depth.txt"
        if txt.is_file():
            d = np.loadtxt(str(txt), dtype=np.float32)
            return _resize_to_rgb(d) * self._DEPTH_MM_TO_M
        bmp = self._depth_dir / f"{base}-Depth.bmp"
        if bmp.is_file():
            raw = cv2.imread(str(bmp), cv2.IMREAD_UNCHANGED)
            if raw is not None and raw.ndim == 2 and raw.dtype == np.uint16:
                return _resize_to_rgb(raw.astype(np.float32)) * self._DEPTH_MM_TO_M
            if not getattr(self, "_warned_bad_depth_bmp", False):
                log.warning(
                    "Depth .bmp at %s is %s %s — looks like the colormap "
                    "visualization, not raw mm. Using .txt sidecar if "
                    "present; otherwise depth is unavailable.",
                    bmp.parent,
                    None if raw is None else raw.shape,
                    None if raw is None else raw.dtype,
                )
                self._warned_bad_depth_bmp = True
        # Missing depth → all NaN; projection skips this frame gracefully.
        log.warning("No depth file for frame %d (%s)", frame_idx, base)
        return np.full((h, w), float("nan"), dtype=np.float32)

    def load_meta(self, frame_idx: int) -> Dict:
        m = dict(self._sidecars[frame_idx])
        m.setdefault("heading_deg", self._heading)
        m.setdefault("estimated_lai", 0.0)
        m.setdefault("timestamp", float(frame_idx))
        return m

    def load_roi_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        try:
            from analyze_days import extract_roi_mask
        except ImportError:
            if not getattr(self, "_warned_roi_import", False):
                log.warning(
                    "analyze_days not importable; ROI masks unavailable "
                    "(3D nearest-tree attribution still works; only the "
                    "ROI-overlap fallback is disabled)",
                )
                self._warned_roi_import = True
            return None
        base = self._base_stem(self._imgs[frame_idx])
        prgb_path = self._prgb_dir / f"{base}-RGB-PP.bmp"
        rgb = self.load_rgb(frame_idx)
        h, w = rgb.shape[:2]
        roi = extract_roi_mask(prgb_path, (h, w))
        if roi is None or not roi.any():
            return roi
        # Mirror analyze_days.py --prgb-extend-vertical: stretch the
        # ROI to full image height. Trunks span vertically while the
        # red sprayer ROI is short (~20% of frame); without this,
        # trunks have low overlap fraction by construction.
        cols = roi.any(axis=0)
        if cols.any():
            x_idx = np.where(cols)[0]
            x1, x2 = int(x_idx.min()), int(x_idx.max()) + 1
            tall = np.zeros_like(roi)
            tall[:, x1:x2] = True
            roi = tall
        return roi

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _base_stem(img_path: Path) -> str:
        """Strip the -RGB-BP suffix (and variants) from the image stem."""
        stem = img_path.stem
        for sfx in ("-RGB-BP", "-RGB-bp", "-RGB", "-rgb"):
            if stem.endswith(sfx):
                return stem[: -len(sfx)]
        return stem

    def _parse_sidecar(self, img_path: Path) -> Dict:
        import re as _re
        base = self._base_stem(img_path)
        # Try a few common layouts: <Info>/<base>.txt, then with -RGB-BP suffix
        # restored, then in the session root, then a glob over Info/.
        info_dir = self._session_dir / "Info"
        candidates = [
            info_dir / f"{base}.txt",
            info_dir / f"{base}-RGB-BP.txt",
            info_dir / f"{base}-RGB.txt",
            info_dir / f"{base}-Info.txt",
            self._session_dir / f"{base}.txt",
        ]
        txt = next((p for p in candidates if p.is_file()), None)
        if txt is None and info_dir.is_dir():
            for p in info_dir.glob(f"{base}*.txt"):
                txt = p
                break

        meta: Dict = {}
        if txt is None:
            if not getattr(self, "_warned_missing_sidecar", False):
                log.warning(
                    "No sidecar .txt found for %s. Tried: %s. "
                    "Info dir contents: %s",
                    img_path.name,
                    [str(c.relative_to(self._session_dir)) for c in candidates],
                    sorted(p.name for p in info_dir.glob("*.txt"))[:5]
                    if info_dir.is_dir() else "<no Info/ dir>",
                )
                self._warned_missing_sidecar = True
            return meta

        content = txt.read_text(errors="replace")

        # GPS Code (NMEA format): N 4044.17959, W 08154.19095
        # (when the GPS is locked and reporting a position fix)
        gps_m = _re.search(
            r"GPS Code \(NMEA format\):\s*([NS])\s+([\d.]+),\s*([EW])\s+([\d.]+)",
            content,
        )
        if gps_m:
            hN, lat_v, hE, lon_v = gps_m.groups()
            lat_deg = int(float(lat_v) / 100)
            lat = lat_deg + (float(lat_v) - lat_deg * 100) / 60.0
            if hN == "S":
                lat = -lat
            lon_deg = int(float(lon_v) / 100)
            lon = lon_deg + (float(lon_v) - lon_deg * 100) / 60.0
            if hE == "W":
                lon = -lon
            meta["gps_lat"] = lat
            meta["gps_lon"] = lon

        # Fallback: parse Raw GPS data line for $GPGGA or $GPRMC sentences.
        # These appear when the GPS Code line has a u-blox text message
        # instead of coordinates (GPS not yet locked on that frame).
        if "gps_lat" not in meta:
            raw_m = _re.search(r"Raw GPS data:\s*(.+)", content)
            if raw_m:
                raw = raw_m.group(1)
                # $GPGGA,HHMMSS,DDMM.MMMM,N,DDDMM.MMMM,W,...
                gg = _re.search(
                    r"\$GPGGA,[\d.]*,([\d.]+),([NS]),([\d.]+),([EW])",
                    raw,
                )
                # $GPRMC,HHMMSS,A,DDMM.MMMM,N,DDDMM.MMMM,W,...
                rm = gg or _re.search(
                    r"\$GPRMC,[\d.]*,[AV],([\d.]+),([NS]),([\d.]+),([EW])",
                    raw,
                )
                if rm:
                    lat_v, hN, lon_v, hE = rm.groups()
                    lat_deg = int(float(lat_v) / 100)
                    lat = lat_deg + (float(lat_v) - lat_deg * 100) / 60.0
                    if hN == "S":
                        lat = -lat
                    lon_deg = int(float(lon_v) / 100)
                    lon = lon_deg + (float(lon_v) - lon_deg * 100) / 60.0
                    if hE == "W":
                        lon = -lon
                    meta["gps_lat"] = lat
                    meta["gps_lon"] = lon

        if "gps_lat" not in meta and not getattr(
            self, "_warned_no_gps_match", False,
        ):
            gps_lines = [
                ln for ln in content.splitlines()
                if "gps" in ln.lower() or "nmea" in ln.lower()
                or ln.strip().startswith("$")
            ]
            # An empty "GPS Code (NMEA format): , " line means the
            # GPS hadn't locked yet -- not a parser bug, just no
            # data. Suppress the warning in that case.
            empty_gps = any(
                _re.search(r"GPS Code \(NMEA format\):\s*,?\s*$", ln)
                for ln in gps_lines
            )
            if not empty_gps:
                log.warning(
                    "GPS regex did not match in %s. GPS-related lines: %r",
                    txt.name,
                    gps_lines[:5],
                )
            self._warned_no_gps_match = True

        # Travel Speed: 3 mph
        spd = _re.search(r"Travel Speed:\s*([\d.]+)\s*mph", content)
        if spd:
            meta["travel_speed_mph"] = float(spd.group(1))

        # Capture Time: 204  (ms since some epoch; convert to seconds)
        ct = _re.search(r"Capture Time:\s*(\d+)", content)
        if ct:
            meta["timestamp"] = int(ct.group(1)) / 1000.0

        return meta

    def _estimate_heading(self) -> float:
        """Compute the compass bearing from the first to last GPS fix."""
        lats = [m["gps_lat"] for m in self._sidecars if "gps_lat" in m]
        lons = [m["gps_lon"] for m in self._sidecars if "gps_lon" in m]
        n_total = len(self._sidecars)
        if len(lats) < 2:
            log.warning(
                "Too few GPS fixes in session (%d/%d frames had coordinates) — "
                "GPS likely not locked. Heading defaults to 0°. "
                "Pass --row-heading-deg to override.",
                len(lats), n_total,
            )
            return 0.0
        lat1, lon1 = math.radians(lats[0]), math.radians(lons[0])
        lat2, lon2 = math.radians(lats[-1]), math.radians(lons[-1])
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2)
             - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        bearing = math.degrees(math.atan2(x, y)) % 360.0
        log.info("GPS-trail heading estimate: %.1f°", bearing)
        return bearing


class PNGSequenceLoader(FrameLoader):
    """Standard layout: frames extracted to ``rgb/{idx:06d}.png`` etc.

    Directory structure expected::

        run_root/
            rgb/000000.png      # uint8 RGB
            depth/000000.npy    # float32 metres, HxW
            meta.json           # {"frames": [{"frame_idx": 0, ...}, ...]}

    If your dataset is in a BAG file, use the RealSense SDK to pre-
    extract once -- SAM2 needs frames-as-files anyway because its
    video predictor takes a directory of JPEGs as input.
    """

    def __init__(self, run_root: str):
        self.root = Path(run_root)
        meta_path = self.root / "meta.json"
        with meta_path.open("r") as fh:
            payload = json.load(fh)
        self._meta_by_idx: Dict[int, Dict] = {
            int(f["frame_idx"]): f for f in payload["frames"]
        }
        self._indices = sorted(self._meta_by_idx)

    def __len__(self) -> int:
        return len(self._indices)

    def frame_indices(self) -> List[int]:
        return list(self._indices)

    def load_rgb(self, frame_idx: int) -> np.ndarray:
        import cv2
        path = self.root / "rgb" / f"{frame_idx:06d}.png"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Missing RGB frame: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def load_depth_m(self, frame_idx: int) -> np.ndarray:
        path = self.root / "depth" / f"{frame_idx:06d}.npy"
        return np.load(path).astype(np.float32)

    def load_meta(self, frame_idx: int) -> Dict:
        return self._meta_by_idx[int(frame_idx)]


# ============================================================================
# Stage 1: Grounding DINO trunk detection
# ============================================================================
def detect_trunks_grounding_dino(
    loader: FrameLoader,
    cfg: SegmenterConfig,
    device: str = "cuda",
) -> Dict[int, List[TrunkDetection]]:
    """Run Grounding DINO trunk detection on every frame.

    Returns a dict mapping ``frame_idx -> List[TrunkDetection]``.
    Detections at this stage have bbox + score only; mask, depth,
    and world coords are populated in later stages.

    The HF ``grounding-dino-base`` checkpoint handles "tree trunk" as
    a zero-shot prompt remarkably well in orchard imagery. On Old
    Block-style canopies with low limbs, increase ``box_threshold``
    to 0.30 and add ``"trunk near ground"`` to the prompt to bias
    toward the lower portion of the image.
    """
    import torch
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )
    from tqdm.auto import tqdm

    log.info("Loading Grounding DINO: %s", cfg.gdino_model_id)
    processor = AutoProcessor.from_pretrained(cfg.gdino_model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        cfg.gdino_model_id
    ).to(device)
    model.eval()

    detections_by_frame: Dict[int, List[TrunkDetection]] = {}
    n_drop_score = n_drop_area = n_drop_aspect = n_drop_roi = 0

    for frame_idx in tqdm(loader.frame_indices(), desc="GDINO trunk detect"):
        rgb = loader.load_rgb(frame_idx)
        h, w = rgb.shape[:2]
        frame_area = float(h * w)
        roi = loader.load_roi_mask(frame_idx) if (
            cfg.trunk_min_roi_overlap > 0
        ) else None

        inputs = processor(
            images=rgb,
            text=cfg.text_prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        try:
            # transformers >= 4.38 accepts thresholds directly.
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=cfg.box_threshold,
                text_threshold=cfg.text_threshold,
                target_sizes=[(h, w)],
            )[0]
        except TypeError:
            # Older / newer API variant — get all boxes and filter manually.
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[(h, w)],
            )[0]
            keep = (results["scores"] >= cfg.box_threshold).nonzero(
                as_tuple=False
            ).squeeze(1)
            results["boxes"] = results["boxes"][keep]
            results["scores"] = results["scores"][keep]

        kept: List[TrunkDetection] = []
        for box, score in zip(results["boxes"], results["scores"]):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            area = bw * bh
            area_frac = area / frame_area
            if area_frac < cfg.min_box_area_frac or area_frac > cfg.max_box_area_frac:
                n_drop_area += 1
                continue
            aspect = bh / bw
            if aspect < cfg.min_aspect_ratio:
                n_drop_aspect += 1
                continue

            # ROI gate: bbox must overlap the PRGB ROI by at least
            # trunk_min_roi_overlap of the bbox area. The ROI marks
            # the tree currently being sprayed; trunks outside it
            # are background.
            if roi is not None and roi.any():
                xi1 = max(0, int(round(x1)))
                yi1 = max(0, int(round(y1)))
                xi2 = min(w, int(round(x2)))
                yi2 = min(h, int(round(y2)))
                if xi2 <= xi1 or yi2 <= yi1:
                    continue
                roi_crop = roi[yi1:yi2, xi1:xi2]
                box_pixels = (xi2 - xi1) * (yi2 - yi1)
                overlap = float(roi_crop.sum()) / float(box_pixels)
                if overlap < cfg.trunk_min_roi_overlap:
                    n_drop_roi += 1
                    continue

            kept.append(TrunkDetection(
                frame_idx=frame_idx,
                bbox_xyxy=(x1, y1, x2, y2),
                score=float(score),
            ))

        detections_by_frame[frame_idx] = kept

    n_total = sum(len(v) for v in detections_by_frame.values())
    log.info(
        "GDINO: %d trunk detections across %d frames "
        "(dropped: area=%d aspect=%d roi=%d)",
        n_total, len(detections_by_frame),
        n_drop_area, n_drop_aspect, n_drop_roi,
    )
    return detections_by_frame


# ============================================================================
# Stage 2: SAM2 bidirectional propagation
# ============================================================================
def _select_seed_detections(
    detections_by_frame: Dict[int, List[TrunkDetection]],
    cfg: SegmenterConfig,
) -> List[TrunkDetection]:
    """Pick the best seed detection per likely-distinct trunk.

    Heuristic: cluster detections by (frame_idx, bbox horizontal
    centre) and keep the highest-confidence representative. Two
    detections in *adjacent* frames at similar horizontal positions
    are almost certainly the same trunk -- SAM2 will tie them
    together via propagation, so we only seed once.

    This avoids re-seeding the same trunk every frame, which would
    fragment the SAM2 tracks and bloat the cluster step.
    """
    # Sort by (frame_idx, score desc) so the highest-conf det wins
    # when two seeds collide.
    all_dets: List[TrunkDetection] = []
    for fid in sorted(detections_by_frame):
        for det in detections_by_frame[fid]:
            all_dets.append(det)

    seeds: List[TrunkDetection] = []
    for det in all_dets:
        cx = (det.bbox_xyxy[0] + det.bbox_xyxy[2]) / 2.0
        is_dup = False
        for kept in seeds:
            if abs(kept.frame_idx - det.frame_idx) > 3:
                continue
            kcx = (kept.bbox_xyxy[0] + kept.bbox_xyxy[2]) / 2.0
            # Within 60 px horizontally and 3 frames -> same trunk.
            if abs(kcx - cx) < 60.0:
                is_dup = True
                if det.score > kept.score:
                    # Replace with higher-confidence seed.
                    seeds.remove(kept)
                    seeds.append(det)
                break
        if not is_dup:
            seeds.append(det)

    seeds.sort(key=lambda d: (d.frame_idx, -d.score))
    log.info("Selected %d seed detections from %d raw", len(seeds), len(all_dets))
    return seeds


def _stage_jpegs_for_sam2(
    loader: FrameLoader,
    work_dir: Path,
) -> List[int]:
    """SAM2 video predictor needs a directory of JPEGs named NNNNN.jpg.

    Returns the frame_indices in order. Names them 00000.jpg ... so
    SAM2's internal sorting matches our chronological order. Caller
    is responsible for ``shutil.rmtree(work_dir)``.
    """
    import cv2

    work_dir.mkdir(parents=True, exist_ok=True)
    indices = loader.frame_indices()
    for sam_idx, frame_idx in enumerate(indices):
        rgb = loader.load_rgb(frame_idx)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(work_dir / f"{sam_idx:05d}.jpg"), bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return indices


def propagate_with_sam2(
    loader: FrameLoader,
    detections_by_frame: Dict[int, List[TrunkDetection]],
    cfg: SegmenterConfig,
    device: str = "cuda",
    work_dir: Optional[Path] = None,
) -> List[TrunkTrack]:
    """Refine Grounding DINO bboxes into precise trunk masks.

    Dispatches to image mode (default, recommended for <=5 fps) or
    video mode (recommended for >=15 fps) based on
    ``cfg.propagation_mode``.

    In image mode, each detection becomes a single-frame "track" --
    DBSCAN clustering in world coordinates is what establishes
    temporal identity (i.e. which detections belong to the same
    physical tree). For 5 fps tractor data this is more reliable
    than SAM2 video propagation because inter-frame displacement
    exceeds SAM2's strong-prior regime.
    """
    mode = cfg.propagation_mode
    if mode == "image":
        return _propagate_image_mode(loader, detections_by_frame, cfg, device)
    if mode == "video":
        return _propagate_video_mode(
            loader, detections_by_frame, cfg, device, work_dir,
        )
    raise ValueError(
        f"propagation_mode must be 'image' or 'video', got {mode!r}"
    )


def _propagate_image_mode(
    loader: FrameLoader,
    detections_by_frame: Dict[int, List[TrunkDetection]],
    cfg: SegmenterConfig,
    device: str = "cuda",
) -> List[TrunkTrack]:
    """SAM2 image-mode mask refinement, one detection at a time.

    For 5 fps tractor capture this is the recommended path. Every
    Grounding DINO detection across every frame gets its own
    SAM2-refined mask, then becomes a single-detection TrunkTrack.
    DBSCAN over the resulting world-coordinate cloud is what
    establishes "this detection in frame 12 and that one in frame
    14 are the same physical tree" -- spatial proximity in world
    coords is a stronger identity signal than mask-pixel continuity
    when frames are 200 ms apart.

    Each track here holds exactly one detection. The cluster step
    groups them by world position into per-tree TreeClusters.

    Tries SAM 2 first (box-prompt API). If unavailable, tries SAM 3
    (Meta) via ``Sam3Processor.add_geometric_prompt`` -- SAM 3 also
    accepts a box prompt and returns a refined mask. As a last
    resort falls back to using the GDINO bbox as a rectangular
    mask, which keeps depth/ROI/DBSCAN working but skips pixel-
    level refinement.
    """
    from tqdm.auto import tqdm
    from PIL import Image

    sam2_predictor = None
    sam3_processor = None
    autocast_dtype = None
    backend = "bbox"

    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        log.info("Loading SAM2 (image mode): %s", cfg.sam2_model_id)
        sam2_predictor = SAM2ImagePredictor.from_pretrained(
            cfg.sam2_model_id, device=device,
        )
        autocast_dtype = (
            torch.bfloat16 if device.startswith("cuda") else torch.float32
        )
        backend = "sam2"
    except ImportError:
        try:
            import torch  # noqa: F401
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            log.info("Loading SAM 3 (geometric box prompt) for trunk masks")
            sam3_model = build_sam3_image_model()
            # Low confidence threshold so SAM 3 returns a mask for
            # essentially every GDINO box we feed it. We trust the
            # GDINO detection -- SAM 3 only needs to produce the
            # pixel mask.
            sam3_processor = Sam3Processor(
                sam3_model, device=device, confidence_threshold=0.05,
            )
            backend = "sam3"
        except ImportError:
            log.warning(
                "Neither sam2 nor sam3 importable — falling back to "
                "bbox-rectangle masks for trunk detections.",
            )

    tracks: List[TrunkTrack] = []
    next_track_id = 1

    def _bbox_mask(box, h: int, w: int) -> np.ndarray:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
        m = np.zeros((h, w), dtype=bool)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
        return m

    def _xyxy_to_norm_cxcywh(box, h: int, w: int):
        x1, y1, x2, y2 = (float(v) for v in box)
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return [cx, cy, bw, bh]

    def _pick_sam3_mask(state, gdino_box, h: int, w: int):
        """Pick the SAM 3 mask whose box overlaps the GDINO box most."""
        import torch as _torch
        masks = state.get("masks")
        boxes = state.get("boxes")
        if masks is None or boxes is None or len(masks) == 0:
            return None, 0.0

        def _to_np(t):
            t = t.detach().cpu()
            # numpy doesn't support bfloat16 / float16 -- upcast.
            if t.dtype in (_torch.bfloat16, _torch.float16):
                t = t.to(_torch.float32)
            return t.numpy()

        masks_np = _to_np(masks).astype(bool)
        if masks_np.ndim == 4:
            masks_np = masks_np.squeeze(1)
        boxes_np = _to_np(boxes)
        scores = state.get("scores")
        scores_np = (
            _to_np(scores) if scores is not None
            else np.ones(len(masks_np))
        )
        gx1, gy1, gx2, gy2 = gdino_box
        ga = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
        best_iou = -1.0
        best_idx = -1
        for i, b in enumerate(boxes_np):
            bx1, by1, bx2, by2 = b
            ix1 = max(gx1, bx1); iy1 = max(gy1, by1)
            ix2 = min(gx2, bx2); iy2 = min(gy2, by2)
            iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = ga + ba - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx < 0:
            return None, 0.0
        return masks_np[best_idx], float(scores_np[best_idx])

    desc_map = {
        "sam2": "SAM2 mask refine",
        "sam3": "SAM3 mask refine",
        "bbox": "bbox-mask build",
    }
    n_tree_mask_ok = n_tree_mask_fail = 0
    n_fail_no_canopy = n_fail_no_cc = n_fail_voronoi = n_fail_exception = 0
    n_canopy_built = n_canopy_empty = n_canopy_error = 0
    for frame_idx in tqdm(loader.frame_indices(), desc=desc_map[backend]):
        dets = detections_by_frame.get(frame_idx, [])
        if not dets:
            continue
        rgb = loader.load_rgb(frame_idx)
        h, w = rgb.shape[:2]

        if backend == "bbox":
            for det in dets:
                m = _bbox_mask(det.bbox_xyxy, h, w)
                if not m.any():
                    continue
                track = TrunkTrack(track_id=next_track_id)
                next_track_id += 1
                refined_det = TrunkDetection(
                    frame_idx=frame_idx,
                    bbox_xyxy=det.bbox_xyxy,
                    score=det.score,
                    mask=m,
                    track_id=track.track_id,
                )
                track.detections.append(refined_det)
                tracks.append(track)
            continue

        if backend == "sam3":
            import torch as _torch
            pil_img = Image.fromarray(rgb)
            ac_dtype = (
                _torch.bfloat16 if device.startswith("cuda") else _torch.float32
            )
            with _torch.autocast(
                "cuda" if device.startswith("cuda") else "cpu",
                dtype=ac_dtype,
            ):
                state = sam3_processor.set_image(pil_img)

                # Pre-compute the sprayer-pipeline canopy mask once
                # per frame. This is the canonical build_tree_mask
                # ported verbatim from sprayer_pipeline/tree_mask.py:
                # depth-thresholded, row-band-filtered, sky/grass-
                # excluded via RGB HSV, with anchor-on-ROI-columns
                # and forward-branch recovery. Identical to the mask
                # the user already trusts for flower filtering.
                # We then split the canopy across the frame's trunks
                # via nearest-trunk assignment so touching trees
                # stay separated.
                # Pre-load depth_mm once per frame; build_tree_mask
                # is called per-detection below (anchored on each
                # trunk's pixel-x column) so each tree gets a mask
                # that respects its own position, the way
                # sprayer_pipeline does in single-tree frames.
                depth_mm: Optional[np.ndarray] = None
                target_depth_band: Optional[Tuple[int, int]] = None
                try:
                    depth_m = loader.load_depth_m(frame_idx)
                    if depth_m is not None and np.isfinite(depth_m).any():
                        depth_mm = np.where(
                            np.isfinite(depth_m) & (depth_m > 0),
                            (depth_m * 1000.0).astype(np.uint16),
                            np.zeros_like(depth_m, dtype=np.uint16),
                        )
                except Exception as exc:
                    log.warning(
                        "Depth load failed on frame %d: %s",
                        frame_idx, exc,
                    )
                    depth_mm = None

                for det in dets:
                    # ── Pass 1: trunk-tight mask via box prompt ──
                    sam3_processor.reset_all_prompts(state)
                    norm_box = _xyxy_to_norm_cxcywh(det.bbox_xyxy, h, w)
                    state = sam3_processor.add_geometric_prompt(
                        box=norm_box, label=True, state=state,
                    )
                    trunk_mask, sam3_score = _pick_sam3_mask(
                        state, det.bbox_xyxy, h, w,
                    )
                    if trunk_mask is None or not trunk_mask.any():
                        trunk_mask = _bbox_mask(det.bbox_xyxy, h, w)
                        if not trunk_mask.any():
                            continue
                        sam3_score = 1.0

                    # ── Per-trunk wide tree mask ──
                    # Use sprayer_pipeline.tree_aggregate.
                    # build_wide_tree_mask -- the function the
                    # validated R²=0.564 LAI estimator uses. Its
                    # depth-gradient ground filter (dDepth/dRow > 3
                    # mm/row -> ground) is what kills the horizontal
                    # ground band that build_tree_mask leaves in.
                    # The hard row-420 bottom cutoff is also
                    # standard. After build, restrict to the trunk's
                    # column to isolate this specific tree, then
                    # apply a depth-coherence cut so adjacent trees
                    # at different distances stay separated even
                    # when their canopies overlap in pixel space.
                    tree_mask: Optional[np.ndarray] = None
                    if depth_mm is not None:
                        try:
                            from tree_mask import build_wide_tree_mask
                            wide_u8 = build_wide_tree_mask(
                                depth_mm, rgb,
                                isolate_target_tree=False,
                                apply_gradient_ground_filter=True,
                                apply_blue_cv_sky_filter=True,
                            )
                            wide_bool = (wide_u8 > 0)

                            # Restrict the wide mask to a column band
                            # around THIS trunk so adjacent trees (and
                            # background trees) get their own slice.
                            # No CC pick -- the whole wide mask
                            # within the column is this tree's canopy
                            # (build_wide_tree_mask already excluded
                            # ground via gradient + bottom cutoff;
                            # column restriction handles per-tree
                            # spatial separation).
                            x1b, _, x2b, _ = (
                                int(round(v)) for v in det.bbox_xyxy
                            )
                            col_pad = 80
                            cb0 = max(0, x1b - col_pad)
                            cb1 = min(w, x2b + col_pad)
                            col_band = np.zeros_like(wide_bool)
                            col_band[:, cb0:cb1] = True
                            in_col = wide_bool & col_band

                            # Background-trunk filter: must have at
                            # least 200 px of canopy in this trunk's
                            # column for the detection to be a real
                            # tree. Cars/posts produce nothing.
                            if int(in_col.sum()) < 200:
                                continue

                            # Depth-coherence: use the median depth
                            # inside this trunk's column-canopy as
                            # the tree's distance, keep ±400 mm.
                            valid = in_col & (depth_mm > 0)
                            if valid.any():
                                med = float(np.median(depth_mm[valid]))
                                coh_lo = max(1, int(med - 400))
                                coh_hi = int(med + 400)
                                coherent = (
                                    in_col
                                    & (depth_mm >= coh_lo)
                                    & (depth_mm <= coh_hi)
                                )
                                tree_mask = coherent | trunk_mask
                            else:
                                tree_mask = in_col | trunk_mask

                            if not tree_mask.any():
                                tree_mask = None
                        except Exception as exc:
                            log.warning(
                                "build_wide_tree_mask failed for det "
                                "in frame %d: %s",
                                frame_idx, exc,
                            )
                            tree_mask = None
                    if tree_mask is not None:
                        n_tree_mask_ok += 1
                        n_canopy_built += 1
                    else:
                        n_tree_mask_fail += 1
                        n_fail_no_canopy += 1
                        n_canopy_empty += 1

                    track = TrunkTrack(track_id=next_track_id)
                    next_track_id += 1
                    refined_det = TrunkDetection(
                        frame_idx=frame_idx,
                        bbox_xyxy=det.bbox_xyxy,
                        score=det.score * sam3_score,
                        mask=trunk_mask.astype(bool),
                        tree_mask=tree_mask,
                        track_id=track.track_id,
                    )
                    track.detections.append(refined_det)
                    tracks.append(track)
            continue

        # SAM 2 path
        with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
            sam2_predictor.set_image(rgb)
            for det in dets:
                box = np.array(det.bbox_xyxy, dtype=np.float32)
                masks, scores, _ = sam2_predictor.predict(
                    box=box[None, :],   # (1, 4)
                    multimask_output=False,
                )
                m = np.asarray(masks).squeeze().astype(bool)
                if m.ndim != 2 or not m.any():
                    continue
                track = TrunkTrack(track_id=next_track_id)
                next_track_id += 1
                refined_det = TrunkDetection(
                    frame_idx=frame_idx,
                    bbox_xyxy=det.bbox_xyxy,
                    score=det.score * float(np.asarray(scores).squeeze()),
                    mask=m,
                    track_id=track.track_id,
                )
                track.detections.append(refined_det)
                tracks.append(track)

    log.info("%s: %d single-frame tracks built", desc_map[backend], len(tracks))
    if backend == "sam3":
        total = n_tree_mask_ok + n_tree_mask_fail
        log.info(
            "===== CANOPY MASK PER FRAME: %d built ok, %d empty, "
            "%d errored =====",
            n_canopy_built, n_canopy_empty, n_canopy_error,
        )
        log.info(
            "===== TREE MASK STATUS: %d/%d detections got a canopy "
            "tree_mask (no-canopy=%d no-CC-in-bbox=%d voronoi-empty=%d "
            "exception=%d) =====",
            n_tree_mask_ok, total,
            n_fail_no_canopy, n_fail_no_cc,
            n_fail_voronoi, n_fail_exception,
        )
    return tracks


def _propagate_video_mode(
    loader: FrameLoader,
    detections_by_frame: Dict[int, List[TrunkDetection]],
    cfg: SegmenterConfig,
    device: str = "cuda",
    work_dir: Optional[Path] = None,
) -> List[TrunkTrack]:
    """SAM2 bidirectional video propagation. Recommended only for >=15 fps.

    Stages frames as JPEGs to a temp directory (SAM2's video API
    requirement), seeds one trunk per likely-distinct detection,
    then forward + backward propagates to build per-trunk multi-
    frame mask tracks. At 5 fps this provides little benefit over
    image mode -- inter-frame displacement is too large for SAM2's
    temporal prior to substantially help. Kept available for
    higher-fps collection runs.
    """
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from tqdm.auto import tqdm

    log.info("Loading SAM2 (video mode): %s", cfg.sam2_checkpoint)
    predictor = build_sam2_video_predictor(
        cfg.sam2_model_cfg, cfg.sam2_checkpoint, device=device,
    )

    cleanup_work = work_dir is None
    work_dir = work_dir or Path(tempfile.mkdtemp(prefix="sam2_orchard_"))
    log.info("Staging frames to %s", work_dir)
    chrono_indices = _stage_jpegs_for_sam2(loader, work_dir)
    sam_to_frame = {i: f for i, f in enumerate(chrono_indices)}
    frame_to_sam = {f: i for i, f in sam_to_frame.items()}

    seeds = _select_seed_detections(detections_by_frame, cfg)

    tracks: List[TrunkTrack] = []
    next_track_id = 1

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        for seed in tqdm(seeds, desc="SAM2 propagate"):
            state = predictor.init_state(video_path=str(work_dir))
            sam_seed_idx = frame_to_sam[seed.frame_idx]
            x1, y1, x2, y2 = seed.bbox_xyxy
            box = np.array([[x1, y1, x2, y2]], dtype=np.float32)

            _, _, _ = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=sam_seed_idx,
                obj_id=1,
                box=box,
            )

            mask_per_frame: Dict[int, np.ndarray] = {}

            for sam_idx, _, mask_logits in predictor.propagate_in_video(state):
                m = (mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(bool)
                if m.any():
                    mask_per_frame[sam_to_frame[sam_idx]] = m

            if cfg.bidirectional:
                for sam_idx, _, mask_logits in predictor.propagate_in_video(
                    state, reverse=True,
                ):
                    m = (mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(bool)
                    fid = sam_to_frame[sam_idx]
                    if m.any() and fid not in mask_per_frame:
                        mask_per_frame[fid] = m

            if len(mask_per_frame) < cfg.min_track_length:
                continue

            track = TrunkTrack(track_id=next_track_id)
            next_track_id += 1
            for fid, mask in sorted(mask_per_frame.items()):
                ys, xs = np.where(mask)
                if xs.size == 0:
                    continue
                bbox = (
                    float(xs.min()), float(ys.min()),
                    float(xs.max()), float(ys.max()),
                )
                det = TrunkDetection(
                    frame_idx=fid,
                    bbox_xyxy=bbox,
                    score=float(seed.score),
                    mask=mask,
                    track_id=track.track_id,
                )
                track.detections.append(det)
            tracks.append(track)

    if cleanup_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    log.info("SAM2 video mode: %d tracks survived (>= %d frames each)",
             len(tracks), cfg.min_track_length)
    return tracks


# ============================================================================
# Stage 3: World projection
# ============================================================================
def _depth_in_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    estimator: str,
) -> float:
    """Robust depth aggregation inside a trunk mask.

    Returns NaN if no valid depth pixels. Default 'median' rejects
    the few aberrant near-zero or maxed-out depth pixels every D455
    frame contains (especially near foliage boundaries and in low-
    texture regions).
    """
    valid = mask & np.isfinite(depth_m) & (depth_m > 0.1) & (depth_m < 20.0)
    if not valid.any():
        return float("nan")
    vals = depth_m[valid]
    if estimator == "median":
        return float(np.median(vals))
    if estimator == "mean":
        return float(np.mean(vals))
    if estimator == "p25":
        return float(np.percentile(vals, 25))
    raise ValueError(f"unknown depth_estimator: {estimator}")


def project_tracks_to_world(
    tracks: List[TrunkTrack],
    loader: FrameLoader,
    cfg: SegmenterConfig,
) -> List[TrunkTrack]:
    """Fill in (depth_m, world_lat, world_lon) on every detection.

    Uses :func:`tree_tracker.camera_to_world` -- same projection as
    your existing pipeline, so output coordinates are directly
    comparable with anything already in your TreeTracker registry.
    """
    from tqdm.auto import tqdm

    n_skipped_no_gps = 0
    n_skipped_no_depth = 0
    n_skipped_out_of_range = 0
    n_projected = 0
    n_used_dilated = 0
    hfov = math.radians(cfg.camera_hfov_deg)

    def _trunk_depth_m(depth: np.ndarray, mask: np.ndarray) -> Tuple[float, bool]:
        """Robust trunk-depth estimate with a dilation fallback.

        RealSense returns no depth for thin verticals against bright
        backgrounds, so a sapling's mask often has 0 valid pixels.
        Fall back to the mask dilated by ~10 px (captures the foliage
        right next to the trunk). Use p10 of those depths so we lock
        onto the closest = trunk-vicinity pixels, not background
        seen through the canopy.
        """
        d = _depth_in_mask(depth, mask, cfg.depth_estimator)
        if math.isfinite(d):
            return d, False
        try:
            import cv2 as _cv2
            kernel = np.ones((21, 21), dtype=np.uint8)
            dil = _cv2.dilate(
                mask.astype(np.uint8), kernel, iterations=1,
            ).astype(bool)
        except Exception:
            return d, False
        # Compute p10 manually — _depth_in_mask only knows median/mean/p25.
        valid = dil & np.isfinite(depth) & (depth > 0.1) & (depth < 20.0)
        if not valid.any():
            return float("nan"), False
        d2 = float(np.percentile(depth[valid], 10))
        return d2, math.isfinite(d2)

    n_used_nominal = 0
    canopy_min_m = max(0.1, cfg.trunk_min_depth_m)
    # Plausible canopy distance — beyond ~5 m is background, not the
    # tree being tracked. Tighter than the [trunk_min, trunk_max]
    # range used for the gate, because we need to *find* canopy
    # depth here, not just validate it.
    canopy_max_m = min(5.0, cfg.trunk_max_depth_m)

    for track in tqdm(tracks, desc="World projection"):
        for det in track.detections:
            depth = loader.load_depth_m(det.frame_idx)
            # Strategy: try several mask-depth aggregations, accept
            # only the first one that lands in [canopy_min, canopy_max].
            # That keeps depths from being dragged to background-
            # through-canopy-gaps, which is what was killing this
            # before. Fall back to nominal distance if nothing fits.
            d_m: float = float("nan")
            used_dilated = False

            def _candidate_in_range(mask: np.ndarray, pct: float) -> float:
                if mask is None or not mask.any():
                    return float("nan")
                v = (mask
                     & np.isfinite(depth)
                     & (depth >= canopy_min_m)
                     & (depth <= canopy_max_m))
                if not v.any():
                    return float("nan")
                return float(np.percentile(depth[v], pct))

            # 1. Tree mask, closest pixels (p10) — best signal when
            #    SAM 3 picked up the tree.
            if det.tree_mask is not None:
                d_m = _candidate_in_range(det.tree_mask, 10)
            # 2. Trunk mask median — fallback when tree mask was
            #    rejected or out of range.
            if not math.isfinite(d_m):
                d_m = _candidate_in_range(det.mask, 50)
            # 3. Dilated trunk mask p10 — picks up foliage near the
            #    trunk when the tight masks have no in-range depth.
            if not math.isfinite(d_m) and det.mask is not None and det.mask.any():
                try:
                    import cv2 as _cv2
                    kernel = np.ones((21, 21), dtype=np.uint8)
                    dil = _cv2.dilate(
                        det.mask.astype(np.uint8), kernel, iterations=1,
                    ).astype(bool)
                    d_m = _candidate_in_range(dil, 10)
                    if math.isfinite(d_m):
                        used_dilated = True
                except Exception:
                    pass
            used_nominal = False
            if used_dilated:
                n_used_dilated += 1
            if not math.isfinite(d_m):
                # Final fallback: nominal aisle distance. Sprayer-row
                # geometry is roughly fixed, so this gives a usable
                # perpendicular distance even when RealSense returns
                # nothing for the trunk. The pixel-x angle still
                # provides the along-row offset.
                if cfg.nominal_tree_distance_m > 0:
                    d_m = cfg.nominal_tree_distance_m
                    used_nominal = True
                    n_used_nominal += 1
                else:
                    n_skipped_no_depth += 1
                    continue
            # Plausible-range gate using the actual SAM-refined trunk
            # mask (much tighter than the GDINO bbox, so background-
            # through-branches pixels don't dominate). Skip the gate
            # for nominal-distance fallbacks (we know the value is in
            # range by construction).
            if not used_nominal and (
                d_m < cfg.trunk_min_depth_m
                or d_m > cfg.trunk_max_depth_m
            ):
                n_skipped_out_of_range += 1
                continue
            meta = loader.load_meta(det.frame_idx)
            if "gps_lat" not in meta or "gps_lon" not in meta:
                n_skipped_no_gps += 1
                continue
            n_projected += 1
            # Pixel-x of trunk centroid → angle off the optical axis →
            # true forward + lateral split. Without this every trunk in
            # a frame projects to the same lat/lon and a single tree
            # smears several metres along the row as the camera passes.
            h, w = depth.shape[:2]
            x1, _, x2, _ = det.bbox_xyxy
            cx = (float(x1) + float(x2)) * 0.5
            angle = ((cx - w * 0.5) / w) * hfov
            forward_m = d_m * math.cos(angle)
            lateral_px = d_m * math.sin(angle)
            wlat, wlon = camera_to_world(
                gps_lat=float(meta["gps_lat"]),
                gps_lon=float(meta["gps_lon"]),
                heading_deg=float(meta["heading_deg"]),
                depth_m=forward_m,
                lateral_offset_m=cfg.camera_lateral_offset_m + lateral_px,
            )
            det.depth_m = d_m
            det.world_lat = wlat
            det.world_lon = wlon
    log.info(
        "World projection: %d projected "
        "(%d dilated-mask, %d nominal-distance fallback %.1f m), "
        "%d skipped no-GPS, %d skipped no-depth, %d skipped out-of-range "
        "(trunk_depth_m=[%.1f, %.1f])",
        n_projected, n_used_dilated, n_used_nominal,
        cfg.nominal_tree_distance_m,
        n_skipped_no_gps, n_skipped_no_depth, n_skipped_out_of_range,
        cfg.trunk_min_depth_m, cfg.trunk_max_depth_m,
    )
    return tracks


# ============================================================================
# Stage 4: DBSCAN clustering -> persistent tree IDs
# ============================================================================
def _world_to_local_xy(
    lats: np.ndarray, lons: np.ndarray,
) -> np.ndarray:
    """Project (lat, lon) to local metric (x, y) for clustering.

    DBSCAN's eps is in the same units as the input. Doing it in
    metres gives an interpretable knob (eps = 1.2 m). Flat-earth
    is fine -- orchard rows are <1 km long.
    """
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    cos_lat = math.cos(math.radians(lat0)) or 1e-6
    x = (lons - lon0) * 111_320.0 * cos_lat
    y = (lats - lat0) * 111_320.0
    return np.stack([x, y], axis=1)


def _estimate_dbscan_eps(
    xy: np.ndarray,
    min_samples: int,
    max_eps_m: float = 2.0,
    min_eps_m: float = 0.2,
) -> float:
    """Auto-estimate DBSCAN eps from the k-distance graph elbow.

    Sorts every detection by its distance to its ``min_samples``-th
    nearest neighbour. The resulting curve has a sharp elbow where
    within-tree jitter ends and between-tree gaps begin.

    eps sits just above the within-tree side of the elbow (the
    *lower* end of the jump, plus a small safety margin) — putting
    it at the midpoint risks bridging two adjacent trees when the
    jump is wide. Result is then clamped to
    ``[min_eps_m, max_eps_m]`` to guard against pathological
    distance distributions.

    Apple orchards typically plant trees 1–3 m apart, so capping
    eps at 1.0 m by default keeps neighbours from being merged
    even when the elbow heuristic mis-fires.
    """
    from sklearn.neighbors import NearestNeighbors

    if len(xy) < 2:
        log.warning(
            "Auto-eps: only %d valid detections — using fallback %.2f m",
            len(xy), max_eps_m,
        )
        return max_eps_m
    k = max(1, min(min_samples, len(xy) - 1))
    nbrs = NearestNeighbors(n_neighbors=k).fit(xy)
    distances, _ = nbrs.kneighbors(xy)
    k_dist = np.sort(distances[:, -1])

    diffs = np.diff(k_dist)
    elbow = int(np.argmax(diffs))
    # Sit just above the within-tree side of the elbow. The 1.15
    # multiplier gives ~15% headroom to absorb noise without
    # pushing eps into the inter-tree gap.
    eps_raw = float(k_dist[elbow]) * 1.15
    eps = float(np.clip(eps_raw, min_eps_m, max_eps_m))

    log.info(
        "Auto-estimated DBSCAN eps=%.2f m (raw %.2f, clamped to "
        "[%.2f, %.2f]) from k-distance elbow (k=%d, %d detections, "
        "jump %.2f→%.2f m)",
        eps, eps_raw, min_eps_m, max_eps_m, k, len(xy),
        k_dist[elbow], k_dist[elbow + 1],
    )
    return eps


def cluster_to_trees(
    tracks: List[TrunkTrack],
    cfg: SegmenterConfig,
) -> List[TreeCluster]:
    """DBSCAN over per-detection trunk world positions.

    We cluster *detections*, not track-medians, because depth noise
    means a single track can drift its world position by 0.3-0.6 m
    across its frame span. Letting every detection vote, then taking
    the cluster centroid as the canonical tree position, averages
    that noise out. ``min_samples`` rejects single-detection clusters
    that are usually false positives in distant background trees.
    """
    from sklearn.cluster import DBSCAN

    # Flatten detections into a single array, keeping a mapping
    # back to (track_idx, det_idx) so we can write tree_id back.
    all_dets: List[Tuple[int, int, TrunkDetection]] = []
    lats: List[float] = []
    lons: List[float] = []
    for ti, track in enumerate(tracks):
        for di, det in enumerate(track.detections):
            if det.world_lat == 0.0 and det.world_lon == 0.0:
                continue                                # projection failed
            all_dets.append((ti, di, det))
            lats.append(det.world_lat)
            lons.append(det.world_lon)

    if not all_dets:
        log.warning("No detections with valid world coords; clustering skipped")
        return []

    xy = _world_to_local_xy(np.array(lats), np.array(lons))
    if cfg.dbscan_eps_m is None:
        eps_m = _estimate_dbscan_eps(xy, cfg.dbscan_min_samples)
    else:
        eps_m = cfg.dbscan_eps_m
        log.info("DBSCAN eps=%.2f m (fixed), min_samples=%d",
                 eps_m, cfg.dbscan_min_samples)
    db = DBSCAN(eps=eps_m, min_samples=cfg.dbscan_min_samples).fit(xy)
    labels = db.labels_

    # Build TreeCluster per label (label==-1 is DBSCAN noise; drop it).
    by_label: Dict[int, TreeCluster] = {}
    next_tree_id = 1

    # Stable tree_id assignment: order clusters by along-row position
    # so tree 1 is at the start of the run, tree 2 next, etc. We use
    # the centroid's projection onto the row's principal axis (PCA
    # on cluster centroids).
    cluster_centroids: Dict[int, Tuple[float, float]] = {}
    for lbl in set(labels):
        if lbl == -1:
            continue
        mask = labels == lbl
        cx = float(np.mean(xy[mask, 0]))
        cy = float(np.mean(xy[mask, 1]))
        cluster_centroids[int(lbl)] = (cx, cy)

    if not cluster_centroids:
        log.warning("DBSCAN found no clusters")
        return []

    centroid_arr = np.array(list(cluster_centroids.values()))
    if centroid_arr.shape[0] >= 2:
        # PCA: principal axis = row direction.
        c0 = centroid_arr - centroid_arr.mean(axis=0, keepdims=True)
        u, _, _ = np.linalg.svd(c0, full_matrices=False)
        # Project onto first principal component direction.
        _, _, vt = np.linalg.svd(c0, full_matrices=False)
        axis = vt[0]
        proj = c0 @ axis
        order = np.argsort(proj)
        ordered_labels = [list(cluster_centroids.keys())[i] for i in order]
    else:
        ordered_labels = list(cluster_centroids.keys())

    label_to_tree_id = {lbl: i + 1 for i, lbl in enumerate(ordered_labels)}

    for (ti, di, det), lbl in zip(all_dets, labels):
        if lbl == -1:
            continue
        tid = label_to_tree_id[int(lbl)]
        if tid not in by_label:
            cx, cy = cluster_centroids[int(lbl)]
            # Convert back to (lat, lon) using the same lat0/lon0.
            lat0 = float(np.mean(lats))
            lon0 = float(np.mean(lons))
            cos_lat = math.cos(math.radians(lat0)) or 1e-6
            wlat = lat0 + cy / 111_320.0
            wlon = lon0 + cx / (111_320.0 * cos_lat)
            by_label[tid] = TreeCluster(
                tree_id=tid, world_lat=wlat, world_lon=wlon,
            )
        cluster = by_label[tid]
        if not cluster.tracks or cluster.tracks[-1].track_id != tracks[ti].track_id:
            cluster.tracks.append(tracks[ti])
        # Frame attribution: count mask pixels for dominance scoring.
        n_pixels = int(det.mask.sum()) if det.mask is not None else 0
        cluster.frame_pixels[det.frame_idx] = (
            cluster.frame_pixels.get(det.frame_idx, 0) + n_pixels
        )

    clusters = sorted(by_label.values(), key=lambda c: c.tree_id)
    n_noise = int(np.sum(labels == -1))
    log.info("DBSCAN: %d trees, %d noise detections discarded",
             len(clusters), n_noise)
    return clusters


# ============================================================================
# Stage 5: Frame attribution
# ============================================================================
def attribute_frames(
    clusters: List[TreeCluster],
    cfg: SegmenterConfig,
) -> Dict[int, List[Tuple[int, float]]]:
    """For each frame, return ranked list of (tree_id, dominance_weight).

    Weight is mask-pixel-count normalised by the largest mask in that
    frame. Trees below ``attribution_dominance_threshold`` are dropped
    (they're in the FOV but only marginally visible).

    Use the first entry in each list as the primary tree_id for that
    frame; the rest are secondary visible trees.
    """
    # Invert: frame_idx -> list of (tree_id, pixels)
    by_frame: Dict[int, List[Tuple[int, int]]] = {}
    for cluster in clusters:
        for fid, npx in cluster.frame_pixels.items():
            by_frame.setdefault(fid, []).append((cluster.tree_id, npx))

    out: Dict[int, List[Tuple[int, float]]] = {}
    for fid, items in by_frame.items():
        max_px = max(p for _, p in items)
        if max_px <= 0:
            continue
        ranked = sorted(items, key=lambda kv: -kv[1])
        kept = [
            (tid, p / max_px)
            for tid, p in ranked
            if (p / max_px) >= cfg.attribution_dominance_threshold
        ]
        if kept:
            out[fid] = kept
    return out


# ============================================================================
# Stage 5b: ROI-based flower attribution
# ============================================================================
def _frame_tree_mask_index(
    clusters: List[TreeCluster],
) -> Dict[int, Dict[int, np.ndarray]]:
    """Build {frame_idx: {tree_id: trunk_mask}} from clustered tracks.

    Only includes (frame, tree) pairs that the cluster step actually
    accepted — i.e. frame_idx appears in cluster.frame_pixels.
    """
    idx: Dict[int, Dict[int, np.ndarray]] = {}
    for cluster in clusters:
        for track in cluster.tracks:
            for det in track.detections:
                if det.mask is not None and det.frame_idx in cluster.frame_pixels:
                    idx.setdefault(det.frame_idx, {})[cluster.tree_id] = det.mask
    return idx


def attribute_flowers_via_roi(
    clusters: List[TreeCluster],
    attribution: Dict[int, List[Tuple[int, float]]],
    flower_counts_by_frame: Dict[int, float],
    roi_masks_by_frame: Dict[int, Optional[np.ndarray]],
    min_trunk_roi_overlap_px: int = 50,
) -> Dict[int, float]:
    """Assign per-frame flower counts to trees using PRGB ROI masks.

    Decision tree per frame
    -----------------------
    1. No ROI for this frame → skip. Flowers are already restricted to
       the ROI by ``analyze_days.py --prgb``, so if there is no ROI the
       frame contributed nothing and there is nothing to assign.
    2. Exactly one tree's trunk overlaps the ROI → all flowers go to
       that tree.
    3. Two or more trunks overlap the same ROI → split proportionally
       by dominance weight (trunk pixel count) among the in-ROI trees
       only.
    4. ROI present but no trunk detected inside it → fall back to the
       primary attributed tree (highest-weight tree for that frame).

    Parameters
    ----------
    flower_counts_by_frame
        ``{frame_idx: n_flowers}`` — use ``n_detections`` or
        ``est_flowers`` from ``analyze_days.py``'s ``results.csv``.
        Build it with :func:`load_flower_counts_from_csv`.
    roi_masks_by_frame
        ``{frame_idx: bool_array_or_None}`` — the (H, W) PRGB ROI mask
        for each frame. None means no ROI was extracted for that frame.
        Build it from ``loader.load_roi_mask()`` or
        ``analyze_days.extract_roi_mask()``.
    min_trunk_roi_overlap_px
        Minimum pixels of trunk-mask ∩ ROI-mask to consider a trunk
        "inside" the ROI. Guards against a trunk that barely grazes the
        ROI edge stealing attribution (default 50 px).

    Returns
    -------
    Dict mapping tree_id → total flower count. Also mutates
    ``cluster.flower_count`` on every cluster in *clusters*.
    """
    frame_tree_masks = _frame_tree_mask_index(clusters)
    totals: Dict[int, float] = {c.tree_id: 0.0 for c in clusters}
    n_assigned = n_skipped_no_roi = n_fallback = n_split = 0

    for frame_idx, n_flowers in flower_counts_by_frame.items():
        if n_flowers <= 0:
            continue

        roi = roi_masks_by_frame.get(frame_idx)
        if roi is None or not roi.any():
            n_skipped_no_roi += 1
            continue

        ranked = attribution.get(frame_idx, [])
        tree_masks_frame = frame_tree_masks.get(frame_idx, {})

        # Which trees have trunks that land inside this ROI?
        in_roi: List[Tuple[int, float]] = []
        for tree_id, weight in ranked:
            mask = tree_masks_frame.get(tree_id)
            if mask is None:
                continue
            overlap_px = int(np.logical_and(mask, roi).sum())
            if overlap_px >= min_trunk_roi_overlap_px:
                in_roi.append((tree_id, weight))

        if not in_roi:
            # Segmenter missed the trunk inside the ROI; use primary tree.
            if ranked:
                totals[ranked[0][0]] += n_flowers
                n_fallback += 1
                log.debug("frame %d: no trunk in ROI, fallback to tree %d",
                          frame_idx, ranked[0][0])
        elif len(in_roi) == 1:
            totals[in_roi[0][0]] += n_flowers
            n_assigned += 1
        else:
            # Multiple trunks in the same ROI → weighted split.
            total_w = sum(w for _, w in in_roi)
            if total_w <= 0:
                totals[in_roi[0][0]] += n_flowers
            else:
                for tree_id, w in in_roi:
                    totals[tree_id] += n_flowers * (w / total_w)
            n_split += 1
            log.debug("frame %d: %d trunks in ROI, split %.1f flowers",
                      frame_idx, len(in_roi), n_flowers)

    for cluster in clusters:
        cluster.flower_count = totals.get(cluster.tree_id, 0.0)

    log.info(
        "Flower attribution: %d frames → 1 tree, %d split, "
        "%d fallback, %d skipped (no ROI)",
        n_assigned, n_split, n_fallback, n_skipped_no_roi,
    )
    return totals


def attribute_flowers_via_tree_mask(
    clusters: List[TreeCluster],
    loader: "FrameLoader",
    dataset_root: Path,
    flower_masks_dir: Path,
    flower_slug: str,
    flower_counts_by_frame: Dict[int, float],
    cfg: SegmenterConfig,
) -> Tuple[Dict[int, float], set]:
    """Attribute each flower mask to the tree whose whole-tree mask contains it.

    Strongest signal we can build: SAM 3 has segmented the entire
    tree (canopy + trunk) per frame, so a flower whose centroid
    lies inside a particular tree's mask is unambiguously *that*
    tree's flower. No 3D projection, no nearest-neighbour
    heuristic, no depth needed. Two touching trees naturally
    separate because each has its own SAM 3 mask seeded from a
    different trunk.

    Frames where no cluster has a tree_mask, or no flower's
    centroid lands inside any tree mask, fall through to the 3D
    nearest-tree / ROI-overlap paths via the caller.

    Returns
    -------
    (totals, frames_handled)
        ``totals`` is ``{tree_id: flower_count}`` aggregated across
        successfully tree-mask-attributed frames.
        ``frames_handled`` is the set of frame indices fully
        covered by this path.
    """
    if not clusters:
        return {}, set()

    # Build {frame_idx: [(tree_id, tree_mask), ...]}: for each
    # frame, which clusters had a detection here, and what tree
    # mask did SAM 3 produce.
    frame_index: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for c in clusters:
        for track in c.tracks:
            for det in track.detections:
                if det.tree_mask is None or not det.tree_mask.any():
                    continue
                frame_index.setdefault(det.frame_idx, []).append(
                    (c.tree_id, det.tree_mask),
                )

    totals: Dict[int, float] = {c.tree_id: 0.0 for c in clusters}
    frames_handled: set = set()
    n_no_index = n_no_npz = n_no_hit = n_handled = 0

    for frame_idx, expected_total in flower_counts_by_frame.items():
        if expected_total <= 0:
            continue
        tree_masks = frame_index.get(frame_idx)
        if not tree_masks:
            n_no_index += 1
            continue
        try:
            img_path = loader._imgs[frame_idx]  # type: ignore[attr-defined]
            rel = img_path.relative_to(dataset_root).with_suffix(".npz")
        except Exception:
            n_no_npz += 1
            continue
        npz_path = flower_masks_dir / flower_slug / rel
        if not npz_path.is_file():
            # Try sibling slugs (analyze_days.py multi-prompt may
            # have stored under "apple_blossom" etc.).
            found = None
            if flower_masks_dir.is_dir():
                for sub in flower_masks_dir.iterdir():
                    cand = sub / rel
                    if cand.is_file():
                        found = cand
                        break
            if found is None:
                n_no_npz += 1
                continue
            npz_path = found
        try:
            data = np.load(npz_path)
            f_masks = data["masks"]
        except Exception:
            n_no_npz += 1
            continue
        if f_masks.ndim != 3 or len(f_masks) == 0:
            n_no_npz += 1
            continue

        per_cluster: Dict[int, int] = {tid: 0 for tid, _ in tree_masks}
        for fm in f_masks:
            fm = np.asarray(fm, dtype=bool)
            if not fm.any():
                continue
            ys, xs = np.where(fm)
            cy = int(round(float(ys.mean())))
            cx = int(round(float(xs.mean())))
            best_tid = None
            best_overlap = 0
            for tid, tmask in tree_masks:
                if 0 <= cy < tmask.shape[0] and 0 <= cx < tmask.shape[1]:
                    if tmask[cy, cx]:
                        # Use mask-pixel intersection size as a
                        # tiebreaker if multiple tree masks contain
                        # the centroid (rare overlap zone).
                        ov = int(np.logical_and(fm, tmask).sum())
                        if ov > best_overlap:
                            best_overlap = ov
                            best_tid = tid
            if best_tid is not None:
                per_cluster[best_tid] = per_cluster.get(best_tid, 0) + 1

        n_attributed = sum(per_cluster.values())
        if n_attributed == 0:
            n_no_hit += 1
            continue
        scale = float(expected_total) / float(n_attributed)
        for tid, n in per_cluster.items():
            if n:
                totals[tid] = totals.get(tid, 0.0) + n * scale
        frames_handled.add(frame_idx)
        n_handled += 1

    log.info(
        "Flower tree-mask attribution: %d frames handled, "
        "%d no-tree-mask, %d no-mask-file, %d no-centroid-hit",
        n_handled, n_no_index, n_no_npz, n_no_hit,
    )
    return totals, frames_handled


def attribute_flowers_3d_nearest_tree(
    clusters: List[TreeCluster],
    loader: "FrameLoader",
    dataset_root: Path,
    flower_masks_dir: Path,
    flower_slug: str,
    flower_counts_by_frame: Dict[int, float],
    cfg: SegmenterConfig,
    max_assign_distance_m: float = 3.0,
) -> Tuple[Dict[int, float], set]:
    """Project each per-flower mask to world coords; assign to nearest tree.

    For each frame:
      1. Load ``<flower_masks_dir>/<slug>/<rel_path>.npz`` produced by
         ``analyze_days.py --save-masks``.
      2. For every flower mask, take the median depth inside that mask
         and project the frame's GPS forward by that depth (same
         projection ``project_tracks_to_world`` uses for trunks).
      3. Find the cluster centroid closest by haversine distance,
         capped at ``max_assign_distance_m``. If no cluster is within
         range, the flower is left unattributed for that frame.
      4. Scale the per-cluster integer hits by
         ``flower_counts_by_frame[frame] / n_attributed`` so each
         frame's CSV-reported total (e.g. ``est_flowers``) is
         preserved while the *split* between trees is the 3D one.

    Frames with no ``.npz``, no GPS, or zero attributable masks are
    returned in *frames_unhandled* so the caller can run the
    ROI-overlap path on them as a fallback.

    Returns
    -------
    (totals, frames_handled)
        ``totals`` is ``{tree_id: flower_count}`` aggregated across
        successfully 3D-attributed frames. ``frames_handled`` is the
        set of frame indices that the 3D path successfully attributed
        — caller should skip these in the ROI fallback.
    """
    if not clusters:
        return {}, set()

    cluster_pos = [
        (c.tree_id, c.world_lat, c.world_lon) for c in clusters
    ]
    totals: Dict[int, float] = {tid: 0.0 for tid, _, _ in cluster_pos}
    frames_handled: set = set()

    n_no_npz = n_no_gps = n_no_attribution = 0
    n_handled = 0
    logged_first_path = False

    # If <flower_masks_dir>/<flower_slug> doesn't exist but the masks
    # dir has a sibling slug (e.g. "apple_blossom" instead of
    # "flower"), pick that. analyze_days.py merges the
    # --flower-multi-prompts list under the FIRST prompt's slug.
    candidate_slugs = [flower_slug]
    if flower_masks_dir.is_dir():
        for sub in sorted(p.name for p in flower_masks_dir.iterdir()
                          if p.is_dir()):
            if sub != flower_slug and sub not in candidate_slugs:
                candidate_slugs.append(sub)
    else:
        log.warning(
            "Flower masks dir does not exist: %s "
            "(3D attribution will be skipped)",
            flower_masks_dir,
        )

    for frame_idx, expected_total in flower_counts_by_frame.items():
        if expected_total <= 0:
            continue

        # Locate the per-frame .npz of flower masks. analyze_days.py
        # writes them at <out>/masks/<slug>/<rel-to-args.root>.npz.
        try:
            img_path = loader._imgs[frame_idx]  # type: ignore[attr-defined]
            rel = img_path.relative_to(dataset_root).with_suffix(".npz")
        except Exception:
            n_no_npz += 1
            continue
        npz_path = None
        for slug in candidate_slugs:
            p = flower_masks_dir / slug / rel
            if p.is_file():
                npz_path = p
                break
        if not logged_first_path:
            log.info(
                "Flower 3D-attribution: looking under %s for %s "
                "(slugs tried: %s) -> %s",
                flower_masks_dir, rel, candidate_slugs,
                "FOUND" if npz_path else "MISSING",
            )
            logged_first_path = True
        if npz_path is None:
            n_no_npz += 1
            continue

        try:
            data = np.load(npz_path)
            masks = data["masks"]
        except Exception:
            n_no_npz += 1
            continue
        if masks.ndim != 3 or len(masks) == 0:
            n_no_npz += 1
            continue

        meta = loader.load_meta(frame_idx)
        if "gps_lat" not in meta or "gps_lon" not in meta:
            n_no_gps += 1
            continue
        depth = loader.load_depth_m(frame_idx)
        if depth is None or not np.isfinite(depth).any():
            n_no_gps += 1
            continue

        h_img, w_img = depth.shape[:2]
        hfov = math.radians(cfg.camera_hfov_deg)
        per_cluster: Dict[int, int] = {tid: 0 for tid, _, _ in cluster_pos}
        for m in masks:
            m = np.asarray(m, dtype=bool)
            if not m.any():
                continue
            d_pixels = depth[m]
            d_pixels = d_pixels[np.isfinite(d_pixels) & (d_pixels > 0)]
            if d_pixels.size == 0:
                continue
            d_m = float(np.median(d_pixels))
            ys, xs = np.where(m)
            cx = float(xs.mean())
            angle = ((cx - w_img * 0.5) / w_img) * hfov
            forward_m = d_m * math.cos(angle)
            lateral_px = d_m * math.sin(angle)
            wlat, wlon = camera_to_world(
                gps_lat=float(meta["gps_lat"]),
                gps_lon=float(meta["gps_lon"]),
                heading_deg=float(meta["heading_deg"]),
                depth_m=forward_m,
                lateral_offset_m=cfg.camera_lateral_offset_m + lateral_px,
            )
            best_tid = None
            best_d = max_assign_distance_m
            for tid, clat, clon in cluster_pos:
                d_to = haversine_m(wlat, wlon, clat, clon)
                if d_to < best_d:
                    best_d = d_to
                    best_tid = tid
            if best_tid is not None:
                per_cluster[best_tid] += 1

        n_attributed = sum(per_cluster.values())
        if n_attributed == 0:
            n_no_attribution += 1
            continue

        # Scale to match the CSV's per-frame total (e.g. est_flowers).
        scale = float(expected_total) / float(n_attributed)
        for tid, n in per_cluster.items():
            if n:
                totals[tid] += n * scale
        frames_handled.add(frame_idx)
        n_handled += 1

    log.info(
        "Flower 3D-attribution: %d frames handled, %d no-mask-file, "
        "%d no-gps/depth, %d zero-in-range",
        n_handled, n_no_npz, n_no_gps, n_no_attribution,
    )
    return totals, frames_handled


def attribute_flowers(
    clusters: List[TreeCluster],
    attribution: Dict[int, List[Tuple[int, float]]],
    flower_counts_by_frame: Dict[int, float],
    roi_masks_by_frame: Dict[int, Optional[np.ndarray]],
    loader: "FrameLoader",
    cfg: SegmenterConfig,
    dataset_root: Optional[Path] = None,
    flower_masks_dir: Optional[Path] = None,
    flower_slug: Optional[str] = None,
    max_assign_distance_m: float = 3.0,
    min_trunk_roi_overlap_px: int = 50,
) -> Dict[int, float]:
    """Assign flowers to trees in three layers, strongest first:

    1. tree-mask containment: a flower whose centroid lies inside
       a SAM-3-segmented whole-tree mask is unambiguously that tree's.
    2. 3D nearest-tree: project flower depth -> world coords -> nearest
       cluster centroid (within ``max_assign_distance_m``).
    3. PRGB-ROI overlap with dominance weights (legacy fallback).

    Each frame is handled by the first layer that finds a hit.
    Mutates ``cluster.flower_count`` and returns ``{tree_id: total}``.
    """
    totals_tm: Dict[int, float] = {}
    frames_handled_tm: set = set()
    if (flower_masks_dir is not None and flower_slug is not None
            and dataset_root is not None):
        totals_tm, frames_handled_tm = attribute_flowers_via_tree_mask(
            clusters, loader, dataset_root, flower_masks_dir, flower_slug,
            flower_counts_by_frame, cfg,
        )

    remaining = {
        f: n for f, n in flower_counts_by_frame.items()
        if f not in frames_handled_tm
    }

    totals_3d: Dict[int, float] = {}
    frames_handled_3d: set = set()
    if (flower_masks_dir is not None and flower_slug is not None
            and dataset_root is not None):
        totals_3d, frames_handled_3d = attribute_flowers_3d_nearest_tree(
            clusters, loader, dataset_root, flower_masks_dir, flower_slug,
            remaining, cfg,
            max_assign_distance_m=max_assign_distance_m,
        )

    fallback_counts = {
        f: n for f, n in remaining.items()
        if f not in frames_handled_3d
    }
    totals_roi = attribute_flowers_via_roi(
        clusters, attribution, fallback_counts, roi_masks_by_frame,
        min_trunk_roi_overlap_px=min_trunk_roi_overlap_px,
    )

    combined: Dict[int, float] = {c.tree_id: 0.0 for c in clusters}
    for tid, n in totals_tm.items():
        combined[tid] = combined.get(tid, 0.0) + n
    for tid, n in totals_3d.items():
        combined[tid] = combined.get(tid, 0.0) + n
    for tid, n in totals_roi.items():
        combined[tid] = combined.get(tid, 0.0) + n
    for cluster in clusters:
        cluster.flower_count = combined.get(cluster.tree_id, 0.0)
    return combined


def load_flower_counts_from_csv(
    csv_path: str,
    session: str,
    loader: "FrameLoader",
    prompt: str = "apple blossom",
    count_col: str = "est_flowers",
) -> Dict[int, float]:
    """Parse ``analyze_days.py``'s ``results.csv`` into ``{loader_frame_idx: n}``.

    Keys are the loader's *internal* frame indices (0..N-1 for the
    --frame-range slice the loader was constructed with), matching
    what ``loader.frame_indices()`` returns. The CSV's ``image``
    column holds the absolute path written by analyze_days.py; we
    map it back to the loader index by image filename stem.

    Parameters
    ----------
    session
        The session folder name to filter on (``session`` column in the CSV).
    loader
        The frame loader for this session — needed to resolve image
        stems back to loader-internal frame indices.
    prompt
        The SAM 3 prompt to sum over (default ``"apple blossom"``).
    count_col
        Which count column to use: ``"est_flowers"`` (density estimate)
        or ``"n_detections"`` (raw SAM 3 detections after quality filter).
    """
    import csv as _csv

    stem_to_idx: Dict[str, int] = {
        p.stem: i for i, p in enumerate(getattr(loader, "_imgs", []))
    }

    counts: Dict[int, float] = {}
    n_unmatched = 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            if row.get("session") != session:
                continue
            if row.get("prompt") != prompt:
                continue
            stem = Path(row["image"]).stem
            frame_idx = stem_to_idx.get(stem)
            if frame_idx is None:
                n_unmatched += 1
                continue
            counts[frame_idx] = (
                counts.get(frame_idx, 0.0)
                + float(row.get(count_col) or 0)
            )
    if n_unmatched:
        log.debug(
            "Flower CSV: %d rows for session %s had image stems not "
            "in the loader's frame slice (frame-range mismatch)",
            n_unmatched, session,
        )
    return counts


# ============================================================================
# Stage 6: Sanity check on inter-cluster spacing
# ============================================================================
def sanity_check_spacing(
    clusters: List[TreeCluster],
    cfg: SegmenterConfig,
) -> List[TreeCluster]:
    """Flag clusters whose neighbour spacing is implausible.

    Two failure modes to catch:
      * **Gap too large** (> tree_spacing × (1 + tol)) -> a tree was
        likely missed between this cluster and its predecessor. Flag
        BOTH endpoints so the operator can scrub the contact sheet
        for either side and decide where to insert.
      * **Gap too small** (< tree_spacing × (1 - tol)) -> two clusters
        likely belong to one physical tree. Flag the smaller-mass
        cluster as a merge candidate.

    Mutates ``flagged_reason`` on the affected clusters and returns
    the same list (chainable).
    """
    if len(clusters) < 2:
        return clusters

    lats = np.array([c.world_lat for c in clusters])
    lons = np.array([c.world_lon for c in clusters])
    xy = _world_to_local_xy(lats, lons)

    # Re-order along the row's principal axis (already done by the
    # cluster step, but be defensive).
    if len(clusters) >= 2:
        c0 = xy - xy.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(c0, full_matrices=False)
        axis = vt[0]
        proj = c0 @ axis
        order = np.argsort(proj)
        clusters_sorted = [clusters[i] for i in order]
        proj_sorted = proj[order]
    else:
        clusters_sorted = list(clusters)
        proj_sorted = np.zeros(len(clusters))

    spacing = np.diff(proj_sorted)
    lo = cfg.tree_spacing_m * (1.0 - cfg.spacing_tolerance)
    hi = cfg.tree_spacing_m * (1.0 + cfg.spacing_tolerance)

    for i, gap in enumerate(spacing):
        gap = float(abs(gap))
        if gap > hi:
            msg = f"large_gap_{gap:.2f}m"
            clusters_sorted[i].flagged_reason = (
                f"{clusters_sorted[i].flagged_reason or ''};{msg}".strip(";")
            )
            clusters_sorted[i + 1].flagged_reason = (
                f"{clusters_sorted[i + 1].flagged_reason or ''};{msg}".strip(";")
            )
        elif gap < lo:
            # Flag the lighter-evidence cluster as merge candidate.
            a, b = clusters_sorted[i], clusters_sorted[i + 1]
            target = a if a.n_frames < b.n_frames else b
            msg = f"close_neighbour_{gap:.2f}m"
            target.flagged_reason = (
                f"{target.flagged_reason or ''};{msg}".strip(";")
            )

    n_flagged = sum(1 for c in clusters if c.flagged_reason)
    log.info("Spacing sanity check: %d/%d clusters flagged",
             n_flagged, len(clusters))
    return clusters


# ============================================================================
# Stage 7: Verification artifact
# ============================================================================
def build_contact_sheets(
    clusters: List[TreeCluster],
    loader: FrameLoader,
    output_dir: str,
    cfg: SegmenterConfig,
) -> List[str]:
    """Render one PNG per tree showing every attributed frame thumbnail.

    Each thumbnail has the SAM2 mask outlined in red. Five minutes
    of human flipping through the sheets catches any residual mis-
    attributions and lets the operator manually merge / split via
    cluster IDs. Sheets for flagged clusters get a warning banner.
    """
    import cv2
    from PIL import Image

    out_paths: List[str] = []
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    THUMB = cfg.contact_sheet_thumb_size

    for cluster in clusters:
        # Pick up to N thumbs evenly spaced through the frame range.
        frames = sorted(cluster.frame_pixels.keys())
        if len(frames) > cfg.contact_sheet_max_thumbs:
            step = len(frames) / cfg.contact_sheet_max_thumbs
            frames = [frames[int(i * step)]
                      for i in range(cfg.contact_sheet_max_thumbs)]

        # Build mappings frame_idx -> trunk_mask, frame_idx -> tree_mask
        trunk_by_frame: Dict[int, np.ndarray] = {}
        tree_by_frame: Dict[int, np.ndarray] = {}
        for track in cluster.tracks:
            for det in track.detections:
                if det.frame_idx not in cluster.frame_pixels:
                    continue
                if det.mask is not None:
                    trunk_by_frame[det.frame_idx] = det.mask
                if det.tree_mask is not None and det.tree_mask.any():
                    tree_by_frame[det.frame_idx] = det.tree_mask

        # Build the grid.
        cols = min(6, len(frames))
        rows = (len(frames) + cols - 1) // cols
        grid = Image.new("RGB", (cols * THUMB, rows * THUMB), (0, 0, 0))
        for k, fid in enumerate(frames):
            rgb = loader.load_rgb(fid)
            tree_mask = tree_by_frame.get(fid)
            trunk_mask = trunk_by_frame.get(fid)
            # Whole-tree mask: translucent green fill + green contour.
            if tree_mask is not None:
                tint = rgb.copy()
                tint[tree_mask] = (
                    tint[tree_mask] * 0.5
                    + np.array([0, 200, 0], dtype=np.float32) * 0.5
                ).astype(np.uint8)
                rgb = tint
                contours, _ = cv2.findContours(
                    tree_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
            # Trunk mask on top: red contour.
            if trunk_mask is not None:
                contours, _ = cv2.findContours(
                    trunk_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(rgb, contours, -1, (255, 0, 0), 3)
            thumb = cv2.resize(rgb, (THUMB, THUMB))
            grid.paste(Image.fromarray(thumb),
                       ((k % cols) * THUMB, (k // cols) * THUMB))

        suffix = f"_FLAGGED_{cluster.flagged_reason}" if cluster.flagged_reason else ""
        path = out_root / f"tree_{cluster.tree_id:03d}{suffix}.png"
        grid.save(path)
        out_paths.append(str(path))

    log.info("Wrote %d contact sheets to %s", len(out_paths), out_root)
    return out_paths


# ============================================================================
# Stage 8: Integration with TreeTracker
# ============================================================================
def populate_tree_tracker(
    tracker,
    clusters: List[TreeCluster],
    loader: FrameLoader,
) -> None:
    """Write cluster results into a fresh ``TreeTracker``.

    Mirrors the ``segments`` -> ``TrackedTree`` registration loop at
    the bottom of ``TreeTracker.refine_with_signal``. After this call
    the tracker holds N TrackedTrees, one per cluster, with full
    TreeDetection observations preserved.

    The tracker is reset() first so this is safe to call repeatedly.
    """
    try:
        from .tree_tracker import TreeDetection, TrackedTree
    except ImportError:
        from tree_tracker import TreeDetection, TrackedTree  # type: ignore

    tracker.reset()
    for cluster in clusters:
        # Allocate the right tree_id by burning IDs up to it.
        while tracker._next_id < cluster.tree_id:
            tracker._next_id += 1
        for fid, npx in sorted(cluster.frame_pixels.items()):
            meta = loader.load_meta(fid)
            # Use the cluster centroid as the canonical world coord
            # (rather than per-frame trunk projection, which carries
            # depth noise).
            det = TreeDetection(
                frame_idx=fid,
                world_lat=cluster.world_lat,
                world_lon=cluster.world_lon,
                depth_m=float(meta.get("mean_depth_m", 0.0)),
                estimated_lai=float(meta.get("estimated_lai", 0.0)),
                gps_lat=float(meta["gps_lat"]),
                gps_lon=float(meta["gps_lon"]),
                heading_deg=float(meta["heading_deg"]),
                timestamp=float(meta["timestamp"]),
                meta={
                    "trunk_mask_pixels": npx,
                    "source": "sam2_orchard_segmenter",
                    "flagged": cluster.flagged_reason,
                },
            )
            with tracker._lock:
                if cluster.tree_id not in tracker._trees:
                    tracker._trees[cluster.tree_id] = TrackedTree(
                        tree_id=cluster.tree_id,
                        world_lat=cluster.world_lat,
                        world_lon=cluster.world_lon,
                    )
                tracker._trees[cluster.tree_id].update_position(det)
                tracker._next_id = max(tracker._next_id, cluster.tree_id + 1)


# ============================================================================
# Top-level orchestrator
# ============================================================================
def process_run(
    run_root: str,
    cfg: Optional[SegmenterConfig] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    write_contact_sheets: bool = True,
    flower_counts_by_frame: Optional[Dict[int, float]] = None,
    roi_masks_by_frame: Optional[Dict[int, Optional[np.ndarray]]] = None,
) -> Tuple[List[TreeCluster], Dict[int, List[Tuple[int, float]]]]:
    """End-to-end: run the full pipeline on a staged orchard run.

    Parameters
    ----------
    run_root : path to the run directory (see :class:`PNGSequenceLoader`)
    cfg      : :class:`SegmenterConfig`. Defaults if None.
    output_dir : where to write contact sheets and a ``trees.json``
                 summary. Defaults to ``run_root/sam2_segmenter_out``.
    device   : "cuda" or "cpu". CPU is feasible but slow (~30 min/run).
    flower_counts_by_frame
        Optional ``{frame_idx: n_flowers}`` from ``analyze_days.py``.
        When provided, calls :func:`attribute_flowers_via_roi` to
        populate ``cluster.flower_count`` on every returned cluster.
        Build with :func:`load_flower_counts_from_csv`.
    roi_masks_by_frame
        Optional ``{frame_idx: bool_array_or_None}`` PRGB ROI masks.
        When None (default), the loader's :meth:`FrameLoader.load_roi_mask`
        is called for each frame. Override that method in your loader
        subclass, or pass precomputed masks here.

    Returns
    -------
    clusters : the persistent tree list (``cluster.flower_count`` is set
               when *flower_counts_by_frame* is provided).
    attribution : ``{frame_idx: [(tree_id, weight), ...]}`` mapping.
    """
    cfg = cfg or SegmenterConfig()
    out_root = Path(output_dir or (Path(run_root) / "sam2_segmenter_out"))
    out_root.mkdir(parents=True, exist_ok=True)

    loader = PNGSequenceLoader(run_root)

    # 1. Detection.
    detections = detect_trunks_grounding_dino(loader, cfg, device=device)

    # 2. SAM2 propagation.
    tracks = propagate_with_sam2(loader, detections, cfg, device=device)

    # 3. World projection.
    tracks = project_tracks_to_world(tracks, loader, cfg)

    # 4. Cluster.
    clusters = cluster_to_trees(tracks, cfg)

    # 5. Sanity check.
    if cfg.spacing_check:
        clusters = sanity_check_spacing(clusters, cfg)

    # 6. Frame attribution.
    attribution = attribute_frames(clusters, cfg)

    # 6b. Flower attribution via PRGB ROI (optional).
    if flower_counts_by_frame:
        if roi_masks_by_frame is None:
            roi_masks_by_frame = {
                fid: loader.load_roi_mask(fid)
                for fid in loader.frame_indices()
            }
        attribute_flowers_via_roi(
            clusters, attribution, flower_counts_by_frame, roi_masks_by_frame,
        )

    # 7. Verification artifacts.
    if write_contact_sheets:
        build_contact_sheets(clusters, loader, str(out_root / "contact_sheets"), cfg)

    # 8. Summary JSON for downstream consumption.
    summary = {
        "n_trees": len(clusters),
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "trees": [
            {
                "tree_id": c.tree_id,
                "world_lat": c.world_lat,
                "world_lon": c.world_lon,
                "n_frames": c.n_frames,
                "first_frame": c.first_frame,
                "last_frame": c.last_frame,
                "flagged_reason": c.flagged_reason,
                "flower_count": round(c.flower_count, 1),
            }
            for c in clusters
        ],
        "frame_attribution": {
            str(fid): [(tid, round(w, 4)) for tid, w in items]
            for fid, items in attribution.items()
        },
    }
    with (out_root / "trees.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    log.info("Done: %d trees, %d frames attributed",
             len(clusters), len(attribution))
    return clusters, attribution


# ============================================================================
# CLI helpers
# ============================================================================
def _walk_all2023_sessions(root: Path) -> List[Path]:
    """Return every session directory under an All2023 root.

    Mirrors the directory walk in ``analyze_days.find_images`` so the
    two tools always see the same sessions.
    """
    sessions: List[Path] = []
    for day_dir in sorted(root.glob("2023 day *")):
        inner = day_dir / day_dir.name
        day_root = inner if inner.is_dir() else day_dir
        for cat in sorted(p for p in day_root.iterdir() if p.is_dir()):
            for sess in sorted(p for p in cat.iterdir() if p.is_dir()):
                if (sess / "RGB").is_dir():
                    sessions.append(sess)
    return sessions


def _run_one_session(
    session_dir: Path,
    cfg: SegmenterConfig,
    args,
    out_base: Path,
) -> None:
    """Run the full pipeline on a single session directory."""
    import json as _json

    session_name = session_dir.name
    frame_range = tuple(args.frame_range) if args.frame_range else None

    loader = All2023FrameLoader(
        str(session_dir),
        row_heading_deg=getattr(args, "row_heading_deg", None),
        frame_range=frame_range,
        require_all_modalities=not getattr(
            args, "allow_missing_modalities", False,
        ),
    )
    if len(loader) == 0:
        log.warning("Session %s: no frames found, skipping", session_name)
        return

    flower_counts = None
    if args.flower_csv:
        flower_counts = load_flower_counts_from_csv(
            args.flower_csv,
            session=session_name,
            loader=loader,
            prompt=args.flower_prompt,
            count_col=args.flower_count_col,
        )
        log.info("Session %s: %d frames with flower data",
                 session_name, len(flower_counts))

    roi_masks = None
    if flower_counts:
        roi_masks = {fid: loader.load_roi_mask(fid) for fid in loader.frame_indices()}

    out_root = out_base / session_name
    out_root.mkdir(parents=True, exist_ok=True)

    detections = detect_trunks_grounding_dino(loader, cfg, device=args.device)
    tracks = propagate_with_sam2(loader, detections, cfg, device=args.device)
    tracks = project_tracks_to_world(tracks, loader, cfg)
    clusters = cluster_to_trees(tracks, cfg)
    if cfg.spacing_check:
        clusters = sanity_check_spacing(clusters, cfg)
    attribution = attribute_frames(clusters, cfg)

    if flower_counts:
        # Prefer 3D nearest-tree (per-flower mask centroid → world →
        # nearest cluster). Fall back per-frame to ROI overlap when
        # the mask .npz is missing or no flower projects in range.
        flower_masks_dir = None
        flower_slug = None
        if getattr(args, "flower_masks_dir", None):
            flower_masks_dir = Path(args.flower_masks_dir)
        elif args.out:
            candidate = Path(args.out) / "masks"
            if candidate.is_dir():
                flower_masks_dir = candidate
        if flower_masks_dir is not None:
            import re as _re
            flower_slug = _re.sub(
                r"[^a-z0-9]+", "_", args.flower_prompt.lower(),
            ).strip("_")
        attribute_flowers(
            clusters, attribution, flower_counts, roi_masks,
            loader=loader, cfg=cfg,
            dataset_root=Path(args.root),
            flower_masks_dir=flower_masks_dir,
            flower_slug=flower_slug,
            max_assign_distance_m=getattr(
                args, "flower_max_assign_distance_m", 3.0,
            ),
        )

    # Voxel + leaf-count + Beer-Lambert LAI via lai_voxel_estimator
    # if enabled. Writes lai_per_tree.json into the session output.
    lai_results = None
    if getattr(args, "lai", False):
        try:
            from lai_voxel_estimator import (
                LAIConfig, CameraIntrinsics, process_clusters_for_lai,
            )
            # LAIConfig defaults to D455 native 1280x720 intrinsics,
            # but the All2023 dataset is RGB-stream 640x480 (half-res).
            # Scale fx/fy/cx/cy proportionally to the actual frame
            # size so backproject_to_world's meshgrid matches the
            # mask/depth shape.
            sample_rgb = loader.load_rgb(loader.frame_indices()[0])
            ah, aw = sample_rgb.shape[:2]
            default_w, default_h = 1280, 720
            scale_x = aw / float(default_w)
            scale_y = ah / float(default_h)
            scaled_intr = CameraIntrinsics(
                fx=644.0 * scale_x,
                fy=644.0 * scale_y,
                cx=644.0 * scale_x,
                cy=360.0 * scale_y,
                width=aw,
                height=ah,
            )
            lai_cfg = LAIConfig(intrinsics=scaled_intr)
            lai_results = process_clusters_for_lai(
                clusters, loader, cfg=lai_cfg,
                output_dir=str(out_root / "lai"),
                device=args.device,
                write_yolo_labels=False,
            )
        except Exception as exc:
            log.error("LAI estimation failed: %s", exc, exc_info=True)
            lai_results = None

    # Per-cluster Beer-Lambert LAI from the per-frame tree masks.
    # For each frame the cluster has a tree_mask, compute canopy
    # fraction inside the tree's bounding column (the column the
    # tree occupies, full vertical extent) and convert via
    # -ln(1 - cf)/k. Average across frames. Mirrors the sprayer-
    # pipeline beer-lambert formula in sprayer_pipeline/tree_mask.py.
    K_BEER = 0.5
    try:
        from tree_mask import (
            zone_canopy_fractions as _zone_cfs,
            beer_lambert_lai as _beer_lambert,
            ROI_RECTS as _ROI_RECTS,
        )
    except ImportError:
        _zone_cfs = None
        _beer_lambert = None
        _ROI_RECTS = []
    for cluster in clusters:
        cfs: List[float] = []
        # Accumulate per-ROI canopy fraction across frames where
        # this cluster has a tree mask.
        per_roi_cfs: List[List[float]] = [[] for _ in _ROI_RECTS]
        for track in cluster.tracks:
            for det in track.detections:
                if det.tree_mask is None or not det.tree_mask.any():
                    continue
                ys, xs = np.where(det.tree_mask)
                x_lo, x_hi = int(xs.min()), int(xs.max()) + 1
                col = det.tree_mask[:, x_lo:x_hi]
                area = col.size
                if area == 0:
                    continue
                cf = float(col.sum()) / float(area)
                cfs.append(cf)
                # Per-ROI canopy fraction inside the 10 sprayer zones.
                if _zone_cfs is not None:
                    zcfs = _zone_cfs(det.tree_mask)
                    for i, z in enumerate(zcfs):
                        per_roi_cfs[i].append(z)
        if cfs:
            mean_cf = float(np.mean(cfs))
            gap = max(0.01, min(0.99, 1.0 - mean_cf))
            cluster.lai_beer_lambert = float(-math.log(gap) / K_BEER)
        else:
            cluster.lai_beer_lambert = float("nan")
        # Per-ROI LAI = mean per-ROI cf across frames -> Beer-Lambert.
        cluster.lai_per_roi = [
            (float(_beer_lambert(float(np.mean(zs))))
             if zs and _beer_lambert is not None
             else float("nan"))
            for zs in per_roi_cfs
        ]

    if not args.no_contact_sheets:
        build_contact_sheets(clusters, loader, str(out_root / "contact_sheets"), cfg)

    summary = {
        "session": session_name,
        "n_trees": len(clusters),
        "trees": [
            {
                "tree_id": c.tree_id,
                "world_lat": c.world_lat,
                "world_lon": c.world_lon,
                "n_frames": c.n_frames,
                "first_frame": c.first_frame,
                "last_frame": c.last_frame,
                "flagged_reason": c.flagged_reason,
                "flower_count": round(c.flower_count, 1),
                "lai_beer_lambert": (
                    round(c.lai_beer_lambert, 3)
                    if math.isfinite(c.lai_beer_lambert) else None
                ),
                "lai_per_roi": [
                    (round(v, 3) if math.isfinite(v) else None)
                    for v in (c.lai_per_roi or [])
                ],
                **(
                    {
                        "lai_voxel": next(
                            (round(r.voxel_lai, 3) for r in lai_results
                             if r.tree_id == c.tree_id),
                            None,
                        ),
                        "lai_leaf_count": next(
                            (round(r.leaf_count_lai, 3) for r in lai_results
                             if r.tree_id == c.tree_id
                             and math.isfinite(r.leaf_count_lai)),
                            None,
                        ),
                        "lai_calibrated": next(
                            (round(r.calibrated_lai, 3) for r in lai_results
                             if r.tree_id == c.tree_id),
                            None,
                        ),
                    } if lai_results else {}
                ),
            }
            for c in clusters
        ],
    }
    with (out_root / "trees.json").open("w") as fh:
        _json.dump(summary, fh, indent=2)

    print(f"\n[{session_name}] {len(clusters)} trees detected")
    for c in clusters:
        flag = f"  [{c.flagged_reason}]" if c.flagged_reason else ""
        flowers = f"  flowers={c.flower_count:.0f}" if c.flower_count else ""
        lai = (f"  LAI={c.lai_beer_lambert:.2f}"
               if math.isfinite(c.lai_beer_lambert) else "")
        print(f"  Tree {c.tree_id:3d}: "
              f"({c.world_lat:.7f}, {c.world_lon:.7f})  "
              f"frames {c.first_frame}-{c.last_frame} "
              f"(n={c.n_frames}){flowers}{lai}{flag}")


# ============================================================================
# CLI
# ============================================================================
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Trunk-anchored orchard tree segmentation",
    )
    parser.add_argument("--root", type=str, required=True,
                        help="All2023 dataset root (same path you pass to "
                             "analyze_days.py --root). Every session folder "
                             "found under this root is processed automatically.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory. One sub-folder per session is "
                             "created here, each containing trees.json and "
                             "optional contact_sheets/.")
    parser.add_argument("--flower-csv", type=str, default=None,
                        help="Path to analyze_days.py results.csv. When provided, "
                             "flower counts are attributed to trees via PRGB ROI.")
    parser.add_argument("--flower-prompt", type=str, default="flower",
                        help="Prompt to pull from results.csv (default: flower).")
    parser.add_argument("--flower-count-col", type=str, default="est_flowers",
                        choices=["est_flowers", "n_detections"],
                        help="Count column to use (default: est_flowers).")
    parser.add_argument("--flower-masks-dir", type=str, default=None,
                        help="Directory containing per-frame flower mask "
                             ".npz files written by analyze_days.py "
                             "--save-masks. Defaults to <out>/masks if that "
                             "directory exists. Enables 3D nearest-tree "
                             "flower attribution; when missing, the pipeline "
                             "falls back to PRGB-ROI overlap only.")
    parser.add_argument("--flower-max-assign-distance-m", type=float,
                        default=3.0,
                        help="Max metres between a flower's projected world "
                             "coord and a tree's centroid for the 3D "
                             "attribution path (default 3.0). Frames with no "
                             "flower in range fall back to ROI overlap.")
    parser.add_argument("--frame-range", type=int, nargs=2,
                        default=None, metavar=("START", "STOP"),
                        help="Process only frames [START, STOP) — use the same "
                             "values as --frame-range in analyze_days.py.")
    parser.add_argument("--dbscan-eps-m", type=float, default=None,
                        help="Max distance in metres between trunk detections of the "
                             "same tree (DBSCAN eps). Default: auto-estimated from "
                             "the k-distance graph elbow so it adapts to actual trunk "
                             "spacing in the data. Override with a fixed value if "
                             "the auto-estimate over- or under-splits.")
    parser.add_argument("--dbscan-min-samples", type=int, default=2,
                        help="Minimum trunk detections required to form a tree "
                             "cluster (default 2). Lower this for sparse "
                             "fly-by data where a tree may only be seen in 2 "
                             "frames; raise it to suppress single-frame "
                             "GDINO false positives.")
    parser.add_argument("--row-heading-deg", type=float, default=None,
                        help="Orchard row compass bearing. Auto-estimated from "
                             "GPS trail when omitted.")
    parser.add_argument("--camera-hfov-deg", type=float, default=87.0,
                        help="Camera horizontal FOV in degrees (default 87 "
                             "= RealSense D455F datasheet, matches "
                             "sprayer_pipeline.config.CAMERA_HFOV_DEG). Used "
                             "to convert each detection's pixel-x into a "
                             "true lateral world offset.")
    parser.add_argument("--trunk-min-depth-m", type=float, default=0.3,
                        help="Reject trunks whose mask depth is below this "
                             "many metres (default 0.3).")
    parser.add_argument("--trunk-max-depth-m", type=float, default=15.0,
                        help="Reject trunks whose mask depth exceeds this "
                             "many metres (default 15.0 — RealSense D435 "
                             "effective range). Lower to e.g. 5 if the "
                             "camera is foreground-aimed; raise if it looks "
                             "sideways across the aisle.")
    parser.add_argument("--nominal-tree-distance-m", type=float,
                        default=1.5,
                        help="Fallback perpendicular distance from camera "
                             "to the sprayed row (default 1.5 m). Used when "
                             "RealSense returns no depth for a thin trunk. "
                             "Combined with the trunk's pixel-x angle this "
                             "gives a usable world projection so the same "
                             "tree across passes / front-back views still "
                             "clusters together. Set to 0 to disable the "
                             "fallback (strict depth-only mode).")
    parser.add_argument("--trunk-min-roi-overlap", type=float, default=0.10,
                        help="Reject trunk detections whose bbox overlaps "
                             "the (vertically-extended) PRGB ROI by less "
                             "than this fraction of the bbox area (default "
                             "0.10, matches analyze_days.py default). The "
                             "PRGB ROI marks the tree currently being "
                             "sprayed; trunks outside it are background. "
                             "Set to 0 to disable.")
    parser.add_argument("--lai", action="store_true",
                        help="Run lai_voxel_estimator after clustering: "
                             "hierarchical SAM 3 sub-segmentation + voxel "
                             "fusion + leaf count + Beer-Lambert. Writes "
                             "lai_per_tree.json into <session>/lai/.")
    parser.add_argument("--allow-missing-modalities", action="store_true",
                        help="By default the loader drops any frame missing "
                             "depth, PRGB, or Info — mirrors analyze_days.py "
                             "--require-all-modalities --skip-no-roi. Set "
                             "this flag to keep partial frames and let "
                             "downstream stages handle missing data.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-contact-sheets", action="store_true")
    parser.add_argument("--gdino-model", type=str,
                        default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam2-cfg", type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-ckpt", type=str,
                        default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Suppress noisy third-party debug loggers regardless of -v.
    for _noisy in ("PIL", "httpx", "httpcore", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    cfg = SegmenterConfig(
        gdino_model_id=args.gdino_model,
        sam2_model_cfg=args.sam2_cfg,
        sam2_checkpoint=args.sam2_ckpt,
        dbscan_eps_m=args.dbscan_eps_m,
        dbscan_min_samples=args.dbscan_min_samples,
        camera_hfov_deg=args.camera_hfov_deg,
        trunk_min_depth_m=args.trunk_min_depth_m,
        trunk_max_depth_m=args.trunk_max_depth_m,
        trunk_min_roi_overlap=args.trunk_min_roi_overlap,
        nominal_tree_distance_m=args.nominal_tree_distance_m,
    )

    sessions = _walk_all2023_sessions(Path(args.root))
    if not sessions:
        print(f"No sessions found under {args.root}")
        return
    print(f"Found {len(sessions)} sessions under {args.root}")

    out_base = Path(args.out) / "trees"
    out_base.mkdir(parents=True, exist_ok=True)

    for session_dir in sessions:
        try:
            _run_one_session(session_dir, cfg, args, out_base)
        except Exception as exc:
            log.error("Session %s failed: %s", session_dir.name, exc, exc_info=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "SegmenterConfig",
    "TrunkDetection",
    "TrunkTrack",
    "TreeCluster",
    "FrameLoader",
    "All2023FrameLoader",
    "PNGSequenceLoader",
    "detect_trunks_grounding_dino",
    "propagate_with_sam2",
    "project_tracks_to_world",
    "cluster_to_trees",
    "sanity_check_spacing",
    "attribute_frames",
    "attribute_flowers_via_roi",
    "attribute_flowers_3d_nearest_tree",
    "attribute_flowers",
    "load_flower_counts_from_csv",
    "build_contact_sheets",
    "populate_tree_tracker",
    "process_run",
]
