"""Global-masked parallel LK optical flow estimator with affine transform support.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LKEstimator:
    """Global-masked parallel LK optical flow estimator.

    Keyframe: extracts and stores feature points inside all target BBoxes.
    Skip frame: tracks points frame-by-frame, accumulates offsets, extracts affine params.

    Parameters
    ----------
    max_features_per_bbox : int
        Max feature points per BBox (default 12).
    affine_min_points : int
        Minimum points required for affine estimation (default 3).
    cumulative_scale_range : tuple[float, float]
        Cumulative scale clamping range (default 0.85 ~ 1.15).
    """

    def __init__(
        self,
        max_features_per_bbox: int = 12,
        affine_min_points: int = 3,
        cumulative_scale_range: tuple[float, float] = (0.85, 1.15),
    ) -> None:
        self._max_points = max_features_per_bbox
        self._affine_min_points = affine_min_points
        self._scale_clamp = cumulative_scale_range

        self._prev_gray: np.ndarray | None = None
        self._feature_points: np.ndarray | None = None   # (N, 1, 2)  updated each frame
        self._point_owners: np.ndarray | None = None      # (N,)       updated each frame
        self._num_bboxes: int = 0
        self._bbox_centers: np.ndarray | None = None      # (K, 2)     keyframe BBox centers

        self._accum_dx: list[float] = []
        self._accum_dy: list[float] = []
        self._accum_scale: list[float] = []

        self._last_offsets: list[tuple[float, float]] = []
        self._last_scales: list[float] = []

        self._mask_time_ms: float = 0.0
        self._goodfeat_time_ms: float = 0.0
        self._lk_time_ms: float = 0.0
        self._total_time_ms: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_keyframe(
        self, gray: np.ndarray, bboxes_xyxy: np.ndarray,
    ) -> None:
        """Extract feature points from a keyframe and reset accumulators.

        Parameters
        ----------
        gray : np.ndarray
            (H, W) uint8 grayscale image.
        bboxes_xyxy : np.ndarray
            (K, 4) float32 detection boxes in ``[x1, y1, x2, y2]`` format.
        """
        h, w = gray.shape[:2]
        K = len(bboxes_xyxy)

        self._accum_dx = [0.0] * K
        self._accum_dy = [0.0] * K
        self._accum_scale = [1.0] * K
        self._last_offsets = []
        self._last_scales = []

        # Store BBox centers for spatial matching (Fix #1)
        if K > 0:
            self._bbox_centers = np.column_stack([
                (bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2,
                (bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) / 2,
            ]).astype(np.float32)
        else:
            self._bbox_centers = None

        if K == 0:
            self._prev_gray = gray.copy()
            self._feature_points = None
            self._point_owners = None
            self._num_bboxes = 0
            return

        t0 = time.perf_counter()

        mask = self._create_mask(h, w, bboxes_xyxy)
        t1 = time.perf_counter()
        self._mask_time_ms = (t1 - t0) * 1000

        max_corners = K * self._max_points
        points = cv2.goodFeaturesToTrack(
            gray, mask=mask, maxCorners=max_corners,
            qualityLevel=0.01, minDistance=5, blockSize=7,
        )
        t2 = time.perf_counter()
        self._goodfeat_time_ms = (t2 - t1) * 1000

        if points is None or len(points) == 0:
            self._prev_gray = gray.copy()
            self._feature_points = None
            self._point_owners = None
            self._num_bboxes = K
            return

        inner_bboxes = self._shrink_bboxes(bboxes_xyxy, h, w)
        filtered_pts, owners = self._limit_points_per_bbox(points, inner_bboxes)

        self._prev_gray = gray.copy()
        self._feature_points = filtered_pts
        self._point_owners = owners
        self._num_bboxes = K

        t3 = time.perf_counter()
        self._total_time_ms = (t3 - t0) * 1000

    def compute_offsets(
        self, cur_gray: np.ndarray,
    ) -> list[tuple[float, float, float]]:
        """Track feature points frame-by-frame and return cumulative offsets.

        Each call tracks points from ``_prev_gray`` to ``cur_gray`` (small displacement),
        updates feature point coordinates, and accumulates total offset per BBox.

        Returns
        -------
        list[tuple[float, float, float]]
            Cumulative ``(dx, dy, scale)`` per BBox since the last keyframe.
            Scale is the cumulative product.
        """
        if (
            self._prev_gray is None
            or self._feature_points is None
            or self._point_owners is None
            or self._num_bboxes == 0
        ):
            self._last_offsets = []
            self._last_scales = []
            return []

        t0 = time.perf_counter()

        new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, cur_gray, self._feature_points, None,
            winSize=(15, 15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        t1 = time.perf_counter()
        self._lk_time_ms = (t1 - t0) * 1000

        status = status.ravel()
        valid_mask = status == 1

        if not np.any(valid_mask):
            # Fix #5: return per-frame (0,0) increments, not cumulative values
            self._prev_gray = cur_gray.copy()
            self._last_offsets = [(0.0, 0.0)] * self._num_bboxes
            self._last_scales = [1.0] * self._num_bboxes
            return [
                (self._accum_dx[k], self._accum_dy[k], self._accum_scale[k])
                for k in range(self._num_bboxes)
            ]

        valid_old = self._feature_points[valid_mask]
        valid_new = new_pts[valid_mask]
        valid_owners = self._point_owners[valid_mask]

        # Fix #3: keep only successfully tracked points, discard failed ones
        self._feature_points = valid_new.copy()
        self._point_owners = valid_owners.copy()

        frame_offsets: list[tuple[float, float]] = []
        frame_scales: list[float] = []

        for k in range(self._num_bboxes):
            idx = np.where(valid_owners == k)[0]
            n_pts = len(idx)

            # Fix #4: affine_min_points=3 lets most targets use affine estimation
            if n_pts < self._affine_min_points:
                if n_pts > 0:
                    dx = float(np.median(valid_new[idx, 0, 0] - valid_old[idx, 0, 0]))
                    dy = float(np.median(valid_new[idx, 0, 1] - valid_old[idx, 0, 1]))
                else:
                    dx, dy = 0.0, 0.0
                scale = 1.0
            else:
                old_k = valid_old[idx].reshape(-1, 2).astype(np.float32)
                new_k = valid_new[idx].reshape(-1, 2).astype(np.float32)
                try:
                    result = cv2.estimateAffinePartial2D(old_k, new_k, method=0)
                    M = result[0] if result is not None else None
                except cv2.error:
                    M = None

                if M is not None and M.shape == (2, 3):
                    dx = float(M[0, 2])
                    dy = float(M[1, 2])
                    scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
                    scale = np.clip(scale, 0.95, 1.05)
                else:
                    dx = float(np.median(valid_new[idx, 0, 0] - valid_old[idx, 0, 0]))
                    dy = float(np.median(valid_new[idx, 0, 1] - valid_old[idx, 0, 1]))
                    scale = 1.0

            frame_offsets.append((dx, dy))
            frame_scales.append(scale)

            self._accum_dx[k] += dx
            self._accum_dy[k] += dy
            self._accum_scale[k] *= scale
            # Fix #6: clamp cumulative scale
            self._accum_scale[k] = np.clip(self._accum_scale[k], *self._scale_clamp)

        self._last_offsets = frame_offsets
        self._last_scales = frame_scales
        self._prev_gray = cur_gray.copy()

        t2 = time.perf_counter()
        self._total_time_ms = (t2 - t0) * 1000

        return [
            (self._accum_dx[k], self._accum_dy[k], self._accum_scale[k])
            for k in range(self._num_bboxes)
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def offsets(self) -> list[tuple[float, float, float]]:
        """Cumulative ``[(dx, dy, scale), ...]`` per BBox since last keyframe."""
        return [
            (self._accum_dx[k], self._accum_dy[k], self._accum_scale[k])
            for k in range(self._num_bboxes)
        ]

    @property
    def bbox_centers(self) -> np.ndarray | None:
        """Keyframe BBox centers ``(K, 2)`` for spatial track-delta matching (Fix #1)."""
        return self._bbox_centers

    @property
    def incremental_offsets(self) -> list[tuple[float, float]]:
        """Per-frame incremental ``[(dx, dy), ...]`` from the most recent call."""
        return self._last_offsets

    @property
    def incremental_scales(self) -> list[float]:
        """Per-frame incremental scale factors from the most recent call."""
        return self._last_scales

    @property
    def timing(self) -> dict[str, float]:
        """Timing breakdown (ms) for the most recent operation."""
        return {
            "mask": self._mask_time_ms,
            "goodFeatures": self._goodfeat_time_ms,
            "lk_flow": self._lk_time_ms,
            "total": self._total_time_ms,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shrink_bboxes(bboxes: np.ndarray, h: int, w: int) -> np.ndarray:
        """Shrink boxes inward by 15% and clamp to image boundaries."""
        b = bboxes.astype(np.float64).copy()
        bw = (b[:, 2] - b[:, 0]) * 0.15
        bh = (b[:, 3] - b[:, 1]) * 0.15
        b[:, 0] += bw; b[:, 1] += bh
        b[:, 2] -= bw; b[:, 3] -= bh
        b[:, 0] = np.clip(np.floor(b[:, 0]), 0, w - 1)
        b[:, 1] = np.clip(np.floor(b[:, 1]), 0, h - 1)
        b[:, 2] = np.clip(np.ceil(b[:, 2]), 1, w)
        b[:, 3] = np.clip(np.ceil(b[:, 3]), 1, h)
        return b.astype(np.int32)

    @classmethod
    def _create_mask(cls, h: int, w: int, bboxes_xyxy: np.ndarray) -> np.ndarray:
        """Build a single-channel mask with ``cv2.rectangle`` (C-level fill, zero Python loops)."""
        if len(bboxes_xyxy) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        bboxes = cls._shrink_bboxes(bboxes_xyxy, h, w)
        mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in bboxes:
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
        return mask

    def _limit_points_per_bbox(
        self, points: np.ndarray, bboxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign points to BBoxes and cap at ``_max_points`` per BBox."""
        K = len(bboxes)
        N = len(points)
        owners = np.full(N, -1, dtype=np.int32)
        pt = points[:, 0]
        for k in range(K):
            x1, y1, x2, y2 = bboxes[k]
            inside = (
                (pt[:, 0] >= x1) & (pt[:, 0] < x2)
                & (pt[:, 1] >= y1) & (pt[:, 1] < y2)
            )
            owners[(owners < 0) & inside] = k
        assigned = owners >= 0
        points = points[assigned]
        owners = owners[assigned]
        if len(points) == 0:
            return points, owners
        keep = np.zeros(len(points), dtype=bool)
        for k in range(K):
            idx = np.where(owners == k)[0]
            keep[idx[: self._max_points]] = True
        return points[keep], owners[keep]
