"""Adaptive frame-skip ByteTrack — spatial matching + covariance deception.
"""

from __future__ import annotations

import logging

import numpy as np

from ultralytics.trackers.byte_tracker import BYTETracker

logger = logging.getLogger(__name__)


class SkippingByteTracker(BYTETracker):
    """Adaptive frame-skip ByteTrack.

    Inherits from ``BYTETracker``, overrides ``update()`` to switch between
    keyframe (full association) and skip-frame (predict + covariance only) logic.

    Parameters
    ----------
    args : namespace
        Tracker config (same as native BYTETracker).
    frame_rate : int
    alpha_cov : float
        Covariance decay coefficient (default 0.6).
    match_dist_px : float
        Max pixel distance for spatial track-delta matching.
    """

    def __init__(
        self,
        args,
        frame_rate: int = 30,
        alpha_cov: float = 0.6,
        match_dist_px: float = 100.0,
    ) -> None:
        super().__init__(args, frame_rate)
        self._skip_counter: int = 0
        self._alpha_cov: float = alpha_cov
        self._match_dist_px: float = match_dist_px

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def update(
        self,
        det_results,
        img: np.ndarray | None = None,
        feats: np.ndarray | None = None,
        *,
        is_keyframe: bool = True,
        optical_flow_deltas: list[tuple[float, float, float]] | None = None,
        optical_flow_centers: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update tracker state.

        Parameters
        ----------
        optical_flow_deltas : cumulative ``[(dx, dy, scale), ...]`` per BBox.
        optical_flow_centers : (K, 2) keyframe BBox centers for spatial matching.
        """
        if is_keyframe:
            return self._update_keyframe(
                det_results, img, feats,
                optical_flow_deltas, optical_flow_centers,
            )
        else:
            return self._update_skip_frame(det_results, img)

    # ------------------------------------------------------------------
    # Keyframe: spatial-match pre-alignment + relaxed matching
    # ------------------------------------------------------------------

    def _update_keyframe(
        self,
        det_results,
        img: np.ndarray | None,
        feats: np.ndarray | None,
        deltas: list[tuple[float, float, float]] | None,
        centers: np.ndarray | None,
    ) -> np.ndarray:
        # Fix #1: spatial matching — nearest BBox center per track
        if deltas and centers is not None and len(centers) > 0:
            for track in self.tracked_stracks:
                if track.mean is None:
                    continue
                tc = np.array([track.mean[0], track.mean[1]], dtype=np.float32)
                dists = np.sqrt(np.sum((centers - tc) ** 2, axis=1))
                nearest = int(np.argmin(dists))
                if dists[nearest] < self._match_dist_px and nearest < len(deltas):
                    dx, dy, scale = deltas[nearest]
                    track.mean[0] += dx
                    track.mean[1] += dy
                    # Scale breathing only for significant scale changes
                    if not (0.98 < scale < 1.02):
                        track.mean[3] *= scale

        had_skips = self._skip_counter > 0
        skip_count_before = self._skip_counter
        self._skip_counter = 0

        if det_results is None or len(det_results) == 0:
            self.frame_id += 1
            for track in self.tracked_stracks:
                if track.is_activated:
                    track.mark_lost()
            removed = []
            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed.append(track)
            for t in removed:
                if t in self.lost_stracks:
                    self.lost_stracks.remove(t)
                if t in self.tracked_stracks:
                    self.tracked_stracks.remove(t)
            self.removed_stracks.extend(removed)
            return np.asarray([], dtype=np.float32).reshape(0, 8)

        # Relax match threshold on first keyframe after skip sequence
        # (distance threshold = 1-IoU, so higher = more permissive)
        original_match_thresh = self.args.match_thresh
        if had_skips:
            relaxed = min(original_match_thresh + 0.15, 0.95)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Relaxing match_thresh: %.2f -> %.2f (after %d skips)",
                    original_match_thresh, relaxed, skip_count_before,
                )
            self.args.match_thresh = relaxed

        try:
            return super().update(det_results, img, feats)
        finally:
            self.args.match_thresh = original_match_thresh

    # ------------------------------------------------------------------
    # Skip frame: Kalman predict + covariance deception only (Fix #2)
    # ------------------------------------------------------------------

    def _update_skip_frame(
        self, det_results, img: np.ndarray | None,
    ) -> np.ndarray:
        """Skip frame: pure Kalman predict + covariance deception.

        No offset injection here. Cumulative offsets are applied once during
        the next keyframe pre-alignment via spatial matching.
        """
        self._skip_counter += 1
        self.frame_id += 1

        for track in self.tracked_stracks:
            if not track.is_activated:
                continue

            track.predict()

            # Zero position-velocity cross-terms to prevent velocity variance leakage
            track.covariance[0, 4] = 0.0
            track.covariance[4, 0] = 0.0
            track.covariance[1, 5] = 0.0
            track.covariance[5, 1] = 0.0

            # Damp position covariance (simulates KF update compression)
            track.covariance[0, 0] *= self._alpha_cov
            track.covariance[1, 1] *= self._alpha_cov

            # Mild velocity damping to prevent Q accumulation
            track.covariance[4, 4] *= (self._alpha_cov + 0.2)
            track.covariance[5, 5] *= (self._alpha_cov + 0.2)

            track.frame_id = self.frame_id

        # Do NOT call super().update() — no association, no track loss
        return np.asarray(
            [x.result for x in self.tracked_stracks if x.is_activated],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    @property
    def skip_counter(self) -> int:
        """Number of consecutive skip frames since the last keyframe."""
        return self._skip_counter

    @property
    def n_tracked(self) -> int:
        """Number of currently activated tracked tracks."""
        return sum(1 for t in self.tracked_stracks if t.is_activated)
