"""Adaptive frame-skip tracker — drop-in replacement for ultralytics.YOLO.

Provides a proxy ``YOLO`` class that wraps ``ultralytics.YOLO`` internally.
The ``track()`` method uses ``cv2.VideoCapture`` as the frame source and
automatically switches between **keyframes** (YOLO + ByteTrack) and
**skip frames** (optical flow + Kalman prediction).

Usage::

    # from ultralytics import YOLO
    from adaptive_tracker import YOLO

    model = YOLO("yolov8n.pt")
    for result in model.track("video.mp4"):
        cv2.imshow("Tracking", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, Union

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO as _UltralyticsYOLO
from ultralytics.engine.results import Results
from ultralytics.utils import IterableSimpleNamespace

from .lk_estimator import LKEstimator
from .skipping_byte_tracker import SkippingByteTracker
from .skip_policy import keyframe_policy

logger = logging.getLogger(__name__)

__all__ = ["YOLO", "Results"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_tracker_config(tracker_name: str = "bytetrack.yaml") -> IterableSimpleNamespace:
    """Load ultralytics tracker config as a namespace object.

    Searches multiple locations to be robust across ultralytics versions
    and installation methods (pip, conda, source).
    """
    import ultralytics as _ultra
    from importlib import resources

    # 1. Try importlib.resources first (Python 3.9+ standard, works with pip)
    try:
        ref = resources.files("ultralytics") / "cfg" / "trackers" / tracker_name
        if ref.is_file():
            cfg_dict = yaml.safe_load(ref.read_text(encoding="utf-8"))
            return IterableSimpleNamespace(**cfg_dict)
    except Exception:
        pass

    # 2. Fallback: resolve via ultralytics package directory
    ultra_dir = os.path.dirname(_ultra.__file__)
    yaml_path = os.path.join(ultra_dir, "cfg", "trackers", tracker_name)
    if os.path.isfile(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as fh:
            cfg_dict = yaml.safe_load(fh)
        return IterableSimpleNamespace(**cfg_dict)

    raise FileNotFoundError(
        f"Cannot locate tracker config '{tracker_name}' in ultralytics installation."
    )


def _build_results(
    frame: np.ndarray,
    frame_idx: int,
    tracks: np.ndarray | None,
    names: dict,
) -> Results:
    """Wrap tracker output ``[x1,y1,x2,y2,track_id,score,cls,idx]`` into a Results object."""
    if tracks is None or len(tracks) == 0:
        return Results(
            orig_img=frame,
            path=str(frame_idx),
            names=names,
            speed={"preprocess": None, "inference": None, "postprocess": None},
        )

    # tracks (N, 8) -> drop last column (idx), keep 7 cols for Boxes
    # [x1, y1, x2, y2, track_id, score, cls]
    boxes_tensor = torch.as_tensor(tracks[:, :-1], dtype=torch.float32)

    return Results(
        orig_img=frame,
        path=str(frame_idx),
        names=names,
        boxes=boxes_tensor,
        speed={"preprocess": None, "inference": None, "postprocess": None},
    )


# ---------------------------------------------------------------------------
# YOLO proxy class
# ---------------------------------------------------------------------------

class YOLO:
    """Adaptive frame-skip tracking proxy — drop-in replacement for ``ultralytics.YOLO``.

    Parameters
    ----------
    model : str or Path
        Path to an ultralytics model (e.g. ``"yolov8n.pt"``).
    task : str or None
        Optional task override (``"detect"``, ``"segment"``, ``"pose"``, ...).
    verbose : bool
        Print model info at load time.
    """

    def __init__(
        self,
        model: str | Path = "yolov8n.pt",
        task: str | None = None,
        verbose: bool = False,
    ) -> None:
        self._model = _UltralyticsYOLO(model=model, task=task, verbose=verbose)

    # ------------------------------------------------------------------
    # Core API — track()
    # ------------------------------------------------------------------

    def track(
        self,
        source: str | Path,
        stream: bool = True,
        tracker: str = "bytetrack.yaml",
        conf: float = 0.25,
        iou: float = 0.7,
        device: str | None = None,
        vid_stride: int = 1,
        half: bool = False,
        classes: list[int] | None = None,
        verbose: bool = True,
        keyframe_interval: int = 5,
        max_features_per_bbox: int = 12,
        alpha_cov: float = 0.6,
        **kwargs,
    ) -> Union[list[Results], Generator[Results, None, None]]:
        """Adaptive frame-skip tracking.

        Parameters
        ----------
        source : str or Path
            Path to video file.
        stream : bool
            ``True`` -> return generator; ``False`` -> return list.
        tracker : str
            Tracker config filename (``"bytetrack.yaml"`` or ``"botsort.yaml"``).
        conf : float
            Detection confidence threshold.
        iou : float
            NMS IoU threshold.
        device : str or None
            Inference device (``"cpu"``, ``"cuda:0"``, ...).
        vid_stride : int
            Process every N-th frame (1 = process all).
        half : bool
            FP16 half-precision inference (GPU only).
        classes : list[int] or None
            Filter to these class indices.
        verbose : bool
            Print per-frame timing and frame type.
        keyframe_interval : int
            Number of frames between YOLO keyframes.
        max_features_per_bbox : int
            Max feature points extracted per bounding box.
        alpha_cov : float
            Kalman covariance decay coefficient.
        **kwargs
            Additional arguments forwarded to ``model.predict()``.

        Returns
        -------
        list[Results] or Generator[Results, None, None]
        """
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {source!s}")

        generator = self._stream_track(
            cap=cap,
            tracker=tracker,
            conf=conf,
            iou=iou,
            device=device,
            vid_stride=vid_stride,
            half=half,
            classes=classes,
            verbose=verbose,
            keyframe_interval=keyframe_interval,
            max_features_per_bbox=max_features_per_bbox,
            alpha_cov=alpha_cov,
            **kwargs,
        )

        if stream:
            return generator
        else:
            return list(generator)

    # ------------------------------------------------------------------
    # Internal frame loop — Plan C: full takeover
    # ------------------------------------------------------------------

    def _stream_track(
        self,
        cap: cv2.VideoCapture,
        tracker: str,
        conf: float,
        iou: float,
        device: str | None,
        vid_stride: int,
        half: bool,
        classes: list[int] | None,
        verbose: bool,
        keyframe_interval: int,
        max_features_per_bbox: int,
        alpha_cov: float,
        **kwargs,
    ) -> Generator[Results, None, None]:
        """Core frame loop: keyframes run YOLO + ByteTrack, skip frames run optical flow + Kalman.

        try/finally guarantees ``cap.release()`` even if the caller exits the generator early.
        """
        lke = LKEstimator(max_features_per_bbox=max_features_per_bbox)
        byte_tracker: SkippingByteTracker | None = None
        frame_idx = 0
        keyframe_count = 0
        skip_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % vid_stride != 0:
                    frame_idx += 1
                    continue

                is_kf = keyframe_policy(frame_idx, keyframe_interval)

                t0 = time.perf_counter()

                if is_kf:
                    # ================================================
                    # Keyframe: YOLO detection + feature extraction + ByteTrack
                    # ================================================

                    # 1. YOLO detection
                    try:
                        results_list = self._model.predict(
                            source=frame,
                            conf=conf,
                            iou=iou,
                            device=device,
                            half=half,
                            classes=classes,
                            stream=False,
                            verbose=False,
                            **kwargs,
                        )
                    except Exception:
                        logger.exception("YOLO error at frame %d — skipping", frame_idx)
                        frame_idx += 1
                        continue

                    r = results_list[0] if isinstance(results_list, list) else results_list
                    det = r.boxes.cpu() if r.boxes is not None and len(r.boxes) > 0 else None

                    # 2. Lazy-init SkippingByteTracker
                    if byte_tracker is None:
                        cfg = _load_tracker_config(tracker)
                        byte_tracker = SkippingByteTracker(
                            args=cfg, frame_rate=30, alpha_cov=alpha_cov
                        )

                    # 3. Grab cumulative offsets + bbox centers BEFORE extract (Fix #1)
                    prev_offsets = lke.offsets if lke.offsets else None
                    prev_centers = lke.bbox_centers

                    # 4. Extract feature points from the new keyframe (resets accumulators)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if det is not None:
                        boxes_data = det.data.cpu().numpy()
                        lke.extract_from_keyframe(gray, boxes_data[:, :4])
                    else:
                        lke.extract_from_keyframe(gray, np.empty((0, 4)))

                    # 5. Keyframe tracker update (spatial-match pre-alignment + relaxed matching)
                    tracks = byte_tracker.update(
                        det,
                        img=frame,
                        is_keyframe=True,
                        optical_flow_deltas=prev_offsets,
                        optical_flow_centers=prev_centers,
                    )

                    yield _build_results(frame, frame_idx, tracks, self._model.names)
                    keyframe_count += 1

                else:
                    # ================================================
                    # Skip frame: optical flow accumulation + Kalman prediction
                    # ================================================

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if byte_tracker is not None and byte_tracker.n_tracked > 0:
                        # Accumulate per-frame optical flow (for next keyframe pre-alignment)
                        lke.compute_offsets(gray)
                        # Skip frame: Kalman predict + covariance deception only (Fix #2)
                        tracks = byte_tracker.update(
                            None, img=frame, is_keyframe=False,
                        )
                    else:
                        tracks = None

                    yield _build_results(frame, frame_idx, tracks, self._model.names)
                    skip_count += 1

                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000

                if verbose:
                    label = "KEY" if is_kf else "SKIP"
                    n_tracked = byte_tracker.n_tracked if byte_tracker else 0
                    logger.info(
                        "[%s] frame %4d | %5.1fms | tracked=%d | skip_cnt=%d",
                        label, frame_idx, elapsed_ms, n_tracked,
                        byte_tracker.skip_counter if byte_tracker else 0,
                    )

                frame_idx += 1

            if verbose:
                logger.info(
                    "Done: %d frames (%d key, %d skip)",
                    frame_idx, keyframe_count, skip_count,
                )

        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Transparent attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying ultralytics model."""
        try:
            _model = self.__dict__["_model"]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' "
                f"(and no _model to delegate to)"
            )
        return getattr(_model, name)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def model(self) -> _UltralyticsYOLO:
        """Underlying ``ultralytics.YOLO`` instance (escape hatch)."""
        return self._model
