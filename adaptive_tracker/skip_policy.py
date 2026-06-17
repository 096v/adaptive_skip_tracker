"""Keyframe scheduling policy — decides whether a frame runs YOLO or optical flow."""

from __future__ import annotations


def keyframe_policy(frame_idx: int, interval: int = 5) -> bool:
    """Fixed-interval keyframe policy.

    Frame 0 and every *interval*-th frame is a keyframe (full YOLO detection +
    ByteTrack association). All other frames are skip frames (optical flow +
    Kalman prediction only).

    Parameters
    ----------
    frame_idx : int
        Current frame index (0-based).
    interval : int
        Keyframe interval. ``5`` = 1 YOLO frame per 5 frames.

    Returns
    -------
    bool
        ``True`` for keyframe, ``False`` for skip frame.
    """
    return frame_idx % interval == 0
