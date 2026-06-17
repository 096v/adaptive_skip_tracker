"""Adaptive-Skip-BYTETracker — drop-in replacement for ultralytics.YOLO.

Frame-skip multi-object tracking that combines YOLO detection, optical flow,
and Kalman filtering to achieve ~3x speedup with minimal accuracy loss.

Usage::

    from adaptive_tracker import YOLO

    model = YOLO("yolov8n.pt")
    for result in model.track("video.mp4", keyframe_interval=5):
        cv2.imshow("Tracking", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

Reference
---------
- G. Bhat et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
- B. D. Lucas & T. Kanade, "An Iterative Image Registration Technique", IJCAI 1981.
"""

__version__ = "0.1.0"
__author__ = "aliyun7869160768"

from .main_api import YOLO, Results

__all__ = ["YOLO", "Results", "__version__"]
