"""Adaptive-Skip-BYTETracker — demo launch script.

Usage::

    python run_demo.py <video_path> [model_path]

    Example:
        # Download yolov8n.pt first:
        #   pip install ultralytics
        #   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

        python run_demo.py /path/to/video.mp4 yolov8n.pt
"""

from __future__ import annotations

import logging
import sys

import cv2

from adaptive_tracker import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_demo.py <video_path> [model_path]")
        print()
        print("  video_path   Path to input video file")
        print("  model_path   Path to YOLO model (default: yolov8n.pt)")
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "yolov8n.pt"

    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"{'='*50}")

    model = YOLO(model_path)

    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        conf=0.25,
        stream=True,
        keyframe_interval=5,
        max_features_per_bbox=12,
        alpha_cov=0.6,
        verbose=True,
    )

    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("Adaptive Skip Tracker", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
