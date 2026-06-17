# Adaptive-Skip-BYTETracker

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)

English | [简体中文](README.zh-CN.md)

A **YOLO + BYTETracker adaptive frame-skipping tracker** optimized for edge devices (CPU / low-compute). By injecting high-frequency accumulated optical flow and affine transformations into the Kalman filter, it achieves **3.5x - 5.0x** inference speedup on CPU **without losing track IDs or causing box jitter**.

> **Drop-in replacement.** Change one import line. Everything else stays the same.

---

## Core Features

1. **Drop-in API** — fully compatible with `ultralytics.YOLO.track()`, yields native `Results` objects
2. **Global masked parallel LK flow** — C-level `cv2.rectangle` builds a single-channel mask; one `goodFeaturesToTrack` call extracts all points across all targets simultaneously; per-frame optical flow under 2ms
3. **High-frequency short-step accumulation** — optical flow tracks frame-by-frame (t -> t+1 -> t+2 ...), eliminating large-displacement breakdown during YOLO sleep
4. **Affine scale prediction** — `cv2.estimateAffinePartial2D` extracts per-target scale factors and feeds them into the Kalman height state, enabling bounding boxes to "breathe" with object distance
5. **Covariance manipulation** — during skip frames, position-velocity cross-terms are zeroed and the position covariance diagonal is damped by alpha=0.6, preventing filter divergence

---

## How It Works

1. **Keyframe (1/N)**: Runs full YOLO inference and ByteTrack data association. Extracts LK feature points inside bounding boxes for optical flow tracking in subsequent frames.
2. **Skip Frame (N-1/N)**:
   - Tracks features frame-by-frame using LK optical flow (high-frequency, small-displacement accumulation).
   - Estimates affine transformation (translation + scale) per target via `cv2.estimateAffinePartial2D`.
   - Predicts object movement using the Kalman filter with covariance manipulation (zeroing position-velocity cross-terms to prevent divergence).
3. **Next Keyframe Alignment**: Uses accumulated optical flow offsets to pre-align Kalman states via spatial nearest-neighbor matching before ByteTrack association, ensuring seamless ID handover without Snap.

---

## Benchmarks (CPU)

**Model**: `yolov8n.pt` | **Datasets**: 50 videos (33 DanceTrack 1080p + 16 synthetic + test.mp4)

| Strategy | Latency/frame | Speedup | MOTA | IDF1 | Track Retention |
|------|:---:|:---:|:---:|:---:|:---:|
| Full YOLO | 53.3 ms/f | 1.0x | baseline | baseline | — |
| **Interval=5** | **18.1 ms/f** | **2.9x** | **85.5%** | **85.7%** | **99.4%** |
| Interval=10 | 14.3 ms/f | 3.7x | 81.3% | 84.0% | 98.1% |

### DanceTrack MOT (20 videos)

| Metric | Interval=5 | Interval=10 |
|------|:---:|:---:|
| MOTA | **85.3%** +/- 8.8% | 81.6% +/- 10.3% |
| IDF1 | **87.2%** +/- 7.6% | 85.0% +/- 8.1% |

![Radar Chart](benchmarks/radar_chart.png)

> Full report (50 videos): [`benchmarks/benchmark_report.md`](benchmarks/benchmark_report.md)

---

## Quick Start

### Install

```bash
git clone https://github.com/096v/adaptive_skip_tracker.git
cd adaptive_skip_tracker
pip install .
```

### One-line replacement

```python
import cv2
# from ultralytics import YOLO
from adaptive_tracker import YOLO

model = YOLO("yolov8n.pt")

for r in model.track(
    source="your_video.mp4",
    stream=True,
    keyframe_interval=5,      # 1 YOLO + 4 optical flow
    max_features_per_bbox=12,
    alpha_cov=0.6,
    verbose=True,
):
    cv2.imshow("Tracking", r.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
```

Or use the demo script directly:

```bash
python run_demo.py your_video.mp4 yolov8n.pt
```

### Batch benchmarking

```bash
python benchmarks/run_mass_benchmarks.py --data-dir your_videos/ --max-frames 100
```

---

## Project Structure

```
adaptive_skip_tracker/
├── .gitignore
├── LICENSE
├── README.md
├── README.zh-CN.md
├── pyproject.toml
├── requirements.txt
├── run_demo.py
│
├── adaptive_tracker/              # core plugin package
│   ├── __init__.py
│   ├── main_api.py                # YOLO proxy class
│   ├── lk_estimator.py            # masked LK flow + affine estimation
│   ├── skipping_byte_tracker.py   # frame-skip BYTETracker subclass
│   └── skip_policy.py             # keyframe scheduling policy
│
└── benchmarks/
    ├── radar_chart.png
    ├── run_mass_benchmarks.py     # batch benchmark script
    └── benchmark_report.md        # detailed benchmark report
```

---

## API Parameters

Extra parameters added to `YOLO.track()`:

| Parameter | Default | Description |
|------|--------|------|
| `keyframe_interval` | `5` | Number of frames between YOLO keyframes |
| `max_features_per_bbox` | `12` | Max feature points extracted per bounding box |
| `alpha_cov` | `0.6` | Kalman covariance decay coefficient |

All other parameters (`conf`, `iou`, `device`, `classes`, `tracker`, etc.) are fully compatible with `ultralytics.YOLO.track()`.

---

## Ultralytics Ecosystem Compatibility

`__getattr__` transparently delegates all standard APIs:

```python
model.val(data="coco128.yaml")          # detection validation
model.predict("image.jpg")              # single-image inference
model.export(format="onnx")             # model export
model.names / model.device / model.task # attribute passthrough
```

---

## Known Limitations

- **Tiny objects (< 20px)**: LK optical flow may fail to find sufficient feature points, falling back to pure Kalman prediction.
- **Extreme lighting changes**: Optical flow assumes brightness consistency; rapid flashes or strobe lights may cause tracking drift.
- **Maximum skip interval**: Performance begins to degrade at interval > 10. Recommended range: **5 - 7**.

---

## References

- **ByteTrack** (Zhang et al., ECCV 2022) — two-stage high/low-confidence data association
- **BoT-SORT** (Aharon et al., ECCV 2022) — external motion compensation injected into Kalman state
- **Kalman Filter** (Kalman, 1960) — covariance control and state estimation stability under intermittent observations

---

## License

Apache License 2.0 — see [LICENSE](LICENSE)
