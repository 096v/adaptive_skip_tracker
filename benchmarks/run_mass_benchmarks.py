"""Batch benchmark script — MOT17 / MOT20 / custom datasets.

Recursively finds all videos in a directory, runs them through
  - interval=1  (full YOLO baseline)
  - interval=5  (adaptive skip)
  - interval=10

Outputs a CSV summary report + per-video JSON details.
"""

import os, sys, time, json, logging, csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from adaptive_tracker import YOLO

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)


def find_videos(data_dir: str) -> list[str]:
    """Recursively find all video files in a directory."""
    exts = (".mp4", ".avi", ".mkv", ".mov", ".webm")
    videos = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(exts):
                videos.append(os.path.join(root, f))
    return sorted(videos)


def run_one(video_path: str, interval: int, model: YOLO, max_frames: int = 0) -> dict:
    """Run tracking on a single video and return performance stats."""
    frame_count = 0
    total_dets = 0
    tracked_counts = []
    t0 = time.perf_counter()

    for result in model.track(
        source=video_path,
        stream=True,
        conf=0.25,
        keyframe_interval=interval,
        max_features_per_bbox=12,
        verbose=False,
    ):
        boxes = result.boxes
        n = len(boxes) if boxes is not None else 0
        total_dets += n
        tracked_counts.append(n)
        frame_count += 1
        _ = result.plot()
        if max_frames > 0 and frame_count >= max_frames:
            break

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000
    avg_ms = elapsed_ms / frame_count if frame_count > 0 else 0

    return {
        "video": os.path.basename(video_path),
        "path": video_path,
        "frames": frame_count,
        "total_ms": round(elapsed_ms, 1),
        "avg_ms": round(avg_ms, 2),
        "fps": round(frame_count / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else 0,
        "avg_tracked": round(np.mean(tracked_counts), 1) if tracked_counts else 0,
        "total_dets": total_dets,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch benchmark — MOT17 / MOT20 / custom")
    parser.add_argument("--data-dir", type=str, default="datasets/MOT17",
                        help="Dataset root directory")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Model path")
    parser.add_argument("--max-videos", type=int, default=0,
                        help="Max videos to test (0=all)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per video (0=all)")
    parser.add_argument("--intervals", type=str, default="1,5,10",
                        help="Intervals to test (comma-separated)")
    parser.add_argument("--output", type=str, default="benchmarks_summary.csv",
                        help="CSV output path")
    args = parser.parse_args()

    videos = find_videos(args.data_dir)
    if not videos:
        print(f"No videos found in: {args.data_dir}")
        for alt in ["datasets/MOT17", "MOT17", "../datasets/MOT17"]:
            videos = find_videos(alt)
            if videos:
                break
    if not videos:
        print("Place MOT17 or test videos in a datasets/ directory.")
        return

    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    intervals = [int(x) for x in args.intervals.split(",")]
    print("=" * 65)
    print(f"  Batch benchmark: {len(videos)} videos x {len(intervals)} intervals")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Model: {args.model}")
    print("=" * 65)

    model = YOLO(args.model, verbose=False)
    all_results = {}

    for interval in intervals:
        label = "full YOLO baseline" if interval == 1 else f"skip interval={interval}"
        print(f"\n--- interval={interval} ({label}) ---")
        video_rows = []

        for vi, vp in enumerate(videos):
            vname = os.path.basename(vp)
            try:
                r = run_one(vp, interval, model, args.max_frames)
                video_rows.append(r)
                print(f"  [{vi+1:3d}/{len(videos)}] {vname}: "
                      f"{r['frames']}f, {r['avg_ms']:.1f}ms/f, "
                      f"tracked={r['avg_tracked']:.1f}")
            except Exception as e:
                print(f"  [{vi+1:3d}/{len(videos)}] {vname}: ERROR - {e}")
                video_rows.append({"video": vname, "error": str(e)})

        all_results[str(interval)] = video_rows

        valid = [v for v in video_rows if "error" not in v and v["frames"] > 0]
        if valid:
            avgs = [v["avg_ms"] for v in valid]
            trks = [v["avg_tracked"] for v in valid]
            print(f"  => avg: {np.mean(avgs):.1f}ms/f, tracked={np.mean(trks):.1f}")

    # Build CSV
    csv_headers = [
        "Video", "Frames", "Base_ms", "Base_Tracked",
        "Skip5_ms", "Skip5_Tracked", "Skip5_Speedup",
        "Skip10_ms", "Skip10_Tracked", "Skip10_Speedup",
    ]
    rows = []
    for vi, vp in enumerate(videos):
        vname = os.path.basename(vp)
        r1 = all_results.get("1", [{}])[vi] if vi < len(all_results.get("1", [])) else {}
        r5 = all_results.get("5", [{}])[vi] if vi < len(all_results.get("5", [])) else {}
        r10 = all_results.get("10", [{}])[vi] if vi < len(all_results.get("10", [])) else {}

        base_ms = r1.get("avg_ms", 0)
        skip_ms = r5.get("avg_ms", 0)
        speedup5 = round(base_ms / skip_ms, 2) if base_ms > 0 and skip_ms > 0 else 0
        speedup10 = round(base_ms / r10.get("avg_ms", 1), 2) if base_ms > 0 and r10.get("avg_ms", 0) > 0 else 0

        rows.append([
            vname,
            r1.get("frames", 0),
            base_ms,
            r1.get("avg_tracked", 0),
            skip_ms,
            r5.get("avg_tracked", 0),
            f"{speedup5}x",
            r10.get("avg_ms", 0),
            r10.get("avg_tracked", 0),
            f"{speedup10}x",
        ])

    # Write CSV via standard library
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(rows)

        # Summary row
        valid_rows = [r for r in rows if r[2] > 0]
        if valid_rows:
            total_frames = sum(r[1] for r in valid_rows)
            avg_base = np.mean([r[2] for r in valid_rows])
            avg_skip5 = np.mean([r[4] for r in valid_rows])
            avg_skip10 = np.mean([r[7] for r in valid_rows])
            avg_trk_base = np.mean([r[3] for r in valid_rows])
            avg_trk5 = np.mean([r[5] for r in valid_rows])
            avg_trk10 = np.mean([r[8] for r in valid_rows])

            writer.writerow([
                f"** SUMMARY ({len(valid_rows)} videos) **",
                total_frames,
                round(avg_base, 1),
                round(avg_trk_base, 1),
                round(avg_skip5, 1),
                round(avg_trk5, 1),
                f"{round(avg_base/avg_skip5, 1)}x" if avg_skip5 else "",
                round(avg_skip10, 1),
                round(avg_trk10, 1),
                f"{round(avg_base/avg_skip10, 1)}x" if avg_skip10 else "",
            ])

    print(f"\n{'='*65}")
    print(f"  Report saved: {args.output}")
    print(f"  Videos: {len(valid_rows)}, Total frames: {total_frames}")
    print(f"  Baseline: {avg_base:.1f}ms/f | Skip5: {avg_skip5:.1f}ms/f ({avg_base/avg_skip5:.1f}x)")
    print(f"  Track retention: {avg_trk_base:.1f} -> {avg_trk5:.1f} ({(avg_trk5/avg_trk_base*100 if avg_trk_base else 0):.1f}%)")
    print(f"{'='*65}")

    # Also save JSON
    json_path = args.output.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Details: {json_path}")


if __name__ == "__main__":
    main()
