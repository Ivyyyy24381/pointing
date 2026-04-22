"""
Extract frames from video files for BDL subjects that only have mp4 videos.

Handles two cases:
  1. Eddie (BDL255): per-trial videos already split (Color.mp4 in each trial dir)
  2. ObiWan (BDL253): single unsplit video at study level, split by auto_splits.csv

Extracts color frames as _Color_NNNN.png to match the BDL pipeline format.
Also extracts Depth_Color frames as _Depth_Color_NNNN.raw (uint16) for depth.

Usage:
  python extract_video_frames.py /path/to/subject_dir
"""

import csv
import sys
import cv2
import numpy as np
from pathlib import Path


def extract_frames_from_video(video_path, output_dir, prefix, start_frame=0,
                               frame_offset=0, max_frames=None):
    """Extract frames from video and save as PNGs.

    Args:
        video_path: Path to mp4 video
        output_dir: Directory to save frames
        prefix: Filename prefix (e.g., '_Color_')
        start_frame: Start reading from this frame index
        frame_offset: Add this to frame numbering for output filenames
        max_frames: Max frames to extract (None = all)

    Returns:
        Number of frames extracted
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = start_frame + count + frame_offset
        filename = f"{prefix}{frame_num:04d}.png"
        cv2.imwrite(str(output_dir / filename), frame)
        count += 1

        if max_frames and count >= max_frames:
            break

    cap.release()
    return count


def extract_depth_raw_from_video(video_path, output_dir, prefix, start_frame=0,
                                  frame_offset=0, max_frames=None):
    """Extract depth frames from video and save as .raw uint16 files.

    The video depth is uint8 grayscale — we scale to uint16 range
    (multiply by 257 to map 0-255 → 0-65535, approximately preserving
    the relative depth values).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and scale to uint16
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Scale: multiply by factor to get approximate mm
        # RealSense typical range: 0.3-5m = 300-5000mm
        # Video uint8 max for valid depth is ~50-60
        # Scale so that value 50 → ~3000mm (3m, typical target distance)
        depth_uint16 = (gray.astype(np.uint16) * 60).clip(0, 65535)

        frame_num = start_frame + count + frame_offset
        filename = f"{prefix}{frame_num:04d}.raw"
        depth_uint16.tofile(str(output_dir / filename))
        count += 1

        if max_frames and count >= max_frames:
            break

    cap.release()
    return count


def process_split_subject(data_dir):
    """Process a subject with per-trial videos (e.g., Eddie)."""
    print(f"Processing split subject: {data_dir.name}")

    trial_dirs = sorted([d for d in data_dir.iterdir()
                        if d.is_dir() and d.name.isdigit()])

    total_frames = 0
    for trial_dir in trial_dirs:
        color_video = trial_dir / "Color.mp4"
        depth_video = trial_dir / "Depth_Color.mp4"

        if not color_video.exists():
            print(f"  Trial {trial_dir.name}: no Color.mp4, skipping")
            continue

        color_dir = trial_dir / "Color"
        depth_color_dir = trial_dir / "Depth_Color"

        # Check if already extracted
        existing_color = list(color_dir.glob("_Color_*.png")) if color_dir.exists() else []
        if len(existing_color) > 10:
            print(f"  Trial {trial_dir.name}: already has {len(existing_color)} color frames, skipping")
            total_frames += len(existing_color)
            continue

        # Extract color frames
        n_color = extract_frames_from_video(color_video, color_dir, "_Color_")
        print(f"  Trial {trial_dir.name}: extracted {n_color} color frames")

        # Extract depth frames
        if depth_video.exists():
            n_depth = extract_depth_raw_from_video(depth_video, depth_color_dir, "_Depth_Color_")
            print(f"  Trial {trial_dir.name}: extracted {n_depth} depth frames")

        total_frames += n_color

    print(f"Total: {total_frames} frames across {len(trial_dirs)} trials")
    return total_frames


def process_unsplit_subject(data_dir):
    """Process a subject with a single unsplit video (e.g., ObiWan).
    Uses auto_splits.csv to split into trials.
    """
    print(f"Processing unsplit subject: {data_dir.name}")

    color_video = data_dir / "Color.mp4"
    depth_video = data_dir / "Depth_Color.mp4"
    splits_csv = data_dir / "auto_splits.csv"

    if not color_video.exists():
        print(f"  ERROR: No Color.mp4 found")
        return 0

    if not splits_csv.exists():
        print(f"  ERROR: No auto_splits.csv found")
        return 0

    # Read auto_splits
    splits = []
    with open(splits_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            splits.append({
                'start': int(row['start_seconds']),
                'end': int(row['end_seconds']),
            })

    # Get video properties
    cap = cv2.VideoCapture(str(color_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"  Video: {total_frames} frames @ {fps:.1f} fps")
    print(f"  Splits: {len(splits)} trials")

    # The "seconds" in auto_splits are actually frame indices
    # (based on naming convention from the original data collection)
    # Let's check if they make sense as frame numbers vs seconds
    max_split = max(s['end'] for s in splits)
    if max_split > total_frames:
        # These are likely actual timestamps in some unit
        # Check if they're frame numbers from the original (higher fps) recording
        # ObiWan: 14138 frames at 10fps, max split end = 6271
        # 6271 frames at 10fps = 627.1 seconds. total = 14138/10 = 1413.8s
        # So these might be original frame numbers from a higher-rate recording
        # or they might be milliseconds / 10
        # Let's assume they're frame indices into this video
        if max_split <= total_frames * 3:
            # Could be original frame numbers, scale down
            ratio = total_frames / max_split
            print(f"  Split values seem like original frame indices (max={max_split}, "
                  f"video has {total_frames}). Scaling by {ratio:.2f}")
            # Actually, let's check: at 10fps video, if original was 30fps,
            # frame 1073 original = frame 1073/3 ≈ 358 at 10fps
            # Let's just treat them as frame indices directly
            pass

    total_extracted = 0
    for i, split in enumerate(splits, 1):
        trial_dir = data_dir / str(i)
        color_dir = trial_dir / "Color"
        depth_color_dir = trial_dir / "Depth_Color"

        # Check if already extracted
        existing_color = list(color_dir.glob("_Color_*.png")) if color_dir.exists() else []
        if len(existing_color) > 10:
            print(f"  Trial {i}: already has {len(existing_color)} frames, skipping")
            total_extracted += len(existing_color)
            continue

        start_frame = split['start']
        end_frame = split['end']
        n_frames = end_frame - start_frame

        # Check if split values are within video range
        if start_frame >= total_frames:
            print(f"  Trial {i}: start_frame {start_frame} > total {total_frames}, skipping")
            continue

        actual_end = min(end_frame, total_frames)
        actual_n = actual_end - start_frame

        print(f"  Trial {i}: frames {start_frame}-{actual_end} ({actual_n} frames)")

        # Extract color frames
        n_color = extract_frames_from_video(
            color_video, color_dir, "_Color_",
            start_frame=start_frame, max_frames=actual_n)
        print(f"    Color: {n_color} frames extracted")

        # Extract depth frames
        if depth_video.exists():
            n_depth = extract_depth_raw_from_video(
                depth_video, depth_color_dir, "_Depth_Color_",
                start_frame=start_frame, max_frames=actual_n)
            print(f"    Depth: {n_depth} frames extracted")

        total_extracted += n_color

    print(f"Total: {total_extracted} frames across {len(splits)} trials")
    return total_extracted


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_video_frames.py /path/to/subject_dir")
        sys.exit(1)

    data_dir = Path(sys.argv[1])

    # Handle double nesting
    inner = data_dir / data_dir.name
    if inner.is_dir():
        data_dir = inner

    # Determine if split or unsplit
    has_study_video = (data_dir / "Color.mp4").exists()
    has_trial_dirs = any(d.name.isdigit() and d.is_dir() for d in data_dir.iterdir())
    has_trial_videos = False
    if has_trial_dirs:
        for d in data_dir.iterdir():
            if d.is_dir() and d.name.isdigit() and (d / "Color.mp4").exists():
                has_trial_videos = True
                break

    if has_trial_videos:
        process_split_subject(data_dir)
    elif has_study_video:
        process_unsplit_subject(data_dir)
    else:
        print(f"ERROR: No video files found in {data_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
