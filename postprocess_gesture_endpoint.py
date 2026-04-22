"""
Post-process production gesture CSVs to filter out frames after the pointing
gesture has completed.

For each trial:
  1. Determine the intended target (the one with the smallest minimum distance)
  2. Find the frame where the distance to that target is minimized (gesture peak)
  3. After the peak, find where the distance starts consistently increasing
     (the pointer is moving away from the target)
  4. Trim frames after that endpoint

This keeps only the approach + at-target phases of each pointing gesture.

Usage:
  python postprocess_gesture_endpoint.py [output_dir]
  python postprocess_gesture_endpoint.py [output_dir] --departure-threshold 0.3
"""

import csv
import json
import sys
import numpy as np
from pathlib import Path

# How far distance can increase above the minimum before we consider
# the gesture "done" (meters)
DEFAULT_DEPARTURE_THRESHOLD = 0.3

# Minimum distance to a target to consider the pointer "at" the target
AT_TARGET_THRESHOLD = 0.5

# Smoothing window for distance signal
SMOOTH_WINDOW = 5

# Which vector type to use for distance computation
VECTOR_TYPE = "eye_to_wrist"


def smooth_signal(values, window=SMOOTH_WINDOW):
    """Simple moving average smoothing, handling NaN."""
    result = np.full_like(values, np.nan)
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        valid = [v for v in values[start:end] if not np.isnan(v)]
        if valid:
            result[i] = np.mean(valid)
    return result


def find_gesture_endpoint(distances, departure_threshold=DEFAULT_DEPARTURE_THRESHOLD):
    """
    Find the frame index where the pointing gesture ends.

    Algorithm:
      1. Find the global minimum distance (gesture peak)
      2. After the peak, find the first frame where smoothed distance exceeds
         min_distance + departure_threshold consistently
      3. Return that frame index as the endpoint

    Returns:
        endpoint_idx: last frame index to keep (inclusive), or len-1 if no clear end
        target_min_dist: minimum distance achieved
        peak_idx: index of minimum distance
    """
    if len(distances) == 0:
        return 0, np.nan, 0

    # Smooth the distance signal
    smoothed = smooth_signal(distances)

    # Find valid (non-NaN) entries
    valid_mask = ~np.isnan(smoothed)
    if not np.any(valid_mask):
        return len(distances) - 1, np.nan, 0

    # Find global minimum
    valid_indices = np.where(valid_mask)[0]
    min_idx = valid_indices[np.argmin(smoothed[valid_mask])]
    min_dist = smoothed[min_idx]

    # After the minimum, find where distance exceeds threshold
    departure_level = min_dist + departure_threshold

    # Also require the minimum is reasonably close to a target
    if min_dist > AT_TARGET_THRESHOLD:
        # Pointer never got close to any target — keep all data
        return len(distances) - 1, min_dist, min_idx

    endpoint_idx = len(distances) - 1
    consecutive_above = 0
    required_consecutive = 3  # Need 3 consecutive frames above threshold

    for i in range(min_idx + 1, len(distances)):
        if np.isnan(smoothed[i]):
            continue
        if smoothed[i] > departure_level:
            consecutive_above += 1
            if consecutive_above >= required_consecutive:
                # The endpoint is where it first crossed the threshold
                endpoint_idx = i - required_consecutive + 1
                break
        else:
            consecutive_above = 0

    return endpoint_idx, min_dist, min_idx


def process_trial(trial_dir, departure_threshold=DEFAULT_DEPARTURE_THRESHOLD):
    """Process one trial CSV: find gesture endpoint and trim."""
    csv_path = trial_dir / "processed_gesture.csv"
    if not csv_path.exists():
        return None

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        return None

    # Compute distance to each target per frame
    n_targets = 4
    target_distances = {t: [] for t in range(1, n_targets + 1)}

    for row in rows:
        for t in range(1, n_targets + 1):
            col = f'{VECTOR_TYPE}_dist_to_target_{t}'
            val = row.get(col, '')
            try:
                target_distances[t].append(float(val) if val and val != 'None' else np.nan)
            except ValueError:
                target_distances[t].append(np.nan)

    # Convert to numpy arrays
    for t in range(1, n_targets + 1):
        target_distances[t] = np.array(target_distances[t])

    # Determine intended target (smallest minimum distance)
    min_per_target = {}
    for t in range(1, n_targets + 1):
        valid = target_distances[t][~np.isnan(target_distances[t])]
        min_per_target[t] = np.min(valid) if len(valid) > 0 else np.inf

    intended_target = min(min_per_target, key=min_per_target.get)
    distances = target_distances[intended_target]

    # Find gesture endpoint
    endpoint_idx, min_dist, peak_idx = find_gesture_endpoint(
        distances, departure_threshold
    )

    original_count = len(rows)
    trimmed_rows = rows[:endpoint_idx + 1]

    # Write back
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trimmed_rows)

    # Save metadata
    meta = {
        "intended_target": f"target_{intended_target}",
        "min_distance": float(min_dist) if not np.isnan(min_dist) else None,
        "peak_frame_idx": int(peak_idx),
        "endpoint_frame_idx": int(endpoint_idx),
        "original_frames": original_count,
        "trimmed_frames": len(trimmed_rows),
        "frames_removed": original_count - len(trimmed_rows),
        "departure_threshold": departure_threshold,
        "vector_type": VECTOR_TYPE,
    }
    meta_path = trial_dir / "gesture_endpoint_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def process_subject(subj_dir, departure_threshold=DEFAULT_DEPARTURE_THRESHOLD):
    """Process all trials for one subject."""
    trial_dirs = sorted([d for d in subj_dir.iterdir()
                        if d.is_dir() and d.name.startswith('trial_')])

    ok = 0
    total_removed = 0
    total_original = 0

    for td in trial_dirs:
        meta = process_trial(td, departure_threshold)
        if meta:
            removed = meta['frames_removed']
            total_removed += removed
            total_original += meta['original_frames']
            pct = 100 * removed / meta['original_frames'] if meta['original_frames'] > 0 else 0
            min_d = meta['min_distance']
            min_d_str = f"{min_d:.3f}" if min_d is not None else "N/A"
            status = f"target={meta['intended_target']}, min_dist={min_d_str}, " \
                     f"kept {meta['trimmed_frames']}/{meta['original_frames']} ({pct:.0f}% removed)"
            print(f"    {td.name}: {status}")
            ok += 1
        else:
            print(f"    {td.name}: no data")

    pct_total = 100 * total_removed / total_original if total_original > 0 else 0
    print(f"  Subject total: {total_removed}/{total_original} frames removed ({pct_total:.0f}%)")
    return ok


def regenerate_combined_csv(subj_dir):
    """Regenerate combined_gesture.csv after filtering."""
    trial_dirs = sorted([d for d in subj_dir.iterdir()
                        if d.is_dir() and d.name.startswith('trial_')])

    all_rows = []
    fieldnames = None
    subject_name = subj_dir.name

    for td in trial_dirs:
        csv_path = td / "processed_gesture.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                row['subject'] = subject_name
                row['trial'] = td.name
                all_rows.append(row)

    if not all_rows or fieldnames is None:
        return 0

    combined_fields = ['subject', 'trial'] + fieldnames
    combined_path = subj_dir / "combined_gesture.csv"
    with open(combined_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=combined_fields,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    return len(all_rows)


def generate_global_csv(output_dir):
    """Generate all_subjects_gesture.csv combining all subjects."""
    all_rows = []
    fieldnames = None

    for subj_dir in sorted(output_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith('_'):
            continue
        combined = subj_dir / "combined_gesture.csv"
        if not combined.exists():
            continue
        with open(combined, newline='') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                all_rows.append(row)

    if not all_rows or fieldnames is None:
        return 0

    global_path = output_dir / "all_subjects_gesture.csv"
    with open(global_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Global gesture CSV: {len(all_rows)} rows")
    return len(all_rows)


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("/home/tigerli/Documents/pointing_data/point_production_BDL_output")

    departure = DEFAULT_DEPARTURE_THRESHOLD
    if '--departure-threshold' in sys.argv:
        idx = sys.argv.index('--departure-threshold')
        departure = float(sys.argv[idx + 1])

    print(f"Output: {output_dir}")
    print(f"Departure threshold: {departure}m")
    print(f"Vector type: {VECTOR_TYPE}")
    print()

    total_ok = 0
    for subj_dir in sorted(output_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith('_'):
            continue
        print(f"{subj_dir.name}")
        ok = process_subject(subj_dir, departure)
        if ok > 0:
            n = regenerate_combined_csv(subj_dir)
            print(f"  Combined: {n} rows")
        total_ok += ok

    print(f"\nTotal: {total_ok} trials filtered")

    # Regenerate global CSV
    generate_global_csv(output_dir)


if __name__ == "__main__":
    main()
