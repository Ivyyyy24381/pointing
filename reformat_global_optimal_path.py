#!/usr/bin/env python3
"""
Reformat global CCD comprehension CSV into the requested column layout.

Merges data from:
  - processed_subject_result_table.csv (trace3d, target distances/angles)
  - optimal_path_deviation.csv (perpendicular deviations, start position)

Output columns:
  study_name, trial_number, frame_index, percent_complete,
  trace3d_x, trace3d_y, trace3d_z,
  dist_to_target_1..4, angle_to_target_1..4,
  perp_dist_to_target_1..4, opt_path_target_1..4

Usage:
  python reformat_global_optimal_path.py [output_dir]
"""

import csv
import math
import sys
from pathlib import Path

OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path("/home/tigerli/Documents/pointing_data/point_comprehension_CCD_output")

FIELDNAMES = [
    "study_name", "trial_number", "frame_index", "percent_complete",
    "trace3d_x", "trace3d_y", "trace3d_z",
    "dist_to_target_1", "angle_to_target_1",
    "dist_to_target_2", "angle_to_target_2",
    "dist_to_target_3", "angle_to_target_3",
    "dist_to_target_4", "angle_to_target_4",
    "perp_dist_to_target_1", "opt_path_target_1",
    "perp_dist_to_target_2", "opt_path_target_2",
    "perp_dist_to_target_3", "opt_path_target_3",
    "perp_dist_to_target_4", "opt_path_target_4",
]


def load_csv(path):
    """Load CSV rows as list of dicts."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def safe_float(val, default=""):
    """Convert to float, return default on failure."""
    if val is None or val == "" or val == "nan" or val == "None":
        return default
    try:
        v = float(val)
        return "" if math.isnan(v) else v
    except (ValueError, TypeError):
        return default


def compute_opt_path_distance(start_x, start_z, target_x, target_z):
    """Straight-line distance from start position to target (optimal path length)."""
    try:
        sx, sz = float(start_x), float(start_z)
        tx, tz = float(target_x), float(target_z)
        return math.sqrt((tx - sx) ** 2 + (tz - sz) ** 2)
    except (ValueError, TypeError):
        return ""


# Fixed CCD target positions (same as in process_comprehension_optimal_path.py)
CCD_TARGETS = [
    {"label": "target_1", "x": 0.541, "y": 0.623, "z": 1.652},
    {"label": "target_2", "x": 0.246, "y": 0.510, "z": 2.353},
    {"label": "target_3", "x": -0.371, "y": 0.403, "z": 2.818},
    {"label": "target_4", "x": -1.060, "y": 0.318, "z": 3.076},
]
TARGET_POS = {t["label"]: (t["x"], t["z"]) for t in CCD_TARGETS}


def process_trial(subject_name, trial_name, trial_dir):
    """Process one trial directory, return list of output rows."""
    pose_csv = trial_dir / "processed_subject_result_table.csv"
    dev_csv = trial_dir / "optimal_path_deviation.csv"

    pose_rows = load_csv(pose_csv)
    dev_rows = load_csv(dev_csv)

    if not pose_rows:
        return []

    # Index deviation rows by frame_index for merging
    dev_by_frame = {}
    for dr in dev_rows:
        dev_by_frame[dr.get("frame_index", "")] = dr

    # Get start position from deviation CSV (constant per trial)
    start_x, start_z = None, None
    for dr in dev_rows:
        sx = safe_float(dr.get("start_x"))
        sz = safe_float(dr.get("start_z"))
        if sx != "" and sz != "":
            start_x, start_z = sx, sz
            break

    # Compute optimal path distances (start → each target)
    opt_path_dists = {}
    for n in range(1, 5):
        tkey = f"target_{n}"
        if start_x is not None and tkey in TARGET_POS:
            tx, tz = TARGET_POS[tkey]
            opt_path_dists[n] = compute_opt_path_distance(start_x, start_z, tx, tz)
        else:
            opt_path_dists[n] = ""

    n_total = len(pose_rows)
    results = []

    for i, pr in enumerate(pose_rows):
        frame_idx = pr.get("frame_index", "")
        dr = dev_by_frame.get(frame_idx, {})

        row = {
            "study_name": subject_name,
            "trial_number": trial_name,
            "frame_index": frame_idx,
            "percent_complete": round(100 * i / max(n_total - 1, 1), 2),
            "trace3d_x": safe_float(pr.get("trace3d_x")),
            "trace3d_y": safe_float(pr.get("trace3d_y")),
            "trace3d_z": safe_float(pr.get("trace3d_z")),
        }

        for n in range(1, 5):
            row[f"dist_to_target_{n}"] = safe_float(pr.get(f"target_{n}_r"))
            row[f"angle_to_target_{n}"] = safe_float(pr.get(f"target_{n}_theta"))
            row[f"perp_dist_to_target_{n}"] = safe_float(dr.get(f"opt_path_dev_target_{n}"))
            row[f"opt_path_target_{n}"] = opt_path_dists[n]

        results.append(row)

    return results


def main():
    print(f"Output dir: {OUTPUT_DIR}")

    all_rows = []
    subjects = sorted([d for d in OUTPUT_DIR.iterdir()
                        if d.is_dir() and not d.name.startswith("_")])

    for subj_dir in subjects:
        subject_name = subj_dir.name
        trial_dirs = sorted([d for d in subj_dir.iterdir()
                              if d.is_dir() and d.name.startswith("trial_")])

        for trial_dir in trial_dirs:
            trial_name = trial_dir.name
            rows = process_trial(subject_name, trial_name, trial_dir)
            all_rows.extend(rows)

    out_path = OUTPUT_DIR / "all_subjects_reformatted.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written: {out_path}")
    print(f"  {len(all_rows)} rows from {len(subjects)} subjects")


if __name__ == "__main__":
    main()
