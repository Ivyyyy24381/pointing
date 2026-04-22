#!/usr/bin/env python3
"""
Post-process existing CCD comprehension results to apply:
1. Range filtering (z > 4 invalid, extreme x invalid)
2. Velocity filtering (max jump between frames)
3. Re-smoothing the filtered trajectory
4. Regenerate trace plots
5. Recompute optimal path with new logic

This avoids re-running SAM3 — it works on the already-saved CSV data.

Usage:
    python postprocess_trace_filter.py /path/to/output_dir [--subject CCD0346]
"""

import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from typing import List, Optional

# Filtering thresholds — z_max is relative to targets (see filter_and_smooth_trace)
VALID_Z_MIN = 0.3
VALID_X_MIN, VALID_X_MAX = -3.0, 1.5
MAX_STEP_M = 0.5  # max displacement per frame


def safe_float(val) -> float:
    if val is None or val == '' or val == 'nan':
        return float('nan')
    try:
        return float(val)
    except (ValueError, TypeError):
        return float('nan')


def filter_and_smooth_trace(xs: List[float], ys: List[float], zs: List[float],
                            smooth_window: int = 5, valid_z_max: float = 3.6):
    """
    Apply range filter, velocity filter, then smooth the remaining trajectory.
    Returns filtered+smoothed (xs, ys, zs) as lists.
    """
    n = len(xs)
    fxs = list(xs)
    fys = list(ys)
    fzs = list(zs)

    n_filtered = 0

    # Range filter
    for i in range(n):
        if np.isnan(fxs[i]):
            continue
        x, z = fxs[i], fzs[i]
        if z < VALID_Z_MIN or z > valid_z_max or x < VALID_X_MIN or x > VALID_X_MAX:
            fxs[i] = fys[i] = fzs[i] = np.nan
            n_filtered += 1

    # Velocity filter
    for i in range(1, n):
        if np.isnan(fxs[i]) or np.isnan(fxs[i-1]):
            continue
        step = np.sqrt((fxs[i] - fxs[i-1])**2 + (fzs[i] - fzs[i-1])**2)
        if step > MAX_STEP_M:
            fxs[i] = fys[i] = fzs[i] = np.nan
            n_filtered += 1

    # Interpolate gaps
    trace_3d = [[fxs[i], fys[i], fzs[i]] for i in range(n)]
    trace_3d = interpolate_trace(trace_3d)

    # Smooth
    trace_3d = smooth_trace(trace_3d, window_size=smooth_window)

    out_xs = [pt[0] for pt in trace_3d]
    out_ys = [pt[1] for pt in trace_3d]
    out_zs = [pt[2] for pt in trace_3d]

    return out_xs, out_ys, out_zs, n_filtered


def interpolate_trace(trace_3d: list, min_valid_ratio: float = 0.1) -> list:
    """Interpolate NaN gaps in 3D trace (linear)."""
    n = len(trace_3d)
    n_valid = sum(1 for pt in trace_3d if not np.isnan(pt[0]))
    if n_valid < n * min_valid_ratio:
        return trace_3d  # Too few valid points

    for dim in range(3):
        vals = [pt[dim] for pt in trace_3d]
        # Find valid indices
        valid_idx = [i for i in range(n) if not np.isnan(vals[i])]
        if len(valid_idx) < 2:
            continue
        # Only interpolate between first and last valid
        for i in range(valid_idx[0], valid_idx[-1] + 1):
            if np.isnan(vals[i]):
                # Find prev and next valid
                prev_i = max(j for j in valid_idx if j < i)
                next_i = min(j for j in valid_idx if j > i)
                frac = (i - prev_i) / (next_i - prev_i)
                vals[i] = vals[prev_i] + frac * (vals[next_i] - vals[prev_i])
        for i in range(n):
            trace_3d[i][dim] = vals[i]

    return trace_3d


def smooth_trace(trace_3d: list, window_size: int = 5) -> list:
    """Apply moving average smoothing to 3D trace."""
    if window_size < 2:
        return trace_3d
    n = len(trace_3d)
    smoothed = [list(pt) for pt in trace_3d]
    half = window_size // 2

    for dim in range(3):
        vals = [pt[dim] for pt in trace_3d]
        for i in range(n):
            if np.isnan(vals[i]):
                continue
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window_vals = [vals[j] for j in range(start, end) if not np.isnan(vals[j])]
            if window_vals:
                smoothed[i][dim] = np.mean(window_vals)

    return smoothed


def recompute_target_metrics(xs, ys, zs, targets):
    """Recompute target distances from filtered trace."""
    metrics = []
    for i in range(len(xs)):
        if np.isnan(xs[i]):
            metrics.append({})
            continue
        m = {}
        for t in targets:
            label = t["label"]
            dx = t["x"] - xs[i]
            dy = t["y"] - ys[i]
            dz = t["z"] - zs[i]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            theta = np.arccos(dz / r) if r > 0 else 0.0
            phi = np.arctan2(dy, dx)
            m[f"{label}_r"] = r
            m[f"{label}_theta"] = theta
            m[f"{label}_phi"] = phi
        metrics.append(m)
    return metrics


def regenerate_trace_plot(xs, zs, targets, fps, output_path: Path, title: str = ""):
    """Regenerate 2D trace plot (X-Z top-down)."""
    valid_idx = [i for i in range(len(xs)) if not np.isnan(xs[i])]
    if len(valid_idx) < 2:
        return

    vx = [xs[i] for i in valid_idx]
    vz = [zs[i] for i in valid_idx]
    times = [i / fps for i in valid_idx] if fps > 0 else list(range(len(valid_idx)))

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(vx, vz, c=times, cmap="viridis", s=8)
    ax.plot(vx, vz, "gray", alpha=0.3, linewidth=0.5)

    for t in targets:
        ax.scatter(t["x"], t["z"], c="red", s=80, marker="*", zorder=5)
        ax.text(t["x"], t["z"], f"  {t['label']}", fontsize=8)

    if vx:
        ax.scatter(vx[0], vz[0], c="green", s=60, marker="^", zorder=5, label="Start")
        ax.scatter(vx[-1], vz[-1], c="blue", s=60, marker="v", zorder=5, label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z / Depth (m)")
    ax.set_title(title or "Baby Trace (filtered)")
    ax.set_aspect("equal")
    ax.legend()
    plt.colorbar(sc, label="Time (s)")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=100)
    plt.close()


def process_trial_csv(csv_path: Path, targets: list, fps: float = 5.0) -> bool:
    """
    Post-process a single trial's CSV: filter, smooth, rewrite CSV, regenerate plots.
    Returns True if successful.
    """
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"    Error reading {csv_path}: {e}")
        return False

    if not rows:
        return False

    # Check if this has trace data
    has_trace = any(row.get('trace3d_x', '') != '' for row in rows)
    if not has_trace:
        return False

    # Extract trace
    xs = [safe_float(r.get('trace3d_x')) for r in rows]
    ys = [safe_float(r.get('trace3d_y')) for r in rows]
    zs = [safe_float(r.get('trace3d_z')) for r in rows]

    n_valid_before = sum(1 for x in xs if not np.isnan(x))
    if n_valid_before < 3:
        return False

    # Filter and smooth (z max = farthest target + 0.5m margin)
    max_target_z = max(t['z'] for t in targets) if targets else 3.1
    fxs, fys, fzs, n_filtered = filter_and_smooth_trace(xs, ys, zs,
                                                          valid_z_max=max_target_z + 0.5)
    n_valid_after = sum(1 for x in fxs if not np.isnan(x))

    if n_filtered > 0:
        print(f"    Filtered {n_filtered} outlier points: {n_valid_before} → {n_valid_after} valid")

    # Recompute target metrics
    metrics = recompute_target_metrics(fxs, fys, fzs, targets)

    # Build updated fieldnames
    fieldnames = list(rows[0].keys())

    # Rewrite CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            # Update trace values
            row['trace3d_x'] = round(fxs[i], 6) if not np.isnan(fxs[i]) else ''
            row['trace3d_y'] = round(fys[i], 6) if not np.isnan(fys[i]) else ''
            row['trace3d_z'] = round(fzs[i], 6) if not np.isnan(fzs[i]) else ''
            # Update target metrics
            if metrics[i]:
                for k, v in metrics[i].items():
                    if k in fieldnames:
                        row[k] = round(v, 6)
            else:
                for k in fieldnames:
                    if '_r' in k or '_theta' in k or '_phi' in k:
                        if k.startswith('target_'):
                            row[k] = ''
            writer.writerow(row)

    # Determine fps from CSV
    if len(rows) >= 2:
        t0 = safe_float(rows[0].get('time_sec', 0))
        t1 = safe_float(rows[1].get('time_sec', 0))
        if not np.isnan(t0) and not np.isnan(t1) and t1 > t0:
            fps = 1.0 / (t1 - t0)

    # Regenerate trace plot
    trial_dir = csv_path.parent
    trial_name = trial_dir.name
    subject_name = trial_dir.parent.name
    regenerate_trace_plot(fxs, fzs, targets, fps,
                         trial_dir / "processed_subject_result_trace.png",
                         title=f"{subject_name} / {trial_name}")

    return True


def process_subject(subject_dir: Path, targets: list):
    """Post-process all trials for a subject."""
    trial_dirs = sorted([d for d in subject_dir.iterdir()
                        if d.is_dir() and d.name.startswith('trial_')])
    ok = 0
    for trial_dir in trial_dirs:
        csv_path = trial_dir / "processed_subject_result_table.csv"
        if csv_path.exists():
            if process_trial_csv(csv_path, targets):
                ok += 1
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Post-process CCD comprehension results: filter + smooth + replot")
    parser.add_argument("output_dir",
                        help="Output directory with extracted results")
    parser.add_argument("--subject", help="Process only this subject")
    parser.add_argument("--targets-json", help="Custom targets JSON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Default CCD targets
    targets = [
        {"label": "target_1", "x":  0.4430, "y": 0.6570, "z": 1.6520},
        {"label": "target_2", "x":  0.1560, "y": 0.5440, "z": 2.3530},
        {"label": "target_3", "x": -0.4620, "y": 0.4370, "z": 2.8180},
        {"label": "target_4", "x": -1.1500, "y": 0.3520, "z": 3.0760},
    ]
    if args.targets_json:
        with open(args.targets_json) as f:
            targets = json.load(f)

    subjects = sorted([d for d in output_dir.iterdir()
                       if d.is_dir() and not d.name.startswith('_')])
    if args.subject:
        subjects = [d for d in subjects if d.name.startswith(args.subject)]

    total_ok = 0
    for i, subject_dir in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] {subject_dir.name}")
        ok = process_subject(subject_dir, targets)
        total_ok += ok
        print(f"  → {ok} trials updated")

    print(f"\nDone: {total_ok} trials post-processed")
    print("Now run process_comprehension_optimal_path.py to regenerate optimal paths")


if __name__ == "__main__":
    main()
