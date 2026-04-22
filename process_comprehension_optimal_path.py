#!/usr/bin/env python3
"""
Optimal Path Deviation for CCD Pointing Comprehension (baby data).

For each subject/trial:
  1. Read baby trace (trace3d_x/z) from old pipeline CSV
  2. Use fixed target positions (camera coordinates)
  3. Detect first-choice target (first one baby approaches within threshold)
  4. Compute perpendicular deviation from optimal straight-line path to ALL targets
  5. Filter to approach phase only (start → first reaching chosen target)
  6. Save per-trial optimal_path_deviation.csv + QC plots
  7. Save per-subject combined CSVs

Usage:
    python process_comprehension_optimal_path.py \
        /home/tigerli/Documents/pointing_data/point_comprehension_CCD_output/
"""

import argparse
import csv
import io
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict


# ─── Fixed target positions for CCD comprehension (camera coordinates) ───
# Recovered from old pipeline CSV spherical coordinates (target_N_r/theta/phi)
# using compute_target_distances convention: theta=arccos(dz/r), phi=arctan2(dy,dx)
# target_1 = closest to camera, target_4 = furthest from camera
CCD_TARGETS = [
    {"label": "target_1", "x":  0.4430, "y": 0.6570, "z": 1.6520},
    {"label": "target_2", "x":  0.1560, "y": 0.5440, "z": 2.3530},
    {"label": "target_3", "x": -0.4620, "y": 0.4370, "z": 2.8180},
    {"label": "target_4", "x": -1.1500, "y": 0.3520, "z": 3.0760},
]

ROI_RADIUS = 0.3  # Region of interest radius around target (m)

# Default ground truth CSV path
DEFAULT_GROUND_TRUTH_CSV = Path("/home/tigerli/Documents/pointing_data/PVPT_Comprehension_Data - Long Data.csv")


def load_ground_truth(csv_path: Path) -> Dict[Tuple[str, int], int]:
    """Load ground truth first-choice from Long Data CSV.

    Returns dict of {(participant_id, trial_number): choice_1}.
    Skips excluded trials and trials with NA/empty choice_1.
    Trial numbers are 1-indexed as in the CSV.
    """
    if not csv_path.exists():
        print(f"  WARNING: ground truth file not found: {csv_path}")
        return {}
    gt = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("excluded", "").strip().upper() == "Y":
                continue
            pid = row.get("participant_id", "").strip()
            c1 = row.get("choice_1", "").strip()
            if not pid or c1 in ("", "NA"):
                continue
            try:
                trial = int(row["trial_number"])
                choice = int(c1)
            except (ValueError, KeyError):
                continue
            gt[(pid, trial)] = choice
    return gt


def extract_ccd_id(subject_name: str) -> Optional[str]:
    """Extract CCD ID from subject directory name (e.g. 'CCD0194_PVPT_008E_side' -> 'CCD0194')."""
    import re
    m = re.match(r'(CCD\d+)', subject_name)
    return m.group(1) if m else None


def safe_float(val) -> float:
    if val is None or val == '' or val == 'nan':
        return float('nan')
    try:
        return float(val)
    except (ValueError, TypeError):
        return float('nan')


def point_to_line_distance(px, pz, ax, az, bx, bz):
    """Perpendicular distance from point P to line A->B in 2D (X-Z plane)."""
    abx = bx - ax
    abz = bz - az
    ab_len = np.sqrt(abx**2 + abz**2)
    if ab_len < 1e-9:
        d = np.sqrt((px - ax)**2 + (pz - az)**2)
        return d, d
    apx = px - ax
    apz = pz - az
    cross = abx * apz - abz * apx
    signed_dist = cross / ab_len
    return abs(signed_dist), signed_dist


def load_baby_positions(csv_rows: List[dict]) -> List[dict]:
    """Extract baby positions from CSV rows."""
    positions = []
    for row in csv_rows:
        x = safe_float(row.get('trace3d_x'))
        z = safe_float(row.get('trace3d_z'))
        positions.append({
            'frame': int(float(row.get('frame_index', 0))),
            'time_sec': safe_float(row.get('time_sec', 0)),
            'x': x,
            'z': z,
        })
    return positions


def detect_first_choice_from_r(csv_rows: List[dict],
                                threshold: float = 1.5) -> Optional[str]:
    """
    Detect first-choice target using target_N_r columns (distance from baby).
    Returns target label (e.g., 'target_1') or None.
    """
    for row in csv_rows:
        for tn in range(1, 5):
            r = safe_float(row.get(f'target_{tn}_r'))
            if not np.isnan(r) and r < threshold:
                return f'target_{tn}'

    # Fallback: find target with minimum distance at any point
    min_r = float('inf')
    best_target = None
    for row in csv_rows:
        for tn in range(1, 5):
            r = safe_float(row.get(f'target_{tn}_r'))
            if not np.isnan(r) and r < min_r:
                min_r = r
                best_target = f'target_{tn}'
    return best_target


def detect_first_choice(positions: List[dict], targets: List[dict],
                        threshold: float = 0.5) -> Optional[dict]:
    """Detect which target baby approaches first from trace3d positions."""
    min_dist_overall = float('inf')
    first_choice = None

    for pos in positions:
        dx, dz = pos['x'], pos['z']
        if np.isnan(dx) or np.isnan(dz):
            continue
        for t in targets:
            tx, tz = t['x'], t['z']
            dist = np.sqrt((dx - tx)**2 + (dz - tz)**2)
            if dist < threshold:
                return t
            if dist < min_dist_overall:
                min_dist_overall = dist
                first_choice = t
    return first_choice


def detect_first_choice_from_initial_heading(positions: List[dict],
                                              targets: List[dict],
                                              start_x: float, start_z: float,
                                              fraction: float = 0.2) -> Optional[dict]:
    """
    Detect first-choice target by analyzing the first fraction (default 20%) of
    valid 3D data. The target whose optimal path (start→target) has the minimum
    average deviation in this initial segment is chosen.
    """
    valid = [(p['x'], p['z']) for p in positions
             if not np.isnan(p['x']) and not np.isnan(p['z'])]
    if len(valid) < 3:
        return None

    n_initial = max(3, int(len(valid) * fraction))
    initial_pts = valid[:n_initial]

    best_target = None
    best_avg_dev = float('inf')

    for t in targets:
        tx, tz = t['x'], t['z']
        devs = []
        for px, pz in initial_pts:
            unsigned_dev, _ = point_to_line_distance(px, pz, start_x, start_z, tx, tz)
            devs.append(unsigned_dev)
        avg_dev = np.mean(devs)
        if avg_dev < best_avg_dev:
            best_avg_dev = avg_dev
            best_target = t

    return best_target


def smooth_trace(xs: List[float], zs: List[float],
                 window: int = 11) -> Tuple[List[float], List[float]]:
    """Smooth trace using Savitzky-Golay filter (or moving average fallback).
    Returns smoothed (xs, zs). NaNs are preserved."""
    xs_arr = np.array(xs, dtype=float)
    zs_arr = np.array(zs, dtype=float)
    valid = ~(np.isnan(xs_arr) | np.isnan(zs_arr))
    n_valid = valid.sum()

    if n_valid < window:
        # Not enough points for smoothing, return as-is
        return list(xs_arr), list(zs_arr)

    try:
        from scipy.signal import savgol_filter
        # Extract valid-only arrays, smooth, put back
        win = min(window, n_valid)
        if win % 2 == 0:
            win -= 1
        win = max(3, win)
        order = min(2, win - 1)
        sx = xs_arr.copy()
        sz = zs_arr.copy()
        sx[valid] = savgol_filter(xs_arr[valid], win, order)
        sz[valid] = savgol_filter(zs_arr[valid], win, order)
        return list(sx), list(sz)
    except ImportError:
        # Fallback: simple moving average
        kernel = np.ones(window) / window
        sx = xs_arr.copy()
        sz = zs_arr.copy()
        if n_valid >= window:
            sx[valid] = np.convolve(xs_arr[valid], kernel, mode='same')
            sz[valid] = np.convolve(zs_arr[valid], kernel, mode='same')
        return list(sx), list(sz)


def compute_start_position(positions: List[dict], targets: List[dict]) -> Optional[Tuple[float, float]]:
    """
    Return the first valid position as start.
    Returns (sx, sz) or None if no valid data.
    """
    for pos in positions:
        sx, sz = pos['x'], pos['z']
        if not np.isnan(sx) and not np.isnan(sz):
            return (float(sx), float(sz))
    return None


def compute_full_deviation(positions: List[dict], targets: List[dict],
                           chosen_target: dict) -> List[dict]:
    """
    Compute per-frame deviation from optimal path to ALL targets.
    Saves deviation metrics for all targets at every frame.
    Also computes approach-phase metrics: once baby reaches the chosen target
    (minimum distance) and starts leaving, the approach_phase flag is set to False.
    """
    start = compute_start_position(positions, targets)
    if start is None:
        return []
    sx, sz = start

    results = []
    cumulative_dist = 0.0
    prev_x, prev_z = None, None

    chosen_tn = chosen_target['label'].replace('target_', '')

    # Pass 1: compute all deviations/distances (no approach_phase yet)
    for i, pos in enumerate(positions):
        dx, dz = pos['x'], pos['z']

        row = {
            'frame_index': pos['frame'],
            'time_sec': pos['time_sec'],
            'baby_x': dx,
            'baby_z': dz,
            'start_x': sx,
            'start_z': sz,
            'first_choice': chosen_target['label'],
        }

        if np.isnan(dx) or np.isnan(dz):
            row['cumulative_distance_traveled'] = float('nan')
            row['approach_phase'] = ''
            for t in targets:
                tn = t['label'].replace('target_', '')
                row[f'dist_to_target_{tn}'] = float('nan')
                row[f'opt_path_dev_target_{tn}'] = float('nan')
                row[f'signed_dev_target_{tn}'] = float('nan')
            results.append(row)
            continue

        # Cumulative distance
        if prev_x is not None and not np.isnan(prev_x):
            step = np.sqrt((dx - prev_x)**2 + (dz - prev_z)**2)
            cumulative_dist += step
        prev_x, prev_z = dx, dz
        row['cumulative_distance_traveled'] = cumulative_dist

        # Deviation to each target
        for t in targets:
            tn = t['label'].replace('target_', '')
            tx, tz = t['x'], t['z']

            unsigned_dev, signed_dev = point_to_line_distance(dx, dz, sx, sz, tx, tz)
            dist_to_t = np.sqrt((dx - tx)**2 + (dz - tz)**2)

            row[f'dist_to_target_{tn}'] = dist_to_t
            row[f'opt_path_dev_target_{tn}'] = unsigned_dev
            row[f'signed_dev_target_{tn}'] = signed_dev

        results.append(row)

    # Pass 2: determine approach cutoff index
    # Primary: ROI entry (dist <= ROI_RADIUS). Fallback: min distance frame.
    dist_key = f'dist_to_target_{chosen_tn}'
    roi_idx = None
    min_dist = float('inf')
    min_idx = 0
    for i, r in enumerate(results):
        if r.get('approach_phase') == '':
            continue  # skip NaN rows
        d = r.get(dist_key, float('inf'))
        if not np.isnan(d):
            if roi_idx is None and d <= ROI_RADIUS:
                roi_idx = i
            if d < min_dist:
                min_dist = d
                min_idx = i

    cutoff = roi_idx if roi_idx is not None else min_idx
    for i, r in enumerate(results):
        if r.get('approach_phase') == '':
            continue  # keep NaN rows as ''
        r['approach_phase'] = 'approach' if i <= cutoff else 'arrived'

    return results


def save_deviation_csv(results: List[dict], output_path: Path, targets: List[dict]):
    """Save deviation results to CSV."""
    if not results:
        return

    fieldnames = ['frame_index', 'time_sec', 'baby_x', 'baby_z',
                  'start_x', 'start_z', 'first_choice',
                  'cumulative_distance_traveled', 'approach_phase']
    for t in targets:
        tn = t['label'].replace('target_', '')
        fieldnames.extend([
            f'dist_to_target_{tn}',
            f'opt_path_dev_target_{tn}',
            f'signed_dev_target_{tn}',
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# ─── Visualization ───

def plot_trace_with_targets(positions: List[dict], targets: List[dict],
                            chosen_target: dict, results: List[dict],
                            output_path: Path, title: str = ""):
    """Plot baby trace in X-Z plane with targets and optimal path line.
    Approach phase: rainbow dots. Arrived phase: gray dots."""
    if not results:
        return

    valid_r = [r for r in results
               if not np.isnan(r.get('baby_x', float('nan')))
               and not np.isnan(r.get('baby_z', float('nan')))]
    if len(valid_r) < 2:
        return

    xs = [r['baby_x'] for r in valid_r]
    zs = [r['baby_z'] for r in valid_r]
    phases = [r.get('approach_phase', '') for r in valid_r]

    app_xs = [x for x, p in zip(xs, phases) if p == 'approach']
    app_zs = [z for z, p in zip(zs, phases) if p == 'approach']
    leave_xs = [x for x, p in zip(xs, phases) if p != 'approach']
    leave_zs = [z for z, p in zip(zs, phases) if p != 'approach']

    fig, ax = plt.subplots(figsize=(7, 6))

    # Optimal path line (start -> chosen target)
    sx, sz = valid_r[0]['start_x'], valid_r[0]['start_z']
    tx, tz = chosen_target['x'], chosen_target['z']
    ax.plot([sx, tx], [sz, tz], '--', color='#2196F3', linewidth=2.0, alpha=0.8,
            label='Optimal path', zorder=2)

    # Arrived / post-ROI phase (gray dots)
    if leave_xs:
        ax.scatter(leave_xs, leave_zs, c='lightgray', s=15, alpha=0.5, zorder=2,
                   label=f'Arrived ({len(leave_xs)} pts)')

    # Approach phase (rainbow dots)
    if app_xs:
        colors = cm.rainbow(np.linspace(0, 1, len(app_xs)))
        ax.scatter(app_xs, app_zs, c=colors, s=30, alpha=0.9, zorder=3,
                   label=f'Approach ({len(app_xs)} pts)')

    # Start / end markers
    ax.scatter(xs[0], zs[0], c='blue', s=100, marker='s', zorder=5, label='Start')
    ax.scatter(xs[-1], zs[-1], c='red', s=100, marker='*', zorder=5, label='End')

    # ROI entry marker (transition point)
    if app_xs and leave_xs:
        ax.scatter(app_xs[-1], app_zs[-1], c='orange', s=80, marker='v',
                   edgecolors='black', linewidths=0.5, zorder=5,
                   label='ROI entry')

    # All targets
    for t in targets:
        marker = 'D' if t['label'] == chosen_target['label'] else 'o'
        color = 'red' if t['label'] == chosen_target['label'] else 'gray'
        ax.scatter(t['x'], t['z'], c=color, s=120, marker=marker,
                   edgecolors='black', linewidths=1, zorder=4)
        ax.annotate(t['label'].replace('target_', 'T'), (t['x'], t['z']),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)

    # Approach-phase stats
    chosen_tn = chosen_target['label'].replace('target_', '')
    approach_devs = [r[f'opt_path_dev_target_{chosen_tn}'] for r in valid_r
                     if r.get('approach_phase') == 'approach'
                     and not np.isnan(r.get(f'opt_path_dev_target_{chosen_tn}', float('nan')))]
    if approach_devs:
        stats_text = (f"Approach: mean dev={np.mean(approach_devs):.2f}m, "
                      f"max={np.max(approach_devs):.2f}m")
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z / Depth (m)')
    ax.set_title(title or 'Baby Trace (X-Z)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def plot_deviation_over_time(results: List[dict], targets: List[dict],
                             chosen_label: str, output_path: Path,
                             title: str = ""):
    """Plot path deviation to all targets and distance to chosen target over time."""
    valid = [r for r in results
             if not np.isnan(r.get('cumulative_distance_traveled', float('nan')))]
    if len(valid) < 2:
        return

    times = [r['time_sec'] for r in valid]
    chosen_tn = chosen_label.replace('target_', '')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    # Top: optimal path deviation to each target
    target_colors = {'1': 'tab:blue', '2': 'tab:orange', '3': 'tab:green', '4': 'tab:red'}
    for t in targets:
        tn = t['label'].replace('target_', '')
        devs = [r.get(f'opt_path_dev_target_{tn}', float('nan')) for r in valid]
        lw = 2 if tn == chosen_tn else 0.8
        alpha = 1.0 if tn == chosen_tn else 0.4
        ax1.plot(times, devs, color=target_colors.get(tn, 'gray'),
                 linewidth=lw, alpha=alpha, label=f'T{tn}')

    chosen_devs = [r.get(f'opt_path_dev_target_{chosen_tn}', float('nan')) for r in valid]
    valid_devs = [d for d in chosen_devs if not np.isnan(d)]
    mean_dev = np.mean(valid_devs) if valid_devs else 0
    ax1.set_ylabel('Path deviation (m)')
    ax1.set_title(title or 'Path Deviation')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, ncol=4, loc='upper right')
    ax1.text(0.02, 0.95, f'chosen T{chosen_tn}: mean |dev| = {mean_dev:.3f}m',
             transform=ax1.transAxes, ha='left', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    # Bottom: distance to each target
    for t in targets:
        tn = t['label'].replace('target_', '')
        dists = [r.get(f'dist_to_target_{tn}', float('nan')) for r in valid]
        lw = 2 if tn == chosen_tn else 0.8
        alpha = 1.0 if tn == chosen_tn else 0.4
        ax2.plot(times, dists, color=target_colors.get(tn, 'gray'),
                 linewidth=lw, alpha=alpha, label=f'T{tn}')
    ax2.set_ylabel('Distance to target (m)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, ncol=4, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


# ─── Per-trial processing ───

def process_trial(csv_rows: List[dict], targets: List[dict],
                  output_path: Path, trial_label: str = "",
                  ground_truth_choice: Optional[int] = None) -> Optional[List[dict]]:
    """
    Process a single trial. Returns results list (for combining) or None on failure.

    If ground_truth_choice is provided (1-4), use it directly as the first-choice target
    instead of auto-detecting. This uses the human annotation from the Long Data CSVs.
    """
    positions = load_baby_positions(csv_rows)
    valid_count = sum(1 for p in positions if not np.isnan(p['x']))

    if valid_count < 5:
        print(f"    {trial_label}: SKIP (only {valid_count} valid frames)")
        return None

    # Compute start position first (needed for heading-based detection)
    start = compute_start_position(positions, targets)
    if start is None:
        print(f"    {trial_label}: SKIP (no valid start position)")
        return None
    sx, sz = start

    # Use ground truth if provided
    if ground_truth_choice is not None:
        target_label = f"target_{ground_truth_choice}"
        target = next((t for t in targets if t['label'] == target_label), None)
        if target is None:
            print(f"    {trial_label}: SKIP (ground truth target_{ground_truth_choice} not in targets)")
            return None
        print(f"    {trial_label}: using ground truth choice_1 = {target_label}")
    else:
        # Primary: detect first-choice using initial heading (first 20% of valid data)
        target = detect_first_choice_from_initial_heading(positions, targets, sx, sz,
                                                           fraction=0.2)

        # Fallback: use r-value approach if heading method fails
        if target is None:
            first_choice_label = detect_first_choice_from_r(csv_rows)
            if first_choice_label:
                target = next((t for t in targets if t['label'] == first_choice_label), None)
            else:
                target = detect_first_choice(positions, targets)

    if target is None:
        print(f"    {trial_label}: SKIP (no target detected)")
        return None

    # Compute deviation to all targets at every frame
    results = compute_full_deviation(positions, targets, target)
    if not results:
        print(f"    {trial_label}: SKIP (no deviation results)")
        return None

    save_deviation_csv(results, output_path, targets)

    # Generate QC plots
    trial_dir = output_path.parent
    plot_trace_with_targets(
        positions, targets, target, results,
        trial_dir / "baby_trace.png",
        title=f"{trial_label} - trace (chosen: {target['label']})")
    plot_deviation_over_time(
        results, targets, target['label'],
        trial_dir / "path_deviation.png",
        title=f"{trial_label} - path deviation ({target['label']})")

    # Summary stats (approach phase only)
    chosen_tn = target['label'].replace('target_', '')
    approach_devs = [r[f'opt_path_dev_target_{chosen_tn}'] for r in results
                     if r.get('approach_phase') == 'approach'
                     and not np.isnan(r.get(f'opt_path_dev_target_{chosen_tn}', float('nan')))]
    if approach_devs:
        optimal_dist = np.sqrt(
            (target['x'] - results[0]['start_x'])**2 +
            (target['z'] - results[0]['start_z'])**2
        )
        last_valid = [r for r in results
                      if not np.isnan(r.get('cumulative_distance_traveled', float('nan')))]
        total_dist = last_valid[-1]['cumulative_distance_traveled'] if last_valid else 0
        print(f"    {trial_label}: {target['label']}, "
              f"mean_dev={np.mean(approach_devs):.3f}m (approach only), "
              f"max={np.max(approach_devs):.3f}m, "
              f"path={total_dist:.2f}m "
              f"(optimal={optimal_dist:.2f}m), "
              f"{len(results)} frames")

    return results


# ─── Combined CSV output ───

def save_combined_pose_csv(subject_dir: Path, subject_name: str):
    """Combine all trial pose CSVs into one subject-level CSV."""
    trial_dirs = sorted([d for d in subject_dir.iterdir()
                         if d.is_dir() and d.name.startswith('trial_')])
    all_rows = []
    fieldnames = None
    for trial_dir in trial_dirs:
        csv_path = trial_dir / "processed_subject_result_table.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    fieldnames = ['subject', 'trial'] + reader.fieldnames
                for row in reader:
                    row['subject'] = subject_name
                    row['trial'] = trial_dir.name
                    all_rows.append(row)
        except (UnicodeDecodeError, csv.Error):
            continue

    if not all_rows or fieldnames is None:
        return
    out_path = subject_dir / "combined_pose.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  -> combined_pose.csv ({len(all_rows)} rows, {len(trial_dirs)} trials)")


def get_deviation_fieldnames(targets: List[dict]) -> List[str]:
    """Return standard fieldnames for deviation CSV."""
    fieldnames = ['subject', 'trial', 'frame_index', 'time_sec', 'baby_x', 'baby_z',
                  'start_x', 'start_z', 'first_choice',
                  'cumulative_distance_traveled', 'approach_phase']
    for t in targets:
        tn = t['label'].replace('target_', '')
        fieldnames.extend([
            f'dist_to_target_{tn}',
            f'opt_path_dev_target_{tn}',
            f'signed_dev_target_{tn}',
        ])
    return fieldnames


def save_combined_deviation_csv(subject_dir: Path, subject_name: str,
                                targets: List[dict]):
    """Combine all trial deviation CSVs into one subject-level CSV."""
    trial_dirs = sorted([d for d in subject_dir.iterdir()
                         if d.is_dir() and d.name.startswith('trial_')])

    fieldnames = get_deviation_fieldnames(targets)

    all_rows = []
    for trial_dir in trial_dirs:
        csv_path = trial_dir / "optimal_path_deviation.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    row['subject'] = subject_name
                    row['trial'] = trial_dir.name
                    all_rows.append(row)
        except (UnicodeDecodeError, csv.Error):
            continue

    if not all_rows:
        return
    out_path = subject_dir / "combined_optimal_path.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  -> combined_optimal_path.csv ({len(all_rows)} rows)")


def save_global_csvs(output_dir: Path, targets: List[dict]):
    """Combine all subjects into global CSVs at the output_dir level."""
    dev_fieldnames = get_deviation_fieldnames(targets)

    # Global optimal path CSV
    all_dev_rows = []
    all_pose_rows = []
    pose_fieldnames = None

    for subject_dir in sorted(output_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        # Deviation
        dev_csv = subject_dir / "combined_optimal_path.csv"
        if dev_csv.exists():
            try:
                with open(dev_csv) as f:
                    for row in csv.DictReader(f):
                        all_dev_rows.append(row)
            except (UnicodeDecodeError, csv.Error):
                pass
        # Pose
        pose_csv = subject_dir / "combined_pose.csv"
        if pose_csv.exists():
            try:
                with open(pose_csv) as f:
                    reader = csv.DictReader(f)
                    if pose_fieldnames is None:
                        pose_fieldnames = reader.fieldnames
                    for row in reader:
                        all_pose_rows.append(row)
            except (UnicodeDecodeError, csv.Error):
                pass

    if all_dev_rows:
        out = output_dir / "all_subjects_optimal_path.csv"
        with open(out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=dev_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_dev_rows)
        print(f"Global: all_subjects_optimal_path.csv ({len(all_dev_rows)} rows)")

    if all_pose_rows and pose_fieldnames:
        out = output_dir / "all_subjects_pose.csv"
        with open(out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=pose_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_pose_rows)
        print(f"Global: all_subjects_pose.csv ({len(all_pose_rows)} rows)")


# ─── Subject-level processing ───

def process_subject_from_output(subject_dir: Path, targets: List[dict],
                                ground_truth: Optional[Dict[Tuple[str, int], int]] = None) -> Tuple[int, int]:
    """Process all trials for a subject from the extracted output directory."""
    trial_dirs = sorted([d for d in subject_dir.iterdir()
                         if d.is_dir() and d.name.startswith('trial_')])

    # Extract CCD ID for ground truth lookup
    ccd_id = extract_ccd_id(subject_dir.name)

    ok = 0
    fail = 0
    for trial_dir in trial_dirs:
        csv_path = trial_dir / "processed_subject_result_table.csv"
        if not csv_path.exists():
            continue

        try:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
        except (UnicodeDecodeError, csv.Error) as e:
            print(f"    {trial_dir.name}: SKIP (corrupted CSV: {e})")
            fail += 1
            continue

        output_path = trial_dir / "optimal_path_deviation.csv"
        trial_label = trial_dir.name

        # Look up ground truth choice (trial folders are 1-indexed, matching annotations)
        gt_choice = None
        if ground_truth and ccd_id:
            trial_num = int(trial_dir.name.replace("trial_", ""))
            gt_choice = ground_truth.get((ccd_id, trial_num))

        result = process_trial(rows, targets, output_path, trial_label,
                               ground_truth_choice=gt_choice)
        if result is not None:
            ok += 1
        else:
            fail += 1

    # Combined CSVs
    if ok > 0:
        subject_name = subject_dir.name
        save_combined_pose_csv(subject_dir, subject_name)
        save_combined_deviation_csv(subject_dir, subject_name, targets)

    return ok, fail


def process_subject_from_zip(zip_path: Path, output_dir: Path,
                             targets: List[dict],
                             ground_truth: Optional[Dict[Tuple[str, int], int]] = None) -> Tuple[int, int]:
    """Process all trials for a subject directly from a zip file."""
    ok = 0
    fail = 0

    # Extract CCD ID for ground truth lookup
    ccd_id = extract_ccd_id(output_dir.name)

    with zipfile.ZipFile(zip_path) as zf:
        trial_csvs = {}
        for name in zf.namelist():
            if name.endswith('processed_subject_result_table.csv'):
                parts = Path(name).parts
                for part in parts:
                    if part.isdigit():
                        trial_csvs[int(part)] = name
                        break

        for trial_num in sorted(trial_csvs.keys()):
            csv_name = trial_csvs[trial_num]
            try:
                data = zf.read(csv_name).decode('utf-8')
                rows = list(csv.DictReader(io.StringIO(data)))
            except (UnicodeDecodeError, csv.Error) as e:
                print(f"    trial_{trial_num}: SKIP (corrupted CSV: {e})")
                fail += 1
                continue

            trial_out_dir = output_dir / f"trial_{trial_num}"
            output_path = trial_out_dir / "optimal_path_deviation.csv"
            trial_label = f"trial_{trial_num}"

            pose_out = trial_out_dir / "processed_subject_result_table.csv"
            if not pose_out.exists():
                pose_out.parent.mkdir(parents=True, exist_ok=True)
                pose_out.write_text(data)

            # Look up ground truth choice (trial numbers match annotations directly)
            gt_choice = None
            if ground_truth and ccd_id:
                gt_choice = ground_truth.get((ccd_id, trial_num))

            result = process_trial(rows, targets, output_path, trial_label,
                                   ground_truth_choice=gt_choice)
            if result is not None:
                ok += 1
            else:
                fail += 1

    # Combined CSVs
    if ok > 0:
        subject_name = output_dir.name
        save_combined_pose_csv(output_dir, subject_name)
        save_combined_deviation_csv(output_dir, subject_name, targets)

    return ok, fail


def main():
    parser = argparse.ArgumentParser(
        description="Compute optimal path deviation for CCD comprehension data")
    parser.add_argument("output_dir",
                        help="Output directory with extracted CSVs (from process_comprehension_CCD.py)")
    parser.add_argument("--source",
                        help="SSD source directory (reads from zips for subjects not yet extracted)")
    parser.add_argument("--subject", help="Process only this subject")
    parser.add_argument("--threshold", type=float, default=1.5,
                        help="Distance threshold for first-choice detection (meters)")
    parser.add_argument("--targets-json",
                        help="Custom targets JSON file (overrides built-in CCD_TARGETS)")
    parser.add_argument("--ground-truth",
                        help="Ground truth Long Data CSV (default: PVPT_Comprehension_Data - Long Data.csv)",
                        default=str(DEFAULT_GROUND_TRUTH_CSV))
    parser.add_argument("--no-ground-truth", action="store_true",
                        help="Disable ground truth and use auto-detection instead")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    targets = CCD_TARGETS

    if args.targets_json:
        with open(args.targets_json) as f:
            targets = json.load(f)

    # Load ground truth annotations
    ground_truth = {}
    if not args.no_ground_truth:
        gt_path = Path(args.ground_truth)
        ground_truth = load_ground_truth(gt_path)
        if ground_truth:
            print(f"Ground truth: {len(ground_truth)} trials from {gt_path.name}")
        else:
            print(f"WARNING: No ground truth loaded, falling back to auto-detection")
    else:
        print("Ground truth: DISABLED (using auto-detection)")

    print(f"Output: {output_dir}")
    print(f"Targets: {len(targets)}")
    for t in targets:
        print(f"  {t['label']}: x={t['x']:.3f}, z={t['z']:.3f}")
    print()

    # Find subjects to process
    subjects = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    if args.subject:
        subjects = [d for d in subjects if d.name.startswith(args.subject)]

    # Also check source dir for subjects not yet extracted
    source_dir = Path(args.source) if args.source else None
    if source_dir:
        existing_names = {d.name for d in subjects}
        for src_dir in sorted(source_dir.iterdir()):
            if not src_dir.is_dir():
                continue
            if src_dir.name not in existing_names:
                for zf_path in src_dir.glob("*.zip"):
                    try:
                        with zipfile.ZipFile(zf_path) as zf:
                            has_csv = any('processed_subject_result_table.csv' in n
                                          for n in zf.namelist())
                        if has_csv:
                            subjects.append(src_dir)
                            break
                    except zipfile.BadZipFile:
                        pass

    total_ok = 0
    total_fail = 0

    for i, subject_path in enumerate(sorted(subjects, key=lambda p: p.name), 1):
        subject_name = subject_path.name
        if args.subject and not subject_name.startswith(args.subject):
            continue

        print(f"[{i}/{len(subjects)}] {subject_name}")

        out_subject = output_dir / subject_name
        if out_subject.exists() and any(out_subject.iterdir()):
            ok, fail = process_subject_from_output(out_subject, targets, ground_truth)
        elif source_dir:
            src_subject = source_dir / subject_name
            if src_subject.exists():
                for zf_path in src_subject.glob("*.zip"):
                    out_subject.mkdir(parents=True, exist_ok=True)
                    ok, fail = process_subject_from_zip(zf_path, out_subject, targets, ground_truth)
                    break
                else:
                    print(f"  No zip found")
                    ok, fail = 0, 0
            else:
                print(f"  Not found in source")
                ok, fail = 0, 0
        else:
            print(f"  No data")
            ok, fail = 0, 0

        total_ok += ok
        total_fail += fail
        print()

    # Global combined CSVs across all subjects
    if total_ok > 0:
        print()
        save_global_csvs(output_dir, targets)

    print(f"\n{'='*60}")
    print(f"  DONE: {total_ok} trials OK, {total_fail} failed")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
