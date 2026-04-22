#!/usr/bin/env python3
"""
Process BDL dog comprehension traces from optimal_path_data.xlsx.

Reads the raw extracted xlsx, maps to per-trial target coordinates from
point_production_BDL_output, and generates:
  1. Per-trial dog_trace.png (approach/leaving with optimal path)
  2. Per-trial optimal_path_deviation.csv
  3. Per-subject combined CSVs
  4. Global all_subjects_optimal_path.csv
  5. Global approach-phase overlay plots
  6. _all_dog_traces/ collection folder

Usage:
    python process_BDL_comprehension_traces.py
"""

import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).parent))
from process_comprehension_optimal_path import (
    smooth_trace, point_to_line_distance, safe_float, CCD_TARGETS,
)

# ─── Filtering parameters ───
DEPTH_MAX = 4.0       # Max z (m) — targets max at 3.076m, allow ~1m margin
JUMP_MAX = 0.3        # Max frame-to-frame position jump (m) before marking as jitter
MIN_FRAMES_AFTER = 5  # Minimum valid frames needed after filtering
ROI_RADIUS = 0.3      # Region of interest radius around target (m)


def filter_trace(xs, zs, frames, depth_max=DEPTH_MAX, jump_max=JUMP_MAX):
    """
    Filter trace data by depth threshold, then smooth remaining jitter.

    1. Remove frames where z > depth_max (dog behind targets / noise)
    2. Smooth remaining trace with Savitzky-Golay to handle jitter

    Returns: (filtered_xs, filtered_zs, filtered_frames) as numpy arrays
    """
    mask = np.ones(len(xs), dtype=bool)

    # Step 1: Depth threshold
    mask &= (zs <= depth_max)

    fxs, fzs, fframes = xs[mask], zs[mask], frames[mask]

    # Step 2: Smooth jitter instead of rejecting
    if len(fxs) >= 5:
        smooth_xs, smooth_zs = smooth_trace(list(fxs), list(fzs), window=11)
        fxs = np.array(smooth_xs)
        fzs = np.array(smooth_zs)

    return fxs, fzs, fframes


# ─── Paths ───
DATA_ROOT = Path("/home/tigerli/Documents/pointing_data")
BDL_COMP_OUTPUT = DATA_ROOT / "point_comprehension_BDL_output"
BDL_PROD_OUTPUT = DATA_ROOT / "point_production_BDL_output"
ANALYSIS_OUTPUT = DATA_ROOT / "_analysis_output"
XLSX_PATH = BDL_COMP_OUTPUT / "optimal_path_data.xlsx"
LONG_DATA = BDL_COMP_OUTPUT / "PVP_Comprehension_Data - Long Data.csv"

# The xlsx was computed using CCD_TARGETS (verified by trilateration)
BDL_TARGETS = CCD_TARGETS


def load_targets_from_production(dog_id, trial_number):
    """Load target coordinates from matching production output folder."""
    bdl_upper = dog_id.upper()
    matches = list(BDL_PROD_OUTPUT.glob(f"{bdl_upper}*"))
    if not matches:
        return None

    subj_dir = matches[0]
    # Try this trial first, then trial_1 as fallback (targets are static per study)
    for trial_name in [f"trial_{trial_number}", "trial_1", "trial_2"]:
        tpath = subj_dir / trial_name / "target_coordinates.json"
        if tpath.exists():
            try:
                with open(tpath) as f:
                    data = json.load(f)
                targets = []
                for t in data.get("targets", []):
                    targets.append({
                        "label": t["label"],
                        "x": t["world_coords"][0],
                        "y": t["world_coords"][1],
                        "z": t["world_coords"][2],
                    })
                if len(targets) >= 2:
                    return targets
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    return None


def reconstruct_targets_from_distances(trial_df):
    """
    Reconstruct approximate target positions from distance data when
    target_coordinates.json is not available.
    Uses trilateration from two frames.
    """
    # Get first and last valid frames
    valid = trial_df.dropna(subset=['trace3d_x', 'trace3d_z'])
    if len(valid) < 2:
        return None

    targets = []
    for tn in range(1, 5):
        dcol = f'dist_to_target_{tn}'
        if dcol not in valid.columns:
            continue

        # Use first frame position + distance, plus angle to estimate
        # We'll use the opt_path equation to get the line, and distance to place the target
        first = valid.iloc[0]
        d1 = first[dcol]
        x1, z1 = first['trace3d_x'], first['trace3d_z']

        # Use angle column to determine direction
        acol = f'angle_to_target_{tn}'
        if acol in valid.columns:
            angle_deg = first[acol]
            if not np.isnan(angle_deg) and not np.isnan(d1):
                # angle is bearing-like, convert to radians
                angle_rad = np.radians(angle_deg)
                tx = x1 + d1 * np.sin(angle_rad)
                tz = z1 + d1 * np.cos(angle_rad)
                targets.append({
                    "label": f"target_{tn}",
                    "x": tx, "y": 0.0, "z": tz,
                })

    return targets if len(targets) >= 2 else None


def compute_start_position_from_trace(xs, zs):
    """Return the first valid point of the (already smoothed) trace as start."""
    for i in range(len(xs)):
        if not np.isnan(xs[i]) and not np.isnan(zs[i]):
            return float(xs[i]), float(zs[i])
    return None, None


def process_trial_from_df(trial_df, targets, dog_id, trial_num):
    """Process a single trial from the xlsx dataframe."""
    valid = trial_df.dropna(subset=['trace3d_x', 'trace3d_z'])
    if len(valid) < 5:
        return None, None

    xs = valid['trace3d_x'].values
    zs = valid['trace3d_z'].values
    frames = valid['frame_index'].values

    # Filter: depth threshold + jitter rejection
    xs, zs, frames = filter_trace(xs, zs, frames)
    if len(xs) < MIN_FRAMES_AFTER:
        return None, None

    # Rebuild valid df to match filtered frames
    valid = valid[valid['frame_index'].isin(frames)].copy()

    # Smoothed start position
    sx, sz = compute_start_position_from_trace(xs, zs)
    if sx is None:
        return None, None

    # Get first choice target
    choice_1 = trial_df['choice_1'].iloc[0]
    if pd.isna(choice_1):
        return None, None
    choice_1 = int(choice_1)

    target_label = f"target_{choice_1}"
    chosen = next((t for t in targets if t['label'] == target_label), None)
    if chosen is None:
        return None, None

    # Compute deviation (pass 1: build results without approach_phase)
    # Use smoothed coordinates from filter_trace (xs, zs arrays)
    chosen_tn = str(choice_1)
    cumul_dist = 0.0
    prev_x, prev_z = None, None

    results = []
    for i in range(len(xs)):
        bx, bz = xs[i], zs[i]
        frame_idx = int(frames[i])

        # Cumulative distance
        if prev_x is not None:
            cumul_dist += np.sqrt((bx - prev_x)**2 + (bz - prev_z)**2)
        prev_x, prev_z = bx, bz

        # Get percent_complete from original data if available
        pc_rows = valid[valid['frame_index'] == frame_idx]
        pct = pc_rows['percent_complete'].iloc[0] if len(pc_rows) > 0 else float('nan')

        r = {
            'frame_index': frame_idx,
            'percent_complete': pct,
            'baby_x': bx, 'baby_z': bz,
            'start_x': sx, 'start_z': sz,
            'first_choice': target_label,
            'cumulative_distance_traveled': cumul_dist,
        }

        # Deviation to each target
        for t in targets:
            tn = t['label'].replace('target_', '')
            unsigned_dev, signed_dev = point_to_line_distance(
                bx, bz, sx, sz, t['x'], t['z'])
            dist_to_t = np.sqrt((bx - t['x'])**2 + (bz - t['z'])**2)
            r[f'dist_to_target_{tn}'] = dist_to_t
            r[f'opt_path_dev_target_{tn}'] = unsigned_dev
            r[f'signed_dev_target_{tn}'] = signed_dev

        results.append(r)

    # Pass 2: determine approach cutoff index
    # Primary: ROI entry (dist <= ROI_RADIUS). Fallback: min distance frame.
    dist_key = f'dist_to_target_{chosen_tn}'
    roi_idx = None
    min_dist = float('inf')
    min_idx = 0
    for i, r in enumerate(results):
        d = r.get(dist_key, float('inf'))
        if not np.isnan(d):
            if roi_idx is None and d <= ROI_RADIUS:
                roi_idx = i
            if d < min_dist:
                min_dist = d
                min_idx = i

    cutoff = roi_idx if roi_idx is not None else min_idx
    for i, r in enumerate(results):
        r['approach_phase'] = 'approach' if i <= cutoff else 'arrived'

    return results, chosen


def plot_dog_trace(results, targets, chosen, output_path, title=""):
    """Plot dog trace with approach/arrived and optimal path.
    Approach phase: rainbow dots. Arrived phase: gray dots."""

    if not results:
        return

    valid_r = [r for r in results if not np.isnan(r['baby_x'])]
    if len(valid_r) < 2:
        return

    xs = [r['baby_x'] for r in valid_r]
    zs = [r['baby_z'] for r in valid_r]
    phases = [r['approach_phase'] for r in valid_r]

    app_xs = [x for x, p in zip(xs, phases) if p == 'approach']
    app_zs = [z for z, p in zip(zs, phases) if p == 'approach']
    leave_xs = [x for x, p in zip(xs, phases) if p != 'approach']
    leave_zs = [z for z, p in zip(zs, phases) if p != 'approach']

    fig, ax = plt.subplots(figsize=(7, 6))

    # Optimal path line
    sx, sz = results[0]['start_x'], results[0]['start_z']
    tx, tz = chosen['x'], chosen['z']
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

    # Start / end
    ax.scatter(xs[0], zs[0], c='blue', s=100, marker='s', zorder=5, label='Start')
    ax.scatter(xs[-1], zs[-1], c='red', s=100, marker='*', zorder=5, label='End')

    if app_xs and leave_xs:
        ax.scatter(app_xs[-1], app_zs[-1], c='orange', s=80, marker='v',
                   edgecolors='black', linewidths=0.5, zorder=5, label='ROI entry')

    # Targets
    for t in targets:
        marker = 'D' if t['label'] == chosen['label'] else 'o'
        color = 'red' if t['label'] == chosen['label'] else 'gray'
        ax.scatter(t['x'], t['z'], c=color, s=120, marker=marker,
                   edgecolors='black', linewidths=1, zorder=4)
        ax.annotate(t['label'].replace('target_', 'T'), (t['x'], t['z']),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)

    # Stats
    chosen_tn = chosen['label'].replace('target_', '')
    approach_devs = [r[f'opt_path_dev_target_{chosen_tn}'] for r in results
                     if r.get('approach_phase') == 'approach'
                     and not np.isnan(r.get(f'opt_path_dev_target_{chosen_tn}', float('nan')))]
    if approach_devs:
        stats = f"Approach: mean dev={np.mean(approach_devs):.2f}m, max={np.max(approach_devs):.2f}m"
        ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=7,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z / Depth (m)')
    ax.set_title(title or 'Dog Trace (X-Z)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def plot_global_approach_phase(all_traces, targets_cache, output_path):
    """Global approach-phase overlay for BDL dog comprehension."""
    target_colors = {'1': 'tab:red', '2': 'tab:orange', '3': 'tab:green', '4': 'tab:blue'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: approach-phase only
    ax = axes[0]
    counts = defaultdict(int)
    for trace in all_traces:
        chosen_tn = trace['chosen'].replace('target_', '')
        color = target_colors.get(chosen_tn, 'gray')
        ax.plot(trace['app_xs'], trace['app_zs'],
                color=color, alpha=0.15, linewidth=0.8)
        counts[trace['chosen']] += 1

    # Mean targets across all studies
    all_targets = []
    for tlist in targets_cache.values():
        all_targets.extend(tlist)
    mean_targets = {}
    for tn in ['1', '2', '3', '4']:
        txs = [t['x'] for t in all_targets if t['label'] == f'target_{tn}']
        tzs = [t['z'] for t in all_targets if t['label'] == f'target_{tn}']
        if txs:
            mean_targets[tn] = (np.mean(txs), np.mean(tzs))
            ax.scatter(mean_targets[tn][0], mean_targets[tn][1],
                       c=target_colors.get(tn, 'gray'), s=150, marker='*',
                       edgecolors='black', linewidths=1, zorder=10)
            ax.annotate(f"T{tn}", mean_targets[tn],
                        textcoords='offset points', xytext=(8, 8),
                        fontsize=10, fontweight='bold')

    count_text = ', '.join(f"T{k.replace('target_','')}: {v}"
                           for k, v in sorted(counts.items()))
    ax.set_title(f"BDL Dog: Approach Phase Only (n={len(all_traces)})\n{count_text}")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z / Depth (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Right: full traces with approach highlighted
    ax2 = axes[1]
    for trace in all_traces:
        ax2.plot(trace['full_xs'], trace['full_zs'],
                 color='lightgray', alpha=0.1, linewidth=0.5)
        chosen_tn = trace['chosen'].replace('target_', '')
        color = target_colors.get(chosen_tn, 'gray')
        ax2.plot(trace['app_xs'], trace['app_zs'],
                 color=color, alpha=0.2, linewidth=1.0)

    for tn, pos in mean_targets.items():
        ax2.scatter(pos[0], pos[1], c=target_colors.get(tn, 'gray'),
                    s=150, marker='*', edgecolors='black', linewidths=1, zorder=10)
        ax2.annotate(f"T{tn}", pos, textcoords='offset points',
                     xytext=(8, 8), fontsize=10, fontweight='bold')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, linewidth=2, label=f'Target {tn}')
                       for tn, c in sorted(target_colors.items())]
    ax2.legend(handles=legend_elements, fontsize=8, loc='best')

    ax2.set_title("BDL Dog: Full Traces (gray) + Approach (colored)")
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z / Depth (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")

    # Per-target breakdown
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, tn in enumerate(['1', '2', '3', '4']):
        ax = axes[idx // 2][idx % 2]
        target_traces = [t for t in all_traces if t['chosen'] == f'target_{tn}']
        color = target_colors.get(tn, 'gray')

        for trace in target_traces:
            ax.plot(trace['app_xs'], trace['app_zs'],
                    color=color, alpha=0.3, linewidth=1.0)
            ax.scatter(trace['app_xs'][0], trace['app_zs'][0],
                       c='green', s=15, alpha=0.3, zorder=3)
            ax.scatter(trace['app_xs'][-1], trace['app_zs'][-1],
                       c='red', s=15, alpha=0.3, zorder=3)

        for t_tn, pos in mean_targets.items():
            m = '*' if t_tn == tn else 'o'
            c = target_colors.get(t_tn, 'gray') if t_tn == tn else 'lightgray'
            ax.scatter(pos[0], pos[1], c=c, s=120, marker=m,
                       edgecolors='black', linewidths=1, zorder=10)
            ax.annotate(f"T{t_tn}", pos, textcoords='offset points',
                        xytext=(5, 5), fontsize=9)

        ax.set_title(f"Target {tn} (n={len(target_traces)})")
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z / Depth (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle("BDL Dog: Approach Phase by Target", fontsize=14, y=1.02)
    plt.tight_layout()
    per_target_path = output_path.parent / output_path.name.replace('.png', '_per_target.png')
    plt.savefig(per_target_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {per_target_path}")


def main():
    print("=" * 70)
    print("  BDL Dog Comprehension: Process Traces from XLSX")
    print("=" * 70)

    # Load xlsx
    print(f"Loading {XLSX_PATH}...")
    df = pd.read_excel(XLSX_PATH)
    print(f"  {len(df)} rows, {df['dog_id'].nunique()} dogs, "
          f"{df.groupby(['dog_id','trial_number']).ngroups} trials")

    # Create output structure
    all_traces_dir = BDL_COMP_OUTPUT / "_all_dog_traces"
    all_traces_dir.mkdir(exist_ok=True)
    ANALYSIS_OUTPUT.mkdir(exist_ok=True)

    targets = BDL_TARGETS  # Same targets for all BDL dogs (verified by trilateration)
    targets_cache = {}
    all_approach_traces = []
    total_ok = 0
    total_skip = 0
    total_frames_before = 0
    total_frames_after = 0
    all_dev_rows = []

    print(f"Using CCD_TARGETS (verified from xlsx distances):")
    for t in targets:
        print(f"  {t['label']}: x={t['x']:.3f}, z={t['z']:.3f}")
    print(f"\nFiltering: depth_max={DEPTH_MAX}m, jump_max={JUMP_MAX}m")

    for dog_id in sorted(df['dog_id'].unique()):
        dog_df = df[df['dog_id'] == dog_id]
        dog_name = dog_df['dog_name'].iloc[0]
        print(f"\n[{dog_id.upper()}] {dog_name}")

        # Create subject directory
        subj_dir = BDL_COMP_OUTPUT / f"{dog_id.upper()}_{dog_name}"
        subj_dir.mkdir(exist_ok=True)

        targets_cache[dog_id] = targets

        subject_ok = 0
        for trial_num in sorted(dog_df['trial_number'].unique()):
            trial_df = dog_df[dog_df['trial_number'] == trial_num].copy()
            trial_df = trial_df.sort_values('frame_index')

            raw_valid = trial_df.dropna(subset=['trace3d_x', 'trace3d_z'])
            n_before = len(raw_valid)
            total_frames_before += n_before

            trial_dir = subj_dir / f"trial_{trial_num}"
            trial_dir.mkdir(exist_ok=True)

            trial_targets = targets

            results, chosen = process_trial_from_df(
                trial_df, trial_targets, dog_id, trial_num)

            if results is None:
                total_skip += 1
                continue

            total_ok += 1
            total_frames_after += len(results)
            subject_ok += 1

            # Save CSV
            dev_path = trial_dir / "optimal_path_deviation.csv"
            fieldnames = ['frame_index', 'percent_complete', 'baby_x', 'baby_z',
                          'start_x', 'start_z', 'first_choice',
                          'cumulative_distance_traveled', 'approach_phase']
            for t in trial_targets:
                tn = t['label'].replace('target_', '')
                fieldnames.extend([f'dist_to_target_{tn}',
                                   f'opt_path_dev_target_{tn}',
                                   f'signed_dev_target_{tn}'])

            with open(dev_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)

            # Add to global rows
            for r in results:
                r['subject'] = f"{dog_id.upper()}_{dog_name}"
                r['trial'] = f"trial_{trial_num}"
                all_dev_rows.append(r)

            # Generate plot
            trial_label = f"{dog_id.upper()}/trial_{trial_num}"
            plot_dog_trace(
                results, trial_targets, chosen,
                trial_dir / "dog_trace.png",
                title=f"{trial_label} - dog trace (chosen: {chosen['label']})")

            # Copy to collection
            src = trial_dir / "dog_trace.png"
            if src.exists():
                dst = all_traces_dir / f"{dog_id.upper()}_{dog_name}_trial_{trial_num}_dog_trace.png"
                shutil.copy2(src, dst)

            # Collect for global plot
            app_xs = [r['baby_x'] for r in results if r['approach_phase'] == 'approach'
                       and not np.isnan(r['baby_x'])]
            app_zs = [r['baby_z'] for r in results if r['approach_phase'] == 'approach'
                       and not np.isnan(r['baby_z'])]
            full_xs = [r['baby_x'] for r in results if not np.isnan(r['baby_x'])]
            full_zs = [r['baby_z'] for r in results if not np.isnan(r['baby_z'])]

            if app_xs:
                all_approach_traces.append({
                    'app_xs': app_xs, 'app_zs': app_zs,
                    'full_xs': full_xs, 'full_zs': full_zs,
                    'chosen': chosen['label'],
                    'label': trial_label,
                })

            # Summary
            chosen_tn = chosen['label'].replace('target_', '')
            approach_devs = [r[f'opt_path_dev_target_{chosen_tn}'] for r in results
                             if r['approach_phase'] == 'approach'
                             and not np.isnan(r.get(f'opt_path_dev_target_{chosen_tn}', float('nan')))]
            if approach_devs:
                print(f"    trial_{trial_num}: {chosen['label']}, "
                      f"mean_dev={np.mean(approach_devs):.3f}m, "
                      f"approach={len(approach_devs)}/{len(results)} frames")

        if subject_ok > 0:
            print(f"  {subject_ok} trials OK")

    # Save global CSV
    if all_dev_rows:
        global_fieldnames = ['subject', 'trial', 'frame_index', 'percent_complete',
                             'baby_x', 'baby_z', 'start_x', 'start_z', 'first_choice',
                             'cumulative_distance_traveled', 'approach_phase']
        for tn in ['1', '2', '3', '4']:
            global_fieldnames.extend([f'dist_to_target_{tn}',
                                      f'opt_path_dev_target_{tn}',
                                      f'signed_dev_target_{tn}'])

        global_path = BDL_COMP_OUTPUT / "all_subjects_optimal_path.csv"
        with open(global_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=global_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_dev_rows)
        print(f"\nGlobal CSV: {global_path} ({len(all_dev_rows)} rows)")

    # Global approach-phase plots
    if all_approach_traces:
        plot_global_approach_phase(
            all_approach_traces, targets_cache,
            ANALYSIS_OUTPUT / "08_BDL_dog_approach_phase_overlay.png")

    pct = (1 - total_frames_after / total_frames_before) * 100 if total_frames_before else 0
    print(f"\n{'='*70}")
    print(f"  DONE: {total_ok} OK, {total_skip} skipped")
    print(f"  Filtering: {total_frames_before} → {total_frames_after} frames "
          f"({pct:.1f}% removed, depth_max={DEPTH_MAX}m, jump_max={JUMP_MAX}m)")
    print(f"  Output: {BDL_COMP_OUTPUT}")
    print(f"  Traces: {all_traces_dir} ({total_ok} plots)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
