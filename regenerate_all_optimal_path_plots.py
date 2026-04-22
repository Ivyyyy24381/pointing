#!/usr/bin/env python3
"""
Regenerate all optimal path deviation plots + global approach-phase visualizations.

Handles:
  1. CCD Comprehension (baby traces) - reprocess with ground truth + smoothed start
  2. Owner Pointing (dog traces) - process with ground truth + smoothed start
  3. Global approach-phase overlay plots for both datasets
  4. Copy updated plots to collection folders

Usage:
    python regenerate_all_optimal_path_plots.py [--ccd-only] [--dog-only] [--plots-only]
"""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).parent))
from process_comprehension_optimal_path import (
    process_trial, CCD_TARGETS, save_combined_deviation_csv,
    save_combined_pose_csv, save_global_csvs, get_deviation_fieldnames,
    load_baby_positions, compute_start_position, compute_full_deviation,
    save_deviation_csv, plot_trace_with_targets, plot_deviation_over_time,
    safe_float, smooth_trace,
)

# ─── Paths ───
DATA_ROOT = Path("/home/tigerli/Documents/pointing_data")
CCD_OUTPUT = DATA_ROOT / "point_comprehension_CCD_output"
OWNER_OUTPUT = DATA_ROOT / "owner_pointing_output"
ANALYSIS_OUTPUT = DATA_ROOT / "_analysis_output"

CCD_ANNOTATIONS = DATA_ROOT / "PVPT_Comprehension_Data - Long Data.csv"
OWNER_LONG_DATA = DATA_ROOT / "PVPO_Production_Data - Long_Data.csv"


def load_annotations(csv_path, id_col, trial_col):
    """Load annotations, return dict of {(participant_id, trial_num): row}."""
    if not csv_path.exists():
        print(f"  WARNING: annotation file not found: {csv_path}")
        return {}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    result = {}
    for r in rows:
        if r.get("excluded", "").upper() == "Y":
            continue
        pid = r[id_col].strip()
        try:
            trial = int(r[trial_col])
        except (ValueError, KeyError):
            continue
        result[(pid, trial)] = r
    return result


def load_targets_from_json(trial_dir):
    """Load per-trial targets from target_coordinates.json."""
    tpath = trial_dir / "target_coordinates.json"
    if tpath.exists():
        try:
            with open(tpath) as f:
                data = json.load(f)
            if "targets" in data:
                targets = []
                for t in data["targets"]:
                    targets.append({
                        "label": t["label"],
                        "x": t["world_coords"][0],
                        "y": t["world_coords"][1],
                        "z": t["world_coords"][2],
                    })
                if len(targets) >= 2:
                    return targets
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    return None


def load_reference_targets(subj_dir):
    """Load study-level targets from reference_targets/cam1_targets.json."""
    ref_path = subj_dir / "reference_targets" / "cam1_targets.json"
    if ref_path.exists():
        try:
            with open(ref_path) as f:
                data = json.load(f)
            targets = []
            for t in data:
                targets.append({
                    "label": t["label"],
                    "x": t["x"],
                    "y": t.get("y", 0),
                    "z": t["z"],
                })
            if len(targets) >= 2:
                return targets
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    return None


def load_dog_positions(csv_rows):
    """Extract dog positions from CSV rows (same structure as baby)."""
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


def plot_dog_trace_with_targets(positions, targets, chosen_target, results,
                                 output_path, title=""):
    """Plot dog trace in X-Z plane with targets and optimal path line.
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

    # Optimal path line
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

    # Start / end
    ax.scatter(xs[0], zs[0], c='blue', s=100, marker='s', zorder=5, label='Start')
    ax.scatter(xs[-1], zs[-1], c='red', s=100, marker='*', zorder=5, label='End')

    if app_xs and leave_xs:
        ax.scatter(app_xs[-1], app_zs[-1], c='orange', s=80, marker='v',
                   edgecolors='black', linewidths=0.5, zorder=5, label='ROI entry')

    # All targets
    for t in targets:
        marker = 'D' if t['label'] == chosen_target['label'] else 'o'
        color = 'red' if t['label'] == chosen_target['label'] else 'gray'
        ax.scatter(t['x'], t['z'], c=color, s=120, marker=marker,
                   edgecolors='black', linewidths=1, zorder=4)
        ax.annotate(t['label'].replace('target_', 'T'), (t['x'], t['z']),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)

    # Stats
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
    ax.set_title(title or 'Dog Trace (X-Z)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  CCD Comprehension Reprocessing
# ═══════════════════════════════════════════════════════════════
def reprocess_ccd():
    print("=" * 70)
    print("  CCD COMPREHENSION: Reprocess with Ground Truth + Smoothed Start")
    print("=" * 70)

    annotations = load_annotations(CCD_ANNOTATIONS, "participant_id", "trial_number")
    print(f"Annotations loaded: {len(annotations)} trials")

    total_ok = 0
    total_skip = 0

    for subj_dir in sorted(CCD_OUTPUT.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith("_"):
            continue

        subj_id = subj_dir.name.split("_")[0]
        print(f"\n[{subj_id}] {subj_dir.name}")

        trial_dirs = sorted([d for d in subj_dir.iterdir()
                            if d.is_dir() and d.name.startswith("trial_")])

        subject_ok = 0
        for trial_dir in trial_dirs:
            trial_num = int(trial_dir.name.replace("trial_", ""))

            targets = load_targets_from_json(trial_dir)
            if targets is None:
                targets = CCD_TARGETS

            # Ground truth choice (0-indexed output → 1-indexed annotation)
            ann = annotations.get((subj_id, trial_num + 1))
            if ann is None:
                ann = annotations.get((subj_id, trial_num))

            ground_truth_choice = None
            if ann is not None:
                c1 = ann.get("choice_1", "NA").strip()
                if c1 not in ("NA", ""):
                    try:
                        ground_truth_choice = int(c1)
                    except ValueError:
                        pass

            csv_path = trial_dir / "processed_subject_result_table.csv"
            if not csv_path.exists():
                continue

            try:
                with open(csv_path) as f:
                    rows = list(csv.DictReader(f))
            except (UnicodeDecodeError, csv.Error):
                continue

            output_path = trial_dir / "optimal_path_deviation.csv"
            trial_label = trial_dir.name

            if ground_truth_choice is not None:
                result = process_trial(rows, targets, output_path, trial_label,
                                      ground_truth_choice=ground_truth_choice)
            else:
                result = process_trial(rows, targets, output_path, trial_label)

            if result is not None:
                total_ok += 1
                subject_ok += 1
            else:
                total_skip += 1

        if subject_ok > 0:
            save_combined_pose_csv(subj_dir, subj_dir.name)
            save_combined_deviation_csv(subj_dir, subj_dir.name, CCD_TARGETS)

    if total_ok > 0:
        print(f"\nRegenerating global CSVs...")
        save_global_csvs(CCD_OUTPUT, CCD_TARGETS)

    # Update _all_baby_traces/
    all_traces_dir = CCD_OUTPUT / "_all_baby_traces"
    if all_traces_dir.exists():
        updated = 0
        for subj_dir in sorted(CCD_OUTPUT.iterdir()):
            if not subj_dir.is_dir() or subj_dir.name.startswith("_"):
                continue
            for trial_dir in sorted(subj_dir.iterdir()):
                if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                    continue
                src = trial_dir / "baby_trace.png"
                if src.exists():
                    dst = all_traces_dir / f"{subj_dir.name}_{trial_dir.name}_baby_trace.png"
                    shutil.copy2(src, dst)
                    updated += 1
        print(f"Updated {updated} plots in _all_baby_traces/")

    print(f"\nCCD DONE: {total_ok} OK, {total_skip} skipped")
    return total_ok


# ═══════════════════════════════════════════════════════════════
#  Owner Pointing Dog Trace Processing
# ═══════════════════════════════════════════════════════════════
def process_dog_trial(csv_rows, targets, output_path, trial_label="",
                      ground_truth_choice=None):
    """Process a single dog trial for optimal path deviation."""
    positions = load_dog_positions(csv_rows)
    valid_count = sum(1 for p in positions if not np.isnan(p['x']))

    if valid_count < 5:
        print(f"    {trial_label}: SKIP (only {valid_count} valid frames)")
        return None

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
        # Auto-detect from initial heading
        from process_comprehension_optimal_path import (
            detect_first_choice_from_initial_heading,
            detect_first_choice_from_r,
            detect_first_choice,
        )
        target = detect_first_choice_from_initial_heading(
            positions, targets, sx, sz, fraction=0.2)
        if target is None:
            first_choice_label = detect_first_choice_from_r(csv_rows)
            if first_choice_label:
                target = next((t for t in targets if t['label'] == first_choice_label), None)
            else:
                target = detect_first_choice(positions, targets)

    if target is None:
        print(f"    {trial_label}: SKIP (no target detected)")
        return None

    results = compute_full_deviation(positions, targets, target)
    if not results:
        print(f"    {trial_label}: SKIP (no deviation results)")
        return None

    # Rename baby_x/baby_z to dog_x/dog_z in column labels for output
    save_deviation_csv(results, output_path, targets)

    # Generate plots
    trial_dir = output_path.parent
    plot_dog_trace_with_targets(
        positions, targets, target, results,
        trial_dir / "dog_trace_optimal.png",
        title=f"{trial_label} - dog trace (chosen: {target['label']})")
    plot_deviation_over_time(
        results, targets, target['label'],
        trial_dir / "dog_path_deviation.png",
        title=f"{trial_label} - dog path deviation ({target['label']})")

    # Summary
    chosen_tn = target['label'].replace('target_', '')
    approach_devs = [r[f'opt_path_dev_target_{chosen_tn}'] for r in results
                     if r.get('approach_phase') == 'approach'
                     and not np.isnan(r.get(f'opt_path_dev_target_{chosen_tn}', float('nan')))]
    if approach_devs:
        optimal_dist = np.sqrt(
            (target['x'] - results[0]['start_x'])**2 +
            (target['z'] - results[0]['start_z'])**2
        )
        print(f"    {trial_label}: {target['label']}, "
              f"mean_dev={np.mean(approach_devs):.3f}m (approach only), "
              f"max={np.max(approach_devs):.3f}m, "
              f"approach={len(approach_devs)}/{len(results)} frames")

    return results


def reprocess_dog():
    print("\n" + "=" * 70)
    print("  OWNER POINTING: Dog Trace Optimal Path with Ground Truth")
    print("=" * 70)

    annotations = load_annotations(OWNER_LONG_DATA, "participant_id", "trial")
    print(f"Annotations loaded: {len(annotations)} trials")

    total_ok = 0
    total_skip = 0

    all_dog_traces_dir = OWNER_OUTPUT / "_all_dog_traces_optimal"
    all_dog_traces_dir.mkdir(exist_ok=True)

    for subj_dir in sorted(OWNER_OUTPUT.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith("_"):
            continue

        subj_id = subj_dir.name.split("_")[0]
        print(f"\n[{subj_id}] {subj_dir.name}")

        # Load reference targets for this study
        targets = load_reference_targets(subj_dir)
        if targets is None:
            print(f"  WARNING: no reference targets found, skipping")
            continue

        trial_dirs = sorted([d for d in subj_dir.iterdir()
                            if d.is_dir() and d.name.startswith("trial_")])

        subject_ok = 0
        for trial_dir in trial_dirs:
            trial_num = int(trial_dir.name.replace("trial_", ""))

            # Ground truth choice (0-indexed output → 1-indexed annotation)
            ann = annotations.get((subj_id, trial_num + 1))
            if ann is None:
                ann = annotations.get((subj_id, trial_num))

            ground_truth_choice = None
            if ann is not None:
                c1 = ann.get("choice_1", "NA").strip()
                if c1 not in ("NA", ""):
                    try:
                        ground_truth_choice = int(c1)
                    except ValueError:
                        pass

            # Dog CSV is in trial_dir/cam1/
            cam1_dir = trial_dir / "cam1"
            csv_path = cam1_dir / "processed_dog_result_table.csv"
            if not csv_path.exists():
                continue

            try:
                with open(csv_path) as f:
                    rows = list(csv.DictReader(f))
            except (UnicodeDecodeError, csv.Error):
                continue

            # Check if per-trial targets exist
            trial_targets = targets
            trial_target_json = cam1_dir / "target_coordinates.json"
            if trial_target_json.exists():
                loaded = load_targets_from_json(cam1_dir)
                if loaded is not None:
                    trial_targets = loaded

            output_path = cam1_dir / "dog_optimal_path_deviation.csv"
            trial_label = f"{subj_id}/{trial_dir.name}"

            result = process_dog_trial(rows, trial_targets, output_path,
                                       trial_label, ground_truth_choice)
            if result is not None:
                total_ok += 1
                subject_ok += 1
                # Copy trace plot
                src = cam1_dir / "dog_trace_optimal.png"
                if src.exists():
                    dst = all_dog_traces_dir / f"{subj_dir.name}_{trial_dir.name}_dog_trace_optimal.png"
                    shutil.copy2(src, dst)
            else:
                total_skip += 1

    print(f"\nDog DONE: {total_ok} OK, {total_skip} skipped")
    return total_ok


# ═══════════════════════════════════════════════════════════════
#  Global Approach-Phase Visualization
# ═══════════════════════════════════════════════════════════════
def plot_global_approach_phase(dataset_name, output_dir, csv_glob_pattern,
                                output_path, subject_col_pattern="baby"):
    """Create global overlay showing approach-phase segments with optimal paths."""
    print(f"\nGenerating global approach-phase plot for {dataset_name}...")

    all_approach_traces = []  # [(xs, zs, start_xy, target_xy, chosen_label, subj_trial)]
    all_full_traces = []

    for subj_dir in sorted(output_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith("_"):
            continue

        subj_id = subj_dir.name.split("_")[0]
        trial_dirs = sorted([d for d in subj_dir.iterdir()
                            if d.is_dir() and d.name.startswith("trial_")])

        for trial_dir in trial_dirs:
            # Find the deviation CSV
            if subject_col_pattern == "dog":
                dev_csv = trial_dir / "cam1" / "dog_optimal_path_deviation.csv"
            else:
                dev_csv = trial_dir / "optimal_path_deviation.csv"

            if not dev_csv.exists():
                continue

            try:
                with open(dev_csv) as f:
                    rows = list(csv.DictReader(f))
            except (UnicodeDecodeError, csv.Error):
                continue

            if not rows:
                continue

            # Extract trace data
            approach_xs, approach_zs = [], []
            full_xs, full_zs = [], []
            start_x = safe_float(rows[0].get('start_x'))
            start_z = safe_float(rows[0].get('start_z'))
            chosen = rows[0].get('first_choice', '')

            for r in rows:
                bx = safe_float(r.get('baby_x'))
                bz = safe_float(r.get('baby_z'))
                if np.isnan(bx) or np.isnan(bz):
                    continue
                full_xs.append(bx)
                full_zs.append(bz)
                if r.get('approach_phase') == 'approach':
                    approach_xs.append(bx)
                    approach_zs.append(bz)

            if not approach_xs or np.isnan(start_x):
                continue

            # Get target position from deviation CSV
            chosen_tn = chosen.replace('target_', '')
            target_x, target_z = None, None
            for r in rows:
                d = safe_float(r.get(f'dist_to_target_{chosen_tn}'))
                bx = safe_float(r.get('baby_x'))
                bz = safe_float(r.get('baby_z'))
                if not any(np.isnan(v) for v in [d, bx, bz]):
                    # Can reconstruct target position from distance + known targets
                    break

            all_approach_traces.append({
                'approach_xs': approach_xs, 'approach_zs': approach_zs,
                'full_xs': full_xs, 'full_zs': full_zs,
                'start': (start_x, start_z),
                'chosen': chosen,
                'label': f"{subj_id}/{trial_dir.name}",
            })

    if not all_approach_traces:
        print(f"  No approach traces found for {dataset_name}")
        return

    # Load targets for rendering
    if dataset_name == "CCD":
        global_targets = CCD_TARGETS
    else:
        # Use mean of all reference targets
        global_targets = None
        for subj_dir in sorted(output_dir.iterdir()):
            if not subj_dir.is_dir() or subj_dir.name.startswith("_"):
                continue
            t = load_reference_targets(subj_dir)
            if t is not None:
                global_targets = t
                break

    # ─── Plot 1: All approach-phase traces with optimal paths ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: approach-phase only with optimal path lines
    ax = axes[0]
    target_colors_map = {'1': 'tab:red', '2': 'tab:orange', '3': 'tab:green', '4': 'tab:blue'}

    for trace in all_approach_traces:
        chosen_tn = trace['chosen'].replace('target_', '')
        color = target_colors_map.get(chosen_tn, 'gray')
        ax.plot(trace['approach_xs'], trace['approach_zs'],
                color=color, alpha=0.15, linewidth=0.8)
        # Optimal path line
        sx, sz = trace['start']
        if global_targets:
            tgt = next((t for t in global_targets if t['label'] == trace['chosen']), None)
            if tgt:
                ax.plot([sx, tgt['x']], [sz, tgt['z']],
                        '--', color=color, alpha=0.05, linewidth=0.5)

    # Target markers
    if global_targets:
        for t in global_targets:
            tn = t['label'].replace('target_', '')
            ax.scatter(t['x'], t['z'], c=target_colors_map.get(tn, 'gray'),
                       s=150, marker='*', edgecolors='black', linewidths=1, zorder=10)
            ax.annotate(f"T{tn}", (t['x'], t['z']),
                        textcoords='offset points', xytext=(8, 8), fontsize=10,
                        fontweight='bold')

    # Count by target
    counts = defaultdict(int)
    for trace in all_approach_traces:
        counts[trace['chosen']] += 1
    count_text = ', '.join(f"T{k.replace('target_','')}: {v}"
                           for k, v in sorted(counts.items()))
    ax.set_title(f"{dataset_name}: Approach Phase Only (n={len(all_approach_traces)})\n{count_text}")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z / Depth (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Right: full traces with approach highlighted
    ax2 = axes[1]
    for trace in all_approach_traces:
        # Full trace in light gray
        ax2.plot(trace['full_xs'], trace['full_zs'],
                 color='lightgray', alpha=0.1, linewidth=0.5)
        # Approach in color
        chosen_tn = trace['chosen'].replace('target_', '')
        color = target_colors_map.get(chosen_tn, 'gray')
        ax2.plot(trace['approach_xs'], trace['approach_zs'],
                 color=color, alpha=0.2, linewidth=1.0)

    if global_targets:
        for t in global_targets:
            tn = t['label'].replace('target_', '')
            ax2.scatter(t['x'], t['z'], c=target_colors_map.get(tn, 'gray'),
                        s=150, marker='*', edgecolors='black', linewidths=1, zorder=10)
            ax2.annotate(f"T{tn}", (t['x'], t['z']),
                         textcoords='offset points', xytext=(8, 8), fontsize=10,
                         fontweight='bold')

    ax2.set_title(f"{dataset_name}: Full Traces (gray) + Approach (colored)")
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z / Depth (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, linewidth=2, label=f'Target {tn}')
                       for tn, c in sorted(target_colors_map.items())]
    ax2.legend(handles=legend_elements, fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")

    # ─── Plot 2: Per-target approach phase overlays ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, tn in enumerate(['1', '2', '3', '4']):
        ax = axes[idx // 2][idx % 2]
        target_traces = [t for t in all_approach_traces
                         if t['chosen'] == f'target_{tn}']
        color = target_colors_map.get(tn, 'gray')

        for trace in target_traces:
            ax.plot(trace['approach_xs'], trace['approach_zs'],
                    color=color, alpha=0.3, linewidth=1.0)
            # Start point
            ax.scatter(trace['approach_xs'][0], trace['approach_zs'][0],
                       c='green', s=15, alpha=0.3, zorder=3)
            # End point
            ax.scatter(trace['approach_xs'][-1], trace['approach_zs'][-1],
                       c='red', s=15, alpha=0.3, zorder=3)

        if global_targets:
            for t in global_targets:
                t_tn = t['label'].replace('target_', '')
                m = '*' if t_tn == tn else 'o'
                c = target_colors_map.get(t_tn, 'gray') if t_tn == tn else 'lightgray'
                ax.scatter(t['x'], t['z'], c=c, s=120, marker=m,
                           edgecolors='black', linewidths=1, zorder=10)
                ax.annotate(f"T{t_tn}", (t['x'], t['z']),
                            textcoords='offset points', xytext=(5, 5), fontsize=9)

        ax.set_title(f"Target {tn} (n={len(target_traces)})")
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z / Depth (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{dataset_name}: Approach Phase by Target", fontsize=14, y=1.02)
    plt.tight_layout()
    per_target_path = output_path.parent / output_path.name.replace('.png', '_per_target.png')
    plt.savefig(per_target_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {per_target_path}")

    return len(all_approach_traces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccd-only", action="store_true")
    parser.add_argument("--dog-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only regenerate global plots (skip per-trial reprocessing)")
    args = parser.parse_args()

    ANALYSIS_OUTPUT.mkdir(exist_ok=True)

    do_ccd = not args.dog_only
    do_dog = not args.ccd_only

    if not args.plots_only:
        if do_ccd:
            reprocess_ccd()
        if do_dog:
            reprocess_dog()

    # Global approach-phase visualizations
    if do_ccd:
        plot_global_approach_phase(
            "CCD", CCD_OUTPUT,
            "optimal_path_deviation.csv",
            ANALYSIS_OUTPUT / "07_CCD_approach_phase_overlay.png",
            subject_col_pattern="baby")

    if do_dog:
        plot_global_approach_phase(
            "Owner Dog", OWNER_OUTPUT,
            "dog_optimal_path_deviation.csv",
            ANALYSIS_OUTPUT / "07_owner_dog_approach_phase_overlay.png",
            subject_col_pattern="dog")

    # Regenerate reformatted CSV
    if do_ccd and not args.plots_only:
        try:
            from reformat_global_optimal_path import main as reformat_main
            reformat_main()
        except Exception as e:
            print(f"  Reformat error: {e}")

    print("\n" + "=" * 70)
    print("  ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
