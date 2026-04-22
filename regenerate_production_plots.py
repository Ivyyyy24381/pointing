"""
Regenerate all production pointing plots from (possibly trimmed) CSVs
and create consolidated _all_traces/ and _all_distance_plots/ folders.

Works on both CCD and BDL production output directories.

Usage:
  python regenerate_production_plots.py /path/to/output_dir
  python regenerate_production_plots.py  # defaults to BDL output
"""

import ast
import csv
import json
import shutil
import sys
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

POINTING_DIR = Path("/home/tigerli/Documents/GitHub/pointing")
sys.path.insert(0, str(POINTING_DIR))

VECTOR_PREFIXES = [
    'eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist',
    'nose_to_wrist', 'head_orientation',
]


def parse_3d_point(s):
    if not s or s.strip() in ('', 'None', 'nan'):
        return None
    try:
        val = ast.literal_eval(s.strip())
        if isinstance(val, (list, tuple)) and len(val) == 3:
            if any(v is None for v in val):
                return None
            return [float(v) for v in val]
    except (ValueError, SyntaxError):
        pass
    return None


def load_analyses_from_csv(csv_path):
    analyses = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_key = row.get('frame', row.get('global_frame', '0'))
            analysis = {}
            for col in ['wrist_location', 'head_orientation_origin',
                        'head_orientation_vector',
                        'eye_to_wrist_ground_intersection',
                        'shoulder_to_wrist_ground_intersection',
                        'elbow_to_wrist_ground_intersection',
                        'nose_to_wrist_ground_intersection',
                        'ground_intersection',
                        'head_orientation_ground_intersection']:
                pt = parse_3d_point(row.get(col, ''))
                analysis[col] = pt if pt else [None, None, None]
            for col in ['eye_to_wrist_vec', 'shoulder_to_wrist_vec',
                        'elbow_to_wrist_vec', 'nose_to_wrist_vec']:
                pt = parse_3d_point(row.get(col, ''))
                analysis[col] = pt if pt else [None, None, None]
            for prefix in VECTOR_PREFIXES:
                for i in range(1, 5):
                    col = f'{prefix}_dist_to_target_{i}'
                    val = row.get(col, '')
                    try:
                        analysis[col] = float(val) if val and val != 'None' else None
                    except ValueError:
                        analysis[col] = None
            analysis['confidence'] = float(row.get('confidence', 0))
            analyses[frame_key] = analysis
    return analyses


def load_targets(trial_dir):
    tc = trial_dir / "target_coordinates.json"
    if not tc.exists():
        return None
    with open(tc) as f:
        data = json.load(f)
    if isinstance(data, dict) and 'targets' in data:
        data = data['targets']
    if not isinstance(data, list):
        return None
    for t in data:
        if 'world_coords' in t and 'x' not in t:
            wc = t['world_coords']
            t['x'], t['y'], t['z'] = wc[0], wc[1], wc[2]
    return data


def compute_human_pos_from_targets(targets):
    """Compute fixed human position from targets (midpoint of t2/t3 + offset)."""
    t2 = next((t for t in targets if t.get('label') == 'target_2'), None)
    t3 = next((t for t in targets if t.get('label') == 'target_3'), None)
    if t2 and t3:
        x = (t2['x'] + t3['x']) / 2.0
        z = (t2['z'] + t3['z']) / 2.0 + 0.3
        return [x, 0, z]
    return [0, 0, 3]


def regenerate_trial_plots(trial_dir, targets, human_pos):
    """Regenerate pointing trace and distance plots for one trial."""
    from step2_skeleton_extraction.plot_pointing_trace import plot_2d_pointing_trace
    from step2_skeleton_extraction.plot_distance_to_targets import plot_distance_to_targets

    csv_path = trial_dir / "processed_gesture.csv"
    if not csv_path.exists():
        return False

    analyses = load_analyses_from_csv(csv_path)
    if not analyses:
        return False

    try:
        plot_2d_pointing_trace(analyses, targets, human_pos,
                               trial_dir / "2d_pointing_trace.png")
    except Exception as e:
        print(f"      Trace plot error: {e}")

    try:
        plot_distance_to_targets(analyses, targets,
                                 trial_dir / "distance_to_targets.png")
    except Exception as e:
        print(f"      Distance plot error: {e}")

    return True


def create_consolidated_folders(output_dir):
    """Create _all_traces/ and _all_distance_plots/ with all plots renamed."""
    traces_dir = output_dir / "_all_traces"
    dist_dir = output_dir / "_all_distance_plots"

    # Clean existing
    if traces_dir.exists():
        shutil.rmtree(traces_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    traces_dir.mkdir()
    dist_dir.mkdir()

    trace_count = 0
    dist_count = 0

    for subj_dir in sorted(output_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith('_'):
            continue

        for trial_dir in sorted(subj_dir.iterdir()):
            if not trial_dir.is_dir() or not trial_dir.name.startswith('trial_'):
                continue

            # Format: subject__trial.png
            name = f"{subj_dir.name}__{trial_dir.name}.png"

            trace_src = trial_dir / "2d_pointing_trace.png"
            if trace_src.exists():
                shutil.copy2(trace_src, traces_dir / name)
                trace_count += 1

            dist_src = trial_dir / "distance_to_targets.png"
            if dist_src.exists():
                shutil.copy2(dist_src, dist_dir / name)
                dist_count += 1

    print(f"  _all_traces/: {trace_count} plots")
    print(f"  _all_distance_plots/: {dist_count} plots")
    return trace_count, dist_count


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("/home/tigerli/Documents/pointing_data/point_production_BDL_output")

    print(f"Output: {output_dir}")
    print()

    total = 0
    for subj_dir in sorted(output_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith('_'):
            continue

        # Load targets from first trial
        targets = None
        trial_dirs = sorted([d for d in subj_dir.iterdir()
                            if d.is_dir() and d.name.startswith('trial_')])
        for td in trial_dirs:
            targets = load_targets(td)
            if targets:
                break

        if not targets:
            print(f"{subj_dir.name}: no targets found, skipping")
            continue

        human_pos = compute_human_pos_from_targets(targets)
        print(f"{subj_dir.name} ({len(trial_dirs)} trials)")

        for td in trial_dirs:
            trial_targets = load_targets(td) or targets
            if regenerate_trial_plots(td, trial_targets, human_pos):
                total += 1
                print(f"    {td.name}: OK")
            else:
                print(f"    {td.name}: no data")

    print(f"\nRegenerated {total} trial plots")
    print()

    # Create consolidated folders
    print("Creating consolidated folders...")
    create_consolidated_folders(output_dir)


if __name__ == "__main__":
    main()
