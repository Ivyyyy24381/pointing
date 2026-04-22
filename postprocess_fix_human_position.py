"""
Post-process CCD production gesture CSVs to fix the human position.

The human (adult pointer) is fixed to a canonical position:
  x = midpoint of target_2 and target_3
  z = midpoint of target_2 and target_3 z + offset (slightly behind/further from camera)

Since all arm vectors are normalized direction vectors (differences between landmarks),
they are invariant to translation. When shifting all 3D landmarks by (Δx, 0, Δz):
  - All ground intersections shift by exactly (Δx, 0, Δz)
  - Distances to targets change and are recomputed from the shifted intersections

Usage:
  python postprocess_fix_human_position.py [output_dir] [--behind METERS]
"""

import ast
import csv
import json
import shutil
import sys
import numpy as np
from pathlib import Path

# How far behind (further from camera, +z) the midpoint of target 2 and 3
DEFAULT_BEHIND_OFFSET = 0.3  # meters


def parse_3d_point(s):
    """Parse a string like '(-0.4, 0.87, 1.2)' or '[-0.4, 0.87, 1.2]' or '[None, None, None]'."""
    if not s or s.strip() in ('', 'None', 'nan'):
        return None
    s = s.strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)) and len(val) == 3:
            if any(v is None for v in val):
                return None
            return [float(v) for v in val]
    except (ValueError, SyntaxError):
        pass
    return None


def format_3d_point(pt, fmt="list"):
    """Format a 3D point back to string."""
    if pt is None:
        return "[None, None, None]"
    if fmt == "tuple":
        return f"({pt[0]}, {pt[1]}, {pt[2]})"
    return f"[{pt[0]}, {pt[1]}, {pt[2]}]"


def parse_landmarks(s):
    """Parse the landmarks column (list of 33 tuples)."""
    if not s or s.strip() in ('', 'None', 'nan'):
        return None
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)) and len(val) >= 25:
            return [list(v) for v in val]
    except (ValueError, SyntaxError):
        pass
    return None


def compute_hip_center(landmarks):
    """Get hip center (midpoint of landmarks 23 and 24) from 3D landmarks."""
    if landmarks is None or len(landmarks) < 25:
        return None
    lh = landmarks[23]
    rh = landmarks[24]
    hc = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2]
    if all(v == 0 for v in hc):
        return None
    return hc


def load_targets(trial_dir):
    """Load target coordinates from JSON. Normalizes to have x, y, z keys."""
    tc = trial_dir / "target_coordinates.json"
    if not tc.exists():
        return None
    with open(tc) as f:
        data = json.load(f)
    # Handle nested structure: {"targets": [...]}
    if isinstance(data, dict) and 'targets' in data:
        data = data['targets']
    if not isinstance(data, list):
        return None
    # Normalize: ensure each target has x, y, z keys
    for t in data:
        if 'world_coords' in t and 'x' not in t:
            wc = t['world_coords']
            t['x'], t['y'], t['z'] = wc[0], wc[1], wc[2]
    return data


def compute_fixed_position(targets, behind_offset=DEFAULT_BEHIND_OFFSET):
    """
    Compute the fixed human position:
      x = midpoint of target_2 and target_3 x
      z = midpoint of target_2 and target_3 z + behind_offset
    """
    t2 = next((t for t in targets if t.get('label') == 'target_2'), None)
    t3 = next((t for t in targets if t.get('label') == 'target_3'), None)
    if t2 is None or t3 is None:
        # Fallback: use middle two targets by z-sorted order
        sorted_t = sorted(targets, key=lambda t: t['z'])
        if len(sorted_t) >= 3:
            t2, t3 = sorted_t[1], sorted_t[2]
        elif len(sorted_t) >= 2:
            t2, t3 = sorted_t[0], sorted_t[1]
        else:
            return None

    fixed_x = (t2['x'] + t3['x']) / 2.0
    fixed_z = (t2['z'] + t3['z']) / 2.0 + behind_offset
    return fixed_x, fixed_z


def compute_distance(pt, target):
    """Euclidean distance from 3D point to target."""
    tx = target['x']
    ty = target['y']
    tz = target['z']
    return np.sqrt((pt[0]-tx)**2 + (pt[1]-ty)**2 + (pt[2]-tz)**2)


# Column groups
INTERSECTION_COLS = [
    'eye_to_wrist_ground_intersection',
    'shoulder_to_wrist_ground_intersection',
    'elbow_to_wrist_ground_intersection',
    'nose_to_wrist_ground_intersection',
    'ground_intersection',
    'head_orientation_ground_intersection',
]

VECTOR_PREFIXES = [
    'eye_to_wrist',
    'shoulder_to_wrist',
    'elbow_to_wrist',
    'nose_to_wrist',
    'head_orientation',
]


def process_trial(trial_dir, targets, fixed_x, fixed_z):
    """Process one trial CSV: shift intersections and recompute distances."""
    csv_path = trial_dir / "processed_gesture.csv"
    if not csv_path.exists():
        return False

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        return False

    n_shifted = 0
    for row in rows:
        # Parse landmarks to get current hip center
        landmarks = parse_landmarks(row.get('landmarks', ''))
        hip = compute_hip_center(landmarks)

        if hip is None:
            # Can't compute offset — skip this frame (leave as-is)
            continue

        # Compute per-frame offset
        dx = fixed_x - hip[0]
        dz = fixed_z - hip[2]

        # Shift wrist_location
        wrist = parse_3d_point(row.get('wrist_location', ''))
        if wrist is not None:
            wrist[0] += dx
            wrist[2] += dz
            row['wrist_location'] = format_3d_point(wrist, fmt="tuple")

        # Shift head_orientation_origin
        ho_origin = parse_3d_point(row.get('head_orientation_origin', ''))
        if ho_origin is not None:
            ho_origin[0] += dx
            ho_origin[2] += dz
            row['head_orientation_origin'] = format_3d_point(ho_origin, fmt="tuple")

        # Shift all ground intersection columns and recompute distances
        for prefix in VECTOR_PREFIXES:
            int_col = f'{prefix}_ground_intersection'
            pt = parse_3d_point(row.get(int_col, ''))
            if pt is not None:
                pt[0] += dx
                pt[2] += dz
                row[int_col] = format_3d_point(pt)

                # Recompute distances to each target
                for t in targets:
                    tn = t['label'].replace('target_', '')
                    dist_col = f'{prefix}_dist_to_target_{tn}'
                    if dist_col in row:
                        row[dist_col] = str(compute_distance(pt, t))

        # Also shift the default 'ground_intersection' column
        gi = parse_3d_point(row.get('ground_intersection', ''))
        if gi is not None:
            gi[0] += dx
            gi[2] += dz
            row['ground_intersection'] = format_3d_point(gi)

        # Shift landmarks too (for consistency and future re-processing)
        if landmarks is not None:
            for lm in landmarks:
                lm[0] += dx
                lm[2] += dz
            row['landmarks'] = str([tuple(lm) for lm in landmarks])

        n_shifted += 1

    # Write back
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return n_shifted > 0


def load_analyses_from_csv(csv_path):
    """Load analyses_dict from a processed_gesture.csv file.
    Returns {frame_key: analysis_dict} suitable for plot functions.
    """
    analyses = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_key = row.get('frame', row.get('global_frame', '0'))
            analysis = {}

            # Parse 3D point columns
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

            # Parse vector columns
            for col in ['eye_to_wrist_vec', 'shoulder_to_wrist_vec',
                        'elbow_to_wrist_vec', 'nose_to_wrist_vec']:
                pt = parse_3d_point(row.get(col, ''))
                analysis[col] = pt if pt else [None, None, None]

            # Parse distance columns (floats)
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


def regenerate_plot(trial_dir, targets, fixed_pos):
    """Regenerate the 2D pointing trace plot with fixed human position."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from step2_skeleton_extraction.plot_pointing_trace import plot_2d_pointing_trace

        csv_path = trial_dir / "processed_gesture.csv"
        if not csv_path.exists():
            return

        analyses = load_analyses_from_csv(csv_path)
        human_pos = [fixed_pos[0], 0, fixed_pos[1]]  # [x, y, z]
        plot_2d_pointing_trace(analyses, targets, human_pos,
                               trial_dir / "2d_pointing_trace.png")
    except Exception as e:
        print(f"    Plot warning: {e}")


def regenerate_distance_plot(trial_dir, targets):
    """Regenerate the distance-to-targets plot."""
    try:
        from step2_skeleton_extraction.plot_distance_to_targets import plot_distance_to_targets

        csv_path = trial_dir / "processed_gesture.csv"
        if not csv_path.exists():
            return

        analyses = load_analyses_from_csv(csv_path)
        plot_distance_to_targets(analyses, targets,
                                 trial_dir / "distance_to_targets.png")
    except Exception as e:
        print(f"    Distance plot warning: {e}")


def process_subject(subj_dir, behind_offset=DEFAULT_BEHIND_OFFSET):
    """Process all trials for one subject."""
    # Load targets from first trial that has them
    targets = None
    trial_dirs = sorted([d for d in subj_dir.iterdir()
                        if d.is_dir() and d.name.startswith('trial_')])
    for td in trial_dirs:
        targets = load_targets(td)
        if targets:
            break

    if not targets:
        print(f"  SKIP: No target coordinates found")
        return 0

    fixed = compute_fixed_position(targets, behind_offset)
    if fixed is None:
        print(f"  SKIP: Cannot compute fixed position")
        return 0

    fixed_x, fixed_z = fixed
    t2 = next((t for t in targets if t.get('label') == 'target_2'), None)
    t3 = next((t for t in targets if t.get('label') == 'target_3'), None)
    print(f"  Fixed position: x={fixed_x:.3f}, z={fixed_z:.3f}")
    if t2 and t3:
        print(f"  (target_2: x={t2['x']:.3f} z={t2['z']:.3f}, "
              f"target_3: x={t3['x']:.3f} z={t3['z']:.3f})")

    ok = 0
    for td in trial_dirs:
        trial_targets = load_targets(td) or targets
        if process_trial(td, trial_targets, fixed_x, fixed_z):
            regenerate_plot(td, trial_targets, (fixed_x, fixed_z))
            regenerate_distance_plot(td, trial_targets)
            ok += 1
            print(f"    {td.name}: OK")
        else:
            print(f"    {td.name}: no data")

    return ok


def regenerate_combined_csv(subj_dir):
    """Regenerate combined_gesture.csv for a subject."""
    trial_dirs = sorted([d for d in subj_dir.iterdir()
                        if d.is_dir() and d.name.startswith('trial_')])

    all_rows = []
    fieldnames = None
    for td in trial_dirs:
        csv_path = td / "processed_gesture.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                row['trial'] = td.name
                all_rows.append(row)

    if not all_rows or fieldnames is None:
        return 0

    combined_fields = ['trial'] + fieldnames
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
                row['subject'] = subj_dir.name
                all_rows.append(row)

    if not all_rows or fieldnames is None:
        return 0

    global_fields = ['subject'] + fieldnames
    global_path = output_dir / "all_subjects_gesture.csv"
    with open(global_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=global_fields,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Global gesture CSV: {len(all_rows)} rows")
    return len(all_rows)


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("/home/tigerli/Documents/pointing_data/point_production_CCD_output")

    behind = DEFAULT_BEHIND_OFFSET
    if '--behind' in sys.argv:
        idx = sys.argv.index('--behind')
        behind = float(sys.argv[idx + 1])

    print(f"Output: {output_dir}")
    print(f"Behind offset: {behind}m")
    print()

    total_ok = 0
    for subj_dir in sorted(output_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith('_'):
            continue
        print(f"{subj_dir.name}")
        ok = process_subject(subj_dir, behind)
        if ok > 0:
            n_combined = regenerate_combined_csv(subj_dir)
            print(f"  Combined: {n_combined} rows")
        total_ok += ok

    print(f"\nTotal: {total_ok} trials fixed")

    # Regenerate global CSV
    generate_global_csv(output_dir)


if __name__ == "__main__":
    main()
