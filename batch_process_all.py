"""
Batch process ALL studies in a data folder.

Iterates over every subfolder in the given directory that looks like a study
(contains trial_N subdirectories) and runs batch_process_study on each.

Usage:
    python batch_process_all.py /home/tigerli/Documents/pointing_data
    python batch_process_all.py /home/tigerli/Documents/pointing_data --use-sam3
    python batch_process_all.py /home/tigerli/Documents/pointing_data --subject dog --use-sam3 --cameras cam1
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from batch_process_study import process_study


def find_study_folders(data_dir):
    """Find all folders that contain trial_N subdirectories (i.e., are studies)."""
    data_dir = Path(data_dir)
    studies = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        # Skip output folders
        if d.name.endswith('_output'):
            continue
        # Check if it contains at least one trial_N folder
        has_trials = any(
            t.is_dir() and t.name.startswith('trial_')
            for t in d.iterdir()
        )
        if has_trials:
            studies.append(d)
    return studies


def main():
    parser = argparse.ArgumentParser(
        description="Batch process ALL study folders in a data directory."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to data directory containing study folders (e.g., /home/tigerli/Documents/pointing_data)"
    )
    parser.add_argument(
        "--trial", type=str, default=None,
        help="Process only this trial in each study (e.g., trial_6)"
    )
    parser.add_argument(
        "--cameras", type=str, nargs='+', default=None,
        help="Process only these cameras (e.g., cam1 cam2)"
    )
    parser.add_argument(
        "--skip-targets", action="store_true",
        help="Skip target detection if results already exist"
    )
    parser.add_argument(
        "--target-frame", type=int, default=1,
        help="Frame number to use for target detection (default: 1)"
    )
    parser.add_argument(
        "--subject", type=str, default='dog', choices=['dog', 'baby', 'none'],
        help="Subject type to detect (default: dog). Use 'none' to skip."
    )
    parser.add_argument(
        "--use-sam3", action="store_true",
        help="Use SAM3 for subject detection instead of DLC/MediaPipe"
    )
    parser.add_argument(
        "--target-depth", type=float, default=None,
        help="Known target depth in meters (e.g., 4.0). Use when you know the "
             "distance from your experiment setup."
    )
    parser.add_argument(
        "--skip-ground-rotation", action="store_true",
        help="Skip ground plane rotation entirely (use raw camera coordinates). "
             "Useful when targets are on a curved arc and rotation would distort the shape."
    )

    args = parser.parse_args()
    subject_type = args.subject if args.subject != 'none' else None

    studies = find_study_folders(args.data_dir)

    if not studies:
        print(f"No study folders found in {args.data_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Found {len(studies)} study folder(s):")
    for s in studies:
        print(f"    - {s.name}")
    print(f"{'='*60}\n")

    for i, study_path in enumerate(studies, 1):
        print(f"\n{'#'*60}")
        print(f"  [{i}/{len(studies)}] Processing: {study_path.name}")
        print(f"{'#'*60}")

        try:
            process_study(
                study_path,
                trial_filter=args.trial,
                camera_filter=args.cameras,
                skip_targets=args.skip_targets,
                target_frame=args.target_frame,
                subject_type=subject_type,
                use_sam3=args.use_sam3,
                expected_target_depth=args.target_depth,
                skip_ground_rotation=args.skip_ground_rotation,
            )
        except Exception as e:
            print(f"\n  ERROR processing {study_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"  ALL DONE: Processed {len(studies)} studies")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
