#!/usr/bin/env python3
"""
Batch reprocess optimal path deviation using ground truth annotations.

1. Load annotation CSVs (PVPT for CCD, PVPO for owner pointing)
2. For each trial, use ground truth choice_1 instead of auto-detection
3. Use per-trial target_coordinates.json instead of hardcoded targets
4. Recompute deviation, trim to approach phase, regenerate plots
5. Regenerate combined CSVs and global sheet
6. Fix arm mismatches for owner pointing (BDL418, BDL420)
7. Copy updated baby_trace.png to _all_baby_traces/

Usage:
    python batch_reprocess_with_ground_truth.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from process_comprehension_optimal_path import (
    process_trial, CCD_TARGETS, save_combined_deviation_csv,
    save_combined_pose_csv, save_global_csvs, get_deviation_fieldnames,
)

# ─── Paths ───
DATA_ROOT = Path("/home/tigerli/Documents/pointing_data")
CCD_OUTPUT = DATA_ROOT / "point_comprehension_CCD_output"
OWNER_OUTPUT = DATA_ROOT / "owner_pointing_output"
BDL_PROD_OUTPUT = DATA_ROOT / "point_production_BDL_output"

CCD_ANNOTATIONS = DATA_ROOT / "PVPT_Comprehension_Data - Long Data.csv"
OWNER_LONG_DATA = DATA_ROOT / "PVPO_Production_Data - Long_Data.csv"
OWNER_TRACKER = OWNER_OUTPUT / "PVPO_Production_Data - Output_Tracker.csv"


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


def load_owner_arm_annotations():
    """Load arm annotations from Output_Tracker.csv."""
    if not OWNER_TRACKER.exists():
        return {}
    arms = {}
    with open(OWNER_TRACKER) as f:
        for r in csv.DictReader(f):
            if r.get("excluded", "").upper() == "Y":
                continue
            pid = r.get("participant_id", "").strip()
            try:
                trial = int(r.get("trial_number", "0"))
            except ValueError:
                continue
            arm = r.get("Owner Arm Used", "").strip().upper()
            if arm in ("L", "R"):
                arms[(pid, trial)] = "left" if arm == "L" else "right"
    return arms


# ═══════════════════════════════════════════════════════════════
#  CCD Comprehension Reprocessing
# ═══════════════════════════════════════════════════════════════
def reprocess_ccd():
    print("=" * 70)
    print("  CCD COMPREHENSION: Reprocess with Ground Truth")
    print("=" * 70)

    annotations = load_annotations(CCD_ANNOTATIONS, "participant_id", "trial_number")
    print(f"Annotations loaded: {len(annotations)} trials")

    total_ok = 0
    total_skip = 0
    total_no_ann = 0
    results_by_subject = defaultdict(int)

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

            # Load per-trial targets (prefer per-trial, fall back to CCD_TARGETS)
            targets = load_targets_from_json(trial_dir)
            if targets is None:
                targets = CCD_TARGETS

            # Look up ground truth choice_1
            # Try mapping: trial_0 → annotation trial 1 (output is 0-indexed, annotations 1-indexed)
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

            # Load CSV data
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

            if ground_truth_choice is None:
                total_no_ann += 1
                # Still process but with auto-detection
                result = process_trial(rows, targets, output_path, trial_label)
            else:
                result = process_trial(rows, targets, output_path, trial_label,
                                      ground_truth_choice=ground_truth_choice)

            if result is not None:
                total_ok += 1
                subject_ok += 1
            else:
                total_skip += 1

        # Regenerate combined CSVs for this subject
        if subject_ok > 0:
            save_combined_pose_csv(subj_dir, subj_dir.name)
            save_combined_deviation_csv(subj_dir, subj_dir.name, CCD_TARGETS)
            results_by_subject[subj_id] = subject_ok

    # Regenerate global CSVs
    if total_ok > 0:
        print(f"\nRegenerating global CSVs...")
        save_global_csvs(CCD_OUTPUT, CCD_TARGETS)

    # Update _all_baby_traces/
    all_traces_dir = CCD_OUTPUT / "_all_baby_traces"
    if all_traces_dir.exists():
        import shutil
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

    print(f"\n{'='*70}")
    print(f"CCD SUMMARY: {total_ok} OK, {total_skip} skipped, {total_no_ann} no annotation (auto-detected)")
    print(f"{'='*70}")
    return total_ok


# ═══════════════════════════════════════════════════════════════
#  Owner Pointing Arm Fix
# ═══════════════════════════════════════════════════════════════
def fix_owner_arm_mismatches():
    """Fix the 5 arm mismatches by writing corrected pointing_hand.json."""
    print("\n" + "=" * 70)
    print("  OWNER POINTING: Fix Arm Mismatches")
    print("=" * 70)

    arm_annotations = load_owner_arm_annotations()
    mismatches = [
        ("BDL418_Remy_OWN028_PVPO_07_P2_output", "trial_0", "left"),
        ("BDL420_Samara_OWN030_PVP_09_P2_output", "trial_1", "right"),
        ("BDL420_Samara_OWN030_PVP_09_P2_output", "trial_2", "left"),
        ("BDL420_Samara_OWN030_PVP_09_P2_output", "trial_4", "right"),
        ("BDL420_Samara_OWN030_PVP_09_P2_output", "trial_5", "left"),
    ]

    fixed = 0
    for subj_name, trial_name, correct_arm in mismatches:
        ph_path = OWNER_OUTPUT / subj_name / trial_name / "cam1" / "pointing_hand.json"
        if ph_path.exists():
            with open(ph_path) as f:
                ph = json.load(f)

            old_arm = ph.get("pointing_hand", "unknown")
            ph["pointing_hand"] = correct_arm
            ph["override"] = True
            ph["override_reason"] = "Ground truth annotation mismatch fix"

            with open(ph_path, "w") as f:
                json.dump(ph, f, indent=2)

            print(f"  Fixed {subj_name}/{trial_name}: {old_arm} -> {correct_arm}")
            fixed += 1
        else:
            print(f"  WARNING: {ph_path} not found")

    print(f"\nFixed {fixed} arm mismatches")

    # These trials need pointing reprocessing
    if fixed > 0:
        print("\nNOTE: These trials need pointing analysis reprocessed with the corrected arm.")
        print("Run batch_postprocess.py with --pointing-only for these subjects.")

    return fixed


# ═══════════════════════════════════════════════════════════════
#  Regenerate Global Reformatted CSV
# ═══════════════════════════════════════════════════════════════
def regenerate_global_csv():
    """Re-run the global CSV reformatter."""
    print("\n" + "=" * 70)
    print("  Regenerating Global Reformatted CSV")
    print("=" * 70)

    # Import and run the reformatter
    try:
        from reformat_global_optimal_path import main as reformat_main
        reformat_main()
    except Exception as e:
        print(f"  Error running reformatter: {e}")
        print("  You can run it manually: python reformat_global_optimal_path.py")


# ═══════════════════════════════════════════════════════════════
#  Re-run Analysis Visualizations
# ═══════════════════════════════════════════════════════════════
def rerun_analysis():
    """Re-run the comprehensive analysis to generate updated plots."""
    print("\n" + "=" * 70)
    print("  Re-running Comprehensive Analysis")
    print("=" * 70)

    try:
        from comprehensive_trace_analysis import main as analysis_main
        analysis_main()
    except Exception as e:
        print(f"  Error running analysis: {e}")
        print("  You can run it manually: python comprehensive_trace_analysis.py")


def main():
    print("Batch Reprocess with Ground Truth Annotations")
    print("=" * 70)

    # Step 1: Reprocess CCD with ground truth first choice
    ccd_ok = reprocess_ccd()

    # Step 2: Fix arm mismatches
    arm_fixed = fix_owner_arm_mismatches()

    # Step 3: Regenerate global CSV
    regenerate_global_csv()

    # Step 4: Re-run analysis visualizations
    rerun_analysis()

    print("\n" + "=" * 70)
    print("  ALL DONE")
    print(f"  CCD trials reprocessed: {ccd_ok}")
    print(f"  Arm mismatches fixed: {arm_fixed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
