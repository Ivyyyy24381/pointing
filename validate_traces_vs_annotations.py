#!/usr/bin/env python3
"""
Validate dog and baby traces against human annotations.

Checks:
1. CCD Comprehension: baby trace first-choice vs annotated choice_1
2. Owner Pointing (PVPO): dog trace first-choice vs annotated choice_1
3. Trace quality: reasonable distances, sufficient frames, plausible paths

Reports per-trial and per-subject agreement + data quality issues.

Usage:
    python validate_traces_vs_annotations.py
"""

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ─── Paths ───
CCD_OUTPUT = Path("/home/tigerli/Documents/pointing_data/point_comprehension_CCD_output")
CCD_ANNOTATIONS = Path("/home/tigerli/Documents/pointing_data/PVPT_Comprehension_Data - Long Data.csv")

OWNER_OUTPUT = Path("/home/tigerli/Documents/pointing_data/owner_pointing_output")
OWNER_ANNOTATIONS = Path("/home/tigerli/Documents/pointing_data/PVPO_Production_Data - Long_Data.csv")

BDL_PROD_OUTPUT = Path("/home/tigerli/Documents/pointing_data/point_production_BDL_output")
BDL_COMP_ANNOTATIONS = Path("/home/tigerli/Documents/pointing_data/point_comprehension_BDL_output/PVP_Comprehension_Data - Long Data.csv")


def load_annotations(csv_path, id_col="participant_id", trial_col="trial_number"):
    """Load annotations, return dict of {(participant_id, trial_num): row}."""
    if not csv_path.exists():
        return {}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    result = {}
    for r in rows:
        if r.get("excluded", "").upper() == "Y":
            continue
        pid = r[id_col]
        trial = int(r[trial_col])
        result[(pid, trial)] = r
    return result


def find_closest_target_from_trace(csv_path, method="min_distance"):
    """
    From a trace CSV, determine which target the subject approached first.
    Returns dict with first_choice, min_distances, trace_quality info.
    """
    if not csv_path.exists():
        return None

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    # Detect CSV format
    has_trace3d = "trace3d_x" in rows[0]
    has_target_r = "target_1_r" in rows[0]

    if not has_target_r:
        return None

    n_total = len(rows)
    n_valid = 0
    target_min_dists = {1: float("inf"), 2: float("inf"),
                        3: float("inf"), 4: float("inf")}
    target_min_frame = {1: -1, 2: -1, 3: -1, 4: -1}

    # Track position variance for quality check
    xs, zs = [], []
    dists_over_time = {1: [], 2: [], 3: [], 4: []}

    for i, row in enumerate(rows):
        if has_trace3d:
            x = row.get("trace3d_x", "")
            z = row.get("trace3d_z", "")
            try:
                x, z = float(x), float(z)
                if not (math.isnan(x) or math.isnan(z)):
                    xs.append(x)
                    zs.append(z)
                    n_valid += 1
            except (ValueError, TypeError):
                pass

        for t in range(1, 5):
            col = f"target_{t}_r"
            val = row.get(col, "")
            try:
                d = float(val)
                if not math.isnan(d):
                    dists_over_time[t].append(d)
                    if d < target_min_dists[t]:
                        target_min_dists[t] = d
                        target_min_frame[t] = i
            except (ValueError, TypeError):
                pass

    # Determine first choice (target reached closest, min distance < threshold)
    first_choice = None
    first_choice_dist = float("inf")
    approach_threshold = 0.5  # must get within 0.5m

    # Method: which target had the earliest minimum distance < threshold
    earliest_close = {}
    for t in range(1, 5):
        dists = dists_over_time[t]
        for i, d in enumerate(dists):
            if d < approach_threshold:
                earliest_close[t] = i
                break

    if earliest_close:
        first_choice = min(earliest_close, key=earliest_close.get)
        first_choice_dist = target_min_dists[first_choice]
    else:
        # Fallback: just pick the target with overall minimum distance
        first_choice = min(target_min_dists, key=target_min_dists.get)
        first_choice_dist = target_min_dists[first_choice]

    # Quality metrics
    x_range = (max(xs) - min(xs)) if xs else 0
    z_range = (max(zs) - min(zs)) if zs else 0
    total_motion = x_range + z_range

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "first_choice": first_choice,
        "first_choice_dist": first_choice_dist,
        "min_dists": target_min_dists,
        "x_range": x_range,
        "z_range": z_range,
        "total_motion": total_motion,
        "xs": xs,
        "zs": zs,
    }


# ═══════════════════════════════════════════════════════════
#  CCD COMPREHENSION (Baby Traces)
# ═══════════════════════════════════════════════════════════
def validate_ccd_comprehension():
    print("=" * 70)
    print("  CCD COMPREHENSION: Baby Trace vs Annotation")
    print("=" * 70)

    annotations = load_annotations(CCD_ANNOTATIONS, "participant_id", "trial_number")
    print(f"Annotations: {len(annotations)} non-excluded trials")

    # Map output folders to CCD IDs
    folder_map = {}
    for d in sorted(CCD_OUTPUT.iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            ccd_id = d.name.split("_")[0]
            folder_map[ccd_id] = d

    match = 0
    mismatch = 0
    no_data = 0
    no_annotation = 0
    quality_issues = []
    results_by_subject = defaultdict(lambda: {"match": 0, "mismatch": 0, "no_data": 0})

    print(f"\n{'Subject':<35} {'Trial':<10} {'Ann c1':<8} {'Trace c1':<10} {'MinDist':<8} {'Motion':<8} {'Status'}")
    print("-" * 105)

    for ccd_id, subj_dir in sorted(folder_map.items()):
        trial_dirs = sorted([d for d in subj_dir.iterdir()
                            if d.is_dir() and d.name.startswith("trial_")])

        for trial_dir in trial_dirs:
            trial_name = trial_dir.name
            trial_num = int(trial_name.replace("trial_", ""))

            # Annotation uses 1-based trial numbers
            # Output uses trial_0, trial_1, etc.
            # The CSV annotation trial_number corresponds to output trial_(N-1)
            ann_key = (ccd_id, trial_num + 1)  # try +1 mapping
            ann = annotations.get(ann_key)
            if ann is None:
                ann_key = (ccd_id, trial_num)
                ann = annotations.get(ann_key)

            csv_path = trial_dir / "processed_subject_result_table.csv"
            trace = find_closest_target_from_trace(csv_path)

            if trace is None:
                no_data += 1
                results_by_subject[ccd_id]["no_data"] += 1
                continue

            ann_c1 = int(ann["choice_1"]) if ann and ann.get("choice_1", "NA") not in ("NA", "") else None
            trace_c1 = trace["first_choice"]
            min_dist = trace["first_choice_dist"]
            motion = trace["total_motion"]

            if ann is None:
                status = "NO_ANN"
                no_annotation += 1
            elif ann_c1 is None:
                status = "ANN_NA"
                no_annotation += 1
            elif ann_c1 == trace_c1:
                status = "MATCH"
                match += 1
                results_by_subject[ccd_id]["match"] += 1
            else:
                status = f"MISMATCH (ann={ann_c1})"
                mismatch += 1
                results_by_subject[ccd_id]["mismatch"] += 1

            # Quality flags
            flags = []
            if min_dist > 1.0:
                flags.append(f"far({min_dist:.1f}m)")
            if motion < 0.3:
                flags.append(f"low_motion({motion:.2f})")
            if trace["n_valid"] < 20:
                flags.append(f"few_frames({trace['n_valid']})")

            flag_str = " " + ",".join(flags) if flags else ""

            print(f"{subj_dir.name:<35} {trial_name:<10} {str(ann_c1):<8} {trace_c1:<10} {min_dist:<8.3f} {motion:<8.2f} {status}{flag_str}")

            if flags:
                quality_issues.append((subj_dir.name, trial_name, flags))

    total = match + mismatch
    pct = 100 * match / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"CCD COMPREHENSION SUMMARY:")
    print(f"  Match: {match}/{total} ({pct:.1f}%)")
    print(f"  Mismatch: {mismatch}/{total}")
    print(f"  No trace data: {no_data}")
    print(f"  No annotation: {no_annotation}")
    print(f"  Quality issues: {len(quality_issues)} trials")

    if quality_issues:
        print(f"\nQuality Issues:")
        for subj, trial, flags in quality_issues[:20]:
            print(f"  {subj}/{trial}: {', '.join(flags)}")

    print(f"\nPer-Subject:")
    for ccd_id, stats in sorted(results_by_subject.items()):
        total_s = stats["match"] + stats["mismatch"]
        pct_s = 100 * stats["match"] / total_s if total_s > 0 else 0
        print(f"  {ccd_id}: {stats['match']}/{total_s} match ({pct_s:.0f}%), {stats['no_data']} no_data")

    return match, mismatch, no_data


# ═══════════════════════════════════════════════════════════
#  OWNER POINTING (Dog Traces via PVPO)
# ═══════════════════════════════════════════════════════════
def validate_owner_pointing():
    print("\n" + "=" * 70)
    print("  OWNER POINTING: Dog Trace vs Annotation")
    print("=" * 70)

    annotations = load_annotations(OWNER_ANNOTATIONS, "participant_id", "trial")
    print(f"Annotations: {len(annotations)} non-excluded trials")

    # Map output folders to BDL IDs
    folder_map = {}
    for d in sorted(OWNER_OUTPUT.iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            bdl_id = d.name.split("_")[0]
            folder_map[bdl_id] = d

    print(f"Output folders: {len(folder_map)}")

    match = 0
    mismatch = 0
    no_data = 0
    no_annotation = 0
    quality_issues = []

    print(f"\n{'Subject':<50} {'Trial':<10} {'Ann c1':<8} {'Trace c1':<10} {'MinDist':<8} {'Motion':<8} {'Status'}")
    print("-" * 115)

    for bdl_id, subj_dir in sorted(folder_map.items()):
        trial_dirs = sorted([d for d in subj_dir.iterdir()
                            if d.is_dir() and d.name.startswith("trial_")])

        for trial_dir in trial_dirs:
            trial_name = trial_dir.name
            trial_num = int(trial_name.replace("trial_", ""))

            ann_key = (bdl_id, trial_num + 1)
            ann = annotations.get(ann_key)
            if ann is None:
                ann_key = (bdl_id, trial_num)
                ann = annotations.get(ann_key)

            # Dog trace CSV is in cam1/ subfolder
            csv_path = trial_dir / "cam1" / "processed_dog_result_table.csv"
            trace = find_closest_target_from_trace(csv_path)

            if trace is None:
                no_data += 1
                continue

            ann_c1 = int(ann["choice_1"]) if ann and ann.get("choice_1", "NA") not in ("NA", "") else None
            trace_c1 = trace["first_choice"]
            min_dist = trace["first_choice_dist"]
            motion = trace["total_motion"]

            if ann is None:
                status = "NO_ANN"
                no_annotation += 1
            elif ann_c1 is None:
                status = "ANN_NA"
                no_annotation += 1
            elif ann_c1 == trace_c1:
                status = "MATCH"
                match += 1
            else:
                status = f"MISMATCH (ann={ann_c1})"
                mismatch += 1

            flags = []
            if min_dist > 1.0:
                flags.append(f"far({min_dist:.1f}m)")
            if motion < 0.3:
                flags.append(f"low_motion({motion:.2f})")
            if trace["n_valid"] < 20:
                flags.append(f"few_frames({trace['n_valid']})")

            flag_str = " " + ",".join(flags) if flags else ""

            print(f"{subj_dir.name:<50} {trial_name:<10} {str(ann_c1):<8} {trace_c1:<10} {min_dist:<8.3f} {motion:<8.2f} {status}{flag_str}")

            if flags:
                quality_issues.append((subj_dir.name, trial_name, flags))

    total = match + mismatch
    pct = 100 * match / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"OWNER POINTING SUMMARY:")
    print(f"  Match: {match}/{total} ({pct:.1f}%)")
    print(f"  Mismatch: {mismatch}/{total}")
    print(f"  No trace data: {no_data}")
    print(f"  No annotation: {no_annotation}")
    print(f"  Quality issues: {len(quality_issues)} trials")

    return match, mismatch, no_data


# ═══════════════════════════════════════════════════════════
#  TRACE QUALITY SUMMARY
# ═══════════════════════════════════════════════════════════
def summarize_trace_quality(output_dir, csv_name, label):
    """Summarize trace quality across all trials in an output directory."""
    print(f"\n{'='*70}")
    print(f"  TRACE QUALITY: {label}")
    print(f"{'='*70}")

    total_trials = 0
    valid_trials = 0
    motions = []
    min_dists_all = []
    frame_counts = []
    issues = []

    for subj_dir in sorted(output_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith("_"):
            continue

        for trial_dir in sorted(subj_dir.iterdir()):
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue

            total_trials += 1

            # Try different CSV locations
            csv_path = trial_dir / csv_name
            if not csv_path.exists():
                csv_path = trial_dir / "cam1" / csv_name
            if not csv_path.exists():
                issues.append((subj_dir.name, trial_dir.name, "NO_CSV"))
                continue

            trace = find_closest_target_from_trace(csv_path)
            if trace is None:
                issues.append((subj_dir.name, trial_dir.name, "EMPTY_CSV"))
                continue

            valid_trials += 1
            motions.append(trace["total_motion"])
            min_dists_all.append(trace["first_choice_dist"])
            frame_counts.append(trace["n_total"])

            if trace["total_motion"] < 0.1:
                issues.append((subj_dir.name, trial_dir.name,
                              f"STATIC (motion={trace['total_motion']:.3f}m)"))
            elif trace["first_choice_dist"] > 2.0:
                issues.append((subj_dir.name, trial_dir.name,
                              f"FAR (min_dist={trace['first_choice_dist']:.2f}m)"))

    print(f"Total trials: {total_trials}")
    print(f"Valid traces: {valid_trials}")
    if motions:
        print(f"Motion (m): mean={np.mean(motions):.2f}, "
              f"median={np.median(motions):.2f}, "
              f"min={np.min(motions):.3f}, max={np.max(motions):.2f}")
        print(f"Min distance to any target (m): mean={np.mean(min_dists_all):.2f}, "
              f"median={np.median(min_dists_all):.2f}, "
              f"min={np.min(min_dists_all):.3f}, max={np.max(min_dists_all):.2f}")
        print(f"Frame counts: mean={np.mean(frame_counts):.0f}, "
              f"median={np.median(frame_counts):.0f}, "
              f"min={np.min(frame_counts)}, max={np.max(frame_counts)}")

    if issues:
        print(f"\nIssues ({len(issues)}):")
        for subj, trial, issue in issues[:30]:
            print(f"  {subj}/{trial}: {issue}")


# ═══════════════════════════════════════════════════════════
#  ANNOTATION ANALYSIS (for trial trimming)
# ═══════════════════════════════════════════════════════════
def analyze_annotations():
    print("\n" + "=" * 70)
    print("  ANNOTATION ANALYSIS (for trial trimming)")
    print("=" * 70)

    # CCD comprehension
    if CCD_ANNOTATIONS.exists():
        with open(CCD_ANNOTATIONS) as f:
            rows = list(csv.DictReader(f))
        non_excl = [r for r in rows if r.get("excluded", "").upper() != "Y"]
        print(f"\nCCD Comprehension: {len(non_excl)} non-excluded trials")

        choices_req = [int(r["choices_required"]) for r in non_excl
                      if r.get("choices_required", "NA") not in ("NA", "")]
        c1_matches_loc = sum(1 for r in non_excl
                            if r.get("choice_1", "NA") not in ("NA", "")
                            and r.get("location", "") != ""
                            and r["choice_1"] == r["location"])
        c1_total = sum(1 for r in non_excl
                      if r.get("choice_1", "NA") not in ("NA", ""))

        print(f"  Choices required: mean={np.mean(choices_req):.1f}, "
              f"median={np.median(choices_req):.0f}, "
              f"1-choice={choices_req.count(1)}, "
              f"2-choice={choices_req.count(2)}, "
              f"3+={sum(1 for c in choices_req if c >= 3)}")
        print(f"  First choice = correct: {c1_matches_loc}/{c1_total} "
              f"({100*c1_matches_loc/max(c1_total,1):.0f}%)")

        # Distribution of choice_1 by location
        loc_c1 = defaultdict(lambda: defaultdict(int))
        for r in non_excl:
            loc = r.get("location", "")
            c1 = r.get("choice_1", "NA")
            if loc and c1 not in ("NA", ""):
                loc_c1[loc][c1] += 1

        print(f"  Choice_1 distribution by location:")
        for loc in sorted(loc_c1.keys()):
            dist = dict(loc_c1[loc])
            total = sum(dist.values())
            correct = dist.get(loc, 0)
            print(f"    Location {loc}: {dist} (correct={correct}/{total}={100*correct/max(total,1):.0f}%)")

    # Owner/BDL production
    if OWNER_ANNOTATIONS.exists():
        with open(OWNER_ANNOTATIONS) as f:
            rows = list(csv.DictReader(f))
        non_excl = [r for r in rows if r.get("excluded", "").upper() != "Y"]
        print(f"\nOwner Production (PVPO): {len(non_excl)} non-excluded trials")

        choices_req = [int(r["choices_required"]) for r in non_excl
                      if r.get("choices_required", "NA") not in ("NA", "")]
        c1_matches_loc = sum(1 for r in non_excl
                            if r.get("choice_1", "NA") not in ("NA", "")
                            and r.get("location", "") != ""
                            and r["choice_1"] == r["location"])
        c1_total = sum(1 for r in non_excl
                      if r.get("choice_1", "NA") not in ("NA", ""))

        print(f"  Choices required: mean={np.mean(choices_req):.1f}, "
              f"median={np.median(choices_req):.0f}, "
              f"1-choice={choices_req.count(1)}, "
              f"2-choice={choices_req.count(2)}, "
              f"3+={sum(1 for c in choices_req if c >= 3)}")
        print(f"  First choice = correct: {c1_matches_loc}/{c1_total} "
              f"({100*c1_matches_loc/max(c1_total,1):.0f}%)")

    # BDL comprehension
    if BDL_COMP_ANNOTATIONS.exists():
        with open(BDL_COMP_ANNOTATIONS) as f:
            rows = list(csv.DictReader(f))
        non_excl = [r for r in rows if r.get("excluded", "").upper() != "Y"]
        print(f"\nBDL Comprehension (PVP): {len(non_excl)} non-excluded trials")

        choices_req = [int(r["choices_required"]) for r in non_excl
                      if r.get("choices_required", "NA") not in ("NA", "")]
        c1_matches_loc = sum(1 for r in non_excl
                            if r.get("choice_1", "NA") not in ("NA", "")
                            and r.get("location", "") != ""
                            and r["choice_1"] == r["location"])
        c1_total = sum(1 for r in non_excl
                      if r.get("choice_1", "NA") not in ("NA", ""))

        print(f"  Choices required: mean={np.mean(choices_req):.1f}, "
              f"median={np.median(choices_req):.0f}, "
              f"1-choice={choices_req.count(1)}, "
              f"2-choice={choices_req.count(2)}, "
              f"3+={sum(1 for c in choices_req if c >= 3)}")
        print(f"  First choice = correct: {c1_matches_loc}/{c1_total} "
              f"({100*c1_matches_loc/max(c1_total,1):.0f}%)")


def main():
    # 1. Trace quality summaries
    summarize_trace_quality(CCD_OUTPUT, "processed_subject_result_table.csv",
                           "CCD Comprehension (Baby)")
    summarize_trace_quality(OWNER_OUTPUT, "processed_dog_result_table.csv",
                           "Owner Pointing (Dog)")

    # 2. Trace vs annotation validation
    validate_ccd_comprehension()
    validate_owner_pointing()

    # 3. Annotation analysis for trimming
    analyze_annotations()


if __name__ == "__main__":
    main()
