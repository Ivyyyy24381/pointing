#!/usr/bin/env python3
"""
Comprehensive trace analysis with visualizations.

Produces:
1. Global target position consistency check
2. All traces overlaid (baby + dog) to spot outliers
3. Pointing arm validation (annotated vs detected)
4. First-choice comparison (annotation vs automated)
5. Per-subject trace quality summary

All outputs saved to _analysis_output/ directory.
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ─── Paths ───
DATA_ROOT = Path("/home/tigerli/Documents/pointing_data")
CCD_OUTPUT = DATA_ROOT / "point_comprehension_CCD_output"
OWNER_OUTPUT = DATA_ROOT / "owner_pointing_output"
BDL_PROD_OUTPUT = DATA_ROOT / "point_production_BDL_output"

CCD_ANNOTATIONS = DATA_ROOT / "PVPT_Comprehension_Data - Long Data.csv"
OWNER_LONG_DATA = DATA_ROOT / "PVPO_Production_Data - Long_Data.csv"
OWNER_TRACKER = OWNER_OUTPUT / "PVPO_Production_Data - Output_Tracker.csv"
BDL_COMP_ANNOTATIONS = BDL_PROD_OUTPUT / "PVP_Comprehension_Data - Long Data.csv"

OUTPUT_DIR = DATA_ROOT / "_analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_csv(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ═══════════════════════════════════════════════════════════════
#  PART 1: Global Target Position Consistency
# ═══════════════════════════════════════════════════════════════
def analyze_target_consistency():
    """Load all target_coordinates.json and check consistency."""
    print("=" * 70)
    print("  PART 1: Target Position Consistency")
    print("=" * 70)

    all_targets = {1: [], 2: [], 3: [], 4: []}  # target_id -> list of (x, y, z, study, trial)
    studies_checked = set()

    for output_dir, label in [(CCD_OUTPUT, "CCD"), (OWNER_OUTPUT, "Owner"), (BDL_PROD_OUTPUT, "BDL_Prod")]:
        if not output_dir.exists():
            continue
        for subj_dir in sorted(output_dir.iterdir()):
            if not subj_dir.is_dir() or subj_dir.name.startswith("_"):
                continue
            for trial_dir in sorted(subj_dir.iterdir()):
                if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                    continue

                # Try multiple locations for target JSON
                for tpath in [
                    trial_dir / "target_coordinates.json",
                    trial_dir / "cam1" / "target_detections_cam_frame.json",
                ]:
                    if tpath.exists():
                        try:
                            with open(tpath) as f:
                                data = json.load(f)

                            if "targets" in data:
                                # target_coordinates.json format
                                for t in data["targets"]:
                                    tid = t["id"]
                                    coords = t["world_coords"]
                                    all_targets[tid].append((*coords, label, subj_dir.name, trial_dir.name))
                            elif isinstance(data, list):
                                # cam_frame format
                                for t in data:
                                    tid = int(t["label"].replace("target_", ""))
                                    all_targets[tid].append((t["x"], t["y"], t["z"], label, subj_dir.name, trial_dir.name))

                            studies_checked.add(f"{label}/{subj_dir.name}")
                            break
                        except (json.JSONDecodeError, KeyError, ValueError):
                            pass

    print(f"Studies checked: {len(studies_checked)}")
    for tid in range(1, 5):
        coords = all_targets[tid]
        if coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            zs = [c[2] for c in coords]
            print(f"  Target {tid}: n={len(coords)}, "
                  f"X={np.mean(xs):.3f}±{np.std(xs):.4f}, "
                  f"Y={np.mean(ys):.3f}±{np.std(ys):.4f}, "
                  f"Z={np.mean(zs):.3f}±{np.std(zs):.4f}")

    # ── Plot: XZ view of all targets ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    colors = {1: "red", 2: "orange", 3: "green", 4: "blue"}
    labels_seen = set()

    for dataset_label, marker, ax_idx in [("CCD", "o", 0), ("Owner", "s", 1), ("BDL_Prod", "^", 2)]:
        ax = axes[ax_idx]
        ax.set_title(f"{dataset_label} Target Positions (XZ plane)", fontsize=12)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m, depth)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        for tid in range(1, 5):
            coords = [c for c in all_targets[tid] if c[3] == dataset_label]
            if coords:
                xs = [c[0] for c in coords]
                zs = [c[2] for c in coords]
                ax.scatter(xs, zs, c=colors[tid], marker=marker, s=20, alpha=0.5,
                          label=f"Target {tid}" if dataset_label not in labels_seen else None)
                # Mark mean with a large X
                ax.scatter(np.mean(xs), np.mean(zs), c=colors[tid], marker="X", s=200, edgecolors="black", linewidth=1.5)

        ax.legend()
        labels_seen.add(dataset_label)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_target_positions_per_dataset.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 01_target_positions_per_dataset.png")

    # ── Plot: All targets overlaid with per-study coloring ──
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("All Target Positions (XZ plane, all datasets)", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m, depth)")
    ax.grid(True, alpha=0.3)

    study_names = sorted(studies_checked)
    cmap = cm.get_cmap("tab20", len(study_names))
    study_color = {s: cmap(i) for i, s in enumerate(study_names)}

    for tid in range(1, 5):
        coords = all_targets[tid]
        for c in coords:
            study_key = f"{c[3]}/{c[4]}"
            ax.scatter(c[0], c[2], c=[study_color.get(study_key, "gray")], s=15, alpha=0.4)

        if coords:
            xs = [c[0] for c in coords]
            zs = [c[2] for c in coords]
            mean_x, mean_z = np.mean(xs), np.mean(zs)
            ax.scatter(mean_x, mean_z, c=colors[tid], marker="*", s=300,
                      edgecolors="black", linewidth=2, zorder=10, label=f"T{tid} mean ({mean_x:.2f}, {mean_z:.2f})")

    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_target_positions_all.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 01_target_positions_all.png")

    return all_targets


# ═══════════════════════════════════════════════════════════════
#  PART 2: All Traces Overlaid (Outlier Detection)
# ═══════════════════════════════════════════════════════════════
def load_trace_from_csv(csv_path):
    """Load trace3d x/z from a CSV, return arrays."""
    if not csv_path.exists():
        return None, None
    rows = load_csv(csv_path)
    if not rows:
        return None, None
    xs, zs = [], []
    for r in rows:
        try:
            x = float(r.get("trace3d_x", "nan"))
            z = float(r.get("trace3d_z", "nan"))
            if not (math.isnan(x) or math.isnan(z)):
                xs.append(x)
                zs.append(z)
        except (ValueError, TypeError):
            pass
    if not xs:
        return None, None
    return np.array(xs), np.array(zs)


def analyze_all_traces():
    """Overlay all traces, detect outliers."""
    print("\n" + "=" * 70)
    print("  PART 2: All Traces Overlaid (Outlier Detection)")
    print("=" * 70)

    # ── CCD baby traces ──
    ccd_traces = []
    ccd_labels = []
    ccd_outliers = []

    for subj_dir in sorted(CCD_OUTPUT.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith("_"):
            continue
        for trial_dir in sorted(subj_dir.iterdir()):
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue
            csv_path = trial_dir / "processed_subject_result_table.csv"
            xs, zs = load_trace_from_csv(csv_path)
            if xs is not None:
                label = f"{subj_dir.name}/{trial_dir.name}"
                ccd_traces.append((xs, zs))
                ccd_labels.append(label)

                # Outlier check: any point >10m from origin or trace range > 6m
                max_x = np.max(np.abs(xs))
                max_z = np.max(np.abs(zs))
                x_range = np.max(xs) - np.min(xs)
                z_range = np.max(zs) - np.min(zs)
                if max_x > 5 or max_z > 10 or x_range > 6 or z_range > 6:
                    ccd_outliers.append((label, max_x, max_z, x_range, z_range))

    print(f"CCD baby traces loaded: {len(ccd_traces)}")
    if ccd_outliers:
        print(f"  Outliers ({len(ccd_outliers)}):")
        for label, mx, mz, xr, zr in ccd_outliers:
            print(f"    {label}: max_x={mx:.1f}, max_z={mz:.1f}, range_x={xr:.1f}, range_z={zr:.1f}")
    else:
        print("  No outliers detected (all within reasonable bounds)")

    # ── Owner dog traces ──
    owner_traces = []
    owner_labels = []
    owner_outliers = []

    for subj_dir in sorted(OWNER_OUTPUT.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith("_"):
            continue
        for trial_dir in sorted(subj_dir.iterdir()):
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue
            csv_path = trial_dir / "cam1" / "processed_dog_result_table.csv"
            xs, zs = load_trace_from_csv(csv_path)
            if xs is not None and len(xs) > 1:
                label = f"{subj_dir.name}/{trial_dir.name}"
                owner_traces.append((xs, zs))
                owner_labels.append(label)

                max_x = np.max(np.abs(xs))
                max_z = np.max(np.abs(zs))
                x_range = np.max(xs) - np.min(xs)
                z_range = np.max(zs) - np.min(zs)
                if max_x > 5 or max_z > 10 or x_range > 8 or z_range > 8:
                    owner_outliers.append((label, max_x, max_z, x_range, z_range))

    print(f"Owner dog traces loaded: {len(owner_traces)}")
    if owner_outliers:
        print(f"  Outliers ({len(owner_outliers)}):")
        for label, mx, mz, xr, zr in owner_outliers:
            print(f"    {label}: max_x={mx:.1f}, max_z={mz:.1f}, range_x={xr:.1f}, range_z={zr:.1f}")
    else:
        print("  No outliers detected")

    # ── Plot: All CCD baby traces overlaid ──
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(f"All CCD Baby Traces Overlaid (n={len(ccd_traces)})", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m, depth)")
    ax.grid(True, alpha=0.3)

    for xs, zs in ccd_traces:
        ax.plot(xs, zs, alpha=0.15, linewidth=0.5, color="steelblue")
        ax.scatter(xs[0], zs[0], c="green", s=8, zorder=5, alpha=0.3)  # start
        ax.scatter(xs[-1], zs[-1], c="red", s=8, zorder=5, alpha=0.3)  # end

    # Highlight outliers
    for i, (label, _, _, _, _) in enumerate(ccd_outliers):
        idx = ccd_labels.index(label)
        xs, zs = ccd_traces[idx]
        ax.plot(xs, zs, alpha=0.8, linewidth=1.5, color="red", label=label if i < 5 else None)

    # Add target reference (mean positions from typical CCD study)
    target_positions = {
        1: (0.796, 1.625), 2: (0.509, 2.326),
        3: (-0.109, 2.791), 4: (-0.797, 3.049)
    }
    for tid, (tx, tz) in target_positions.items():
        ax.scatter(tx, tz, c=["red", "orange", "green", "blue"][tid-1],
                  marker="*", s=200, edgecolors="black", linewidth=1.5, zorder=10,
                  label=f"Target {tid}")

    ax.scatter([], [], c="green", s=30, label="Start points")
    ax.scatter([], [], c="red", s=30, label="End points")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_all_CCD_baby_traces.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 02_all_CCD_baby_traces.png")

    # ── Plot: All owner dog traces overlaid ──
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(f"All Owner Dog Traces Overlaid (n={len(owner_traces)})", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m, depth)")
    ax.grid(True, alpha=0.3)

    for xs, zs in owner_traces:
        ax.plot(xs, zs, alpha=0.15, linewidth=0.5, color="darkorange")
        ax.scatter(xs[0], zs[0], c="green", s=8, zorder=5, alpha=0.3)
        ax.scatter(xs[-1], zs[-1], c="red", s=8, zorder=5, alpha=0.3)

    for i, (label, _, _, _, _) in enumerate(owner_outliers):
        idx = owner_labels.index(label)
        xs, zs = owner_traces[idx]
        ax.plot(xs, zs, alpha=0.8, linewidth=1.5, color="red", label=label if i < 5 else None)

    ax.scatter([], [], c="green", s=30, label="Start points")
    ax.scatter([], [], c="red", s=30, label="End points")
    if owner_outliers:
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_all_owner_dog_traces.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 02_all_owner_dog_traces.png")

    # ── Plot: Distribution of trace statistics ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row_idx, (traces, labels, dataset_name) in enumerate([
        (ccd_traces, ccd_labels, "CCD Baby"),
        (owner_traces, owner_labels, "Owner Dog"),
    ]):
        motions = []
        start_xs, start_zs = [], []
        end_xs, end_zs = [], []
        n_frames_list = []

        for xs, zs in traces:
            dx = np.diff(xs)
            dz = np.diff(zs)
            total_motion = np.sum(np.sqrt(dx**2 + dz**2))
            motions.append(total_motion)
            start_xs.append(xs[0])
            start_zs.append(zs[0])
            end_xs.append(xs[-1])
            end_zs.append(zs[-1])
            n_frames_list.append(len(xs))

        axes[row_idx, 0].hist(motions, bins=50, color="steelblue" if row_idx == 0 else "darkorange", alpha=0.7)
        axes[row_idx, 0].set_title(f"{dataset_name}: Total Path Length (m)")
        axes[row_idx, 0].set_xlabel("Distance (m)")
        axes[row_idx, 0].axvline(np.median(motions), color="red", linestyle="--", label=f"median={np.median(motions):.2f}")
        axes[row_idx, 0].legend()

        axes[row_idx, 1].scatter(start_xs, start_zs, c="green", s=10, alpha=0.3, label="Start")
        axes[row_idx, 1].scatter(end_xs, end_zs, c="red", s=10, alpha=0.3, label="End")
        axes[row_idx, 1].set_title(f"{dataset_name}: Start & End Positions")
        axes[row_idx, 1].set_xlabel("X (m)")
        axes[row_idx, 1].set_ylabel("Z (m)")
        axes[row_idx, 1].set_aspect("equal")
        axes[row_idx, 1].legend()
        axes[row_idx, 1].grid(True, alpha=0.3)

        axes[row_idx, 2].hist(n_frames_list, bins=50, color="steelblue" if row_idx == 0 else "darkorange", alpha=0.7)
        axes[row_idx, 2].set_title(f"{dataset_name}: Frame Count per Trial")
        axes[row_idx, 2].set_xlabel("Frames")
        axes[row_idx, 2].axvline(np.median(n_frames_list), color="red", linestyle="--",
                                  label=f"median={np.median(n_frames_list):.0f}")
        axes[row_idx, 2].legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_trace_statistics.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 02_trace_statistics.png")

    return ccd_traces, ccd_labels, ccd_outliers, owner_traces, owner_labels, owner_outliers


# ═══════════════════════════════════════════════════════════════
#  PART 3: Pointing Arm Validation
# ═══════════════════════════════════════════════════════════════
def validate_pointing_arm():
    """Compare annotated arm (Output_Tracker) vs pipeline detected arm."""
    print("\n" + "=" * 70)
    print("  PART 3: Pointing Arm Validation")
    print("=" * 70)

    if not OWNER_TRACKER.exists():
        print("  Output_Tracker.csv not found, skipping")
        return {}

    tracker_rows = load_csv(OWNER_TRACKER)
    print(f"  Loaded {len(tracker_rows)} rows from Output_Tracker")

    # Build annotation lookup: (participant_id, trial_number) -> arm
    ann_arms = {}
    for r in tracker_rows:
        if r.get("excluded", "").upper() == "Y":
            continue
        pid = r.get("participant_id", "").strip()
        try:
            trial = int(r.get("trial_number", "0"))
        except ValueError:
            continue
        arm = r.get("Owner Arm Used", "").strip().upper()
        if arm in ("L", "R"):
            ann_arms[(pid, trial)] = arm

    print(f"  Annotated arms: {len(ann_arms)} trials with L/R annotation")

    # Find detected arms from pipeline
    results = []
    match_count = 0
    mismatch_count = 0
    no_data_count = 0

    for subj_dir in sorted(OWNER_OUTPUT.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith("_"):
            continue

        # Extract BDL ID from folder name
        bdl_id = subj_dir.name.split("_")[0]

        for trial_dir in sorted(subj_dir.iterdir()):
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue

            trial_num = int(trial_dir.name.replace("trial_", ""))
            ann_trial = trial_num + 1  # output trial_0 → annotation trial 1

            ann_arm = ann_arms.get((bdl_id, ann_trial))
            if ann_arm is None:
                continue

            # Try pointing_hand.json first (cam1)
            detected_arm = None
            ph_path = trial_dir / "cam1" / "pointing_hand.json"
            if ph_path.exists():
                try:
                    with open(ph_path) as f:
                        ph = json.load(f)
                    da = ph.get("pointing_hand", "").lower()
                    if da in ("left", "right"):
                        detected_arm = "L" if da == "left" else "R"
                except (json.JSONDecodeError, KeyError):
                    pass

            if detected_arm is None:
                no_data_count += 1
                continue

            status = "MATCH" if ann_arm == detected_arm else "MISMATCH"
            if status == "MATCH":
                match_count += 1
            else:
                mismatch_count += 1

            results.append({
                "subject": subj_dir.name,
                "trial": trial_dir.name,
                "ann_arm": ann_arm,
                "detected_arm": detected_arm,
                "status": status,
            })

    total = match_count + mismatch_count
    pct = 100 * match_count / total if total > 0 else 0
    print(f"\n  Arm Validation Results:")
    print(f"    Match: {match_count}/{total} ({pct:.1f}%)")
    print(f"    Mismatch: {mismatch_count}/{total}")
    print(f"    No detection data: {no_data_count}")

    # Show mismatches
    mismatches = [r for r in results if r["status"] == "MISMATCH"]
    if mismatches:
        print(f"\n  Arm Mismatches ({len(mismatches)}):")
        for r in mismatches[:30]:
            print(f"    {r['subject']}/{r['trial']}: ann={r['ann_arm']}, detected={r['detected_arm']}")

    # ── Plot: Confusion matrix ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix
    cm_data = np.zeros((2, 2), dtype=int)
    for r in results:
        i = 0 if r["ann_arm"] == "L" else 1
        j = 0 if r["detected_arm"] == "L" else 1
        cm_data[i, j] += 1

    ax = axes[0]
    im = ax.imshow(cm_data, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Left", "Right"])
    ax.set_yticklabels(["Left", "Right"])
    ax.set_xlabel("Detected Arm")
    ax.set_ylabel("Annotated Arm")
    ax.set_title(f"Pointing Arm: Annotated vs Detected\n({match_count}/{total} match, {pct:.0f}%)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_data[i, j]), ha="center", va="center", fontsize=18,
                   color="white" if cm_data[i, j] > total/4 else "black")
    fig.colorbar(im, ax=ax)

    # Per-subject accuracy
    subj_stats = defaultdict(lambda: {"match": 0, "total": 0})
    for r in results:
        subj = r["subject"].split("_")[0]
        subj_stats[subj]["total"] += 1
        if r["status"] == "MATCH":
            subj_stats[subj]["match"] += 1

    subjects = sorted(subj_stats.keys())
    accuracies = [100 * subj_stats[s]["match"] / subj_stats[s]["total"] for s in subjects]
    totals = [subj_stats[s]["total"] for s in subjects]

    ax = axes[1]
    bars = ax.bar(range(len(subjects)), accuracies, color="steelblue", alpha=0.7)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Arm Match Accuracy (%)")
    ax.set_title("Per-Subject Arm Detection Accuracy")
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_ylim(0, 105)
    for i, (acc, tot) in enumerate(zip(accuracies, totals)):
        ax.text(i, acc + 2, f"{tot}t", ha="center", fontsize=7)
    ax.legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_arm_validation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 03_arm_validation.png")

    return {r["subject"] + "/" + r["trial"]: r for r in results}


# ═══════════════════════════════════════════════════════════════
#  PART 4: First-Choice Comparison (Annotation vs Automated)
# ═══════════════════════════════════════════════════════════════
def validate_first_choice():
    """Compare annotation choice_1 vs automated first_choice from optimal_path_deviation.csv."""
    print("\n" + "=" * 70)
    print("  PART 4: First-Choice Validation (Annotation vs Automated)")
    print("=" * 70)

    # Load CCD annotations
    ccd_ann = {}
    if CCD_ANNOTATIONS.exists():
        for r in load_csv(CCD_ANNOTATIONS):
            if r.get("excluded", "").upper() == "Y":
                continue
            pid = r.get("participant_id", "").strip()
            try:
                trial = int(r.get("trial_number", "0"))
            except ValueError:
                continue
            c1 = r.get("choice_1", "NA").strip()
            if c1 not in ("NA", ""):
                try:
                    ccd_ann[(pid, trial)] = int(c1)
                except ValueError:
                    pass

    print(f"CCD annotations loaded: {len(ccd_ann)} trials with choice_1")

    # Load owner annotations (from Long Data, NOT tracker)
    owner_ann = {}
    if OWNER_LONG_DATA.exists():
        for r in load_csv(OWNER_LONG_DATA):
            if r.get("excluded", "").upper() == "Y":
                continue
            pid = r.get("participant_id", "").strip()
            try:
                trial = int(r.get("trial", "0"))
            except ValueError:
                continue
            c1 = r.get("choice_1", "NA").strip()
            if c1 not in ("NA", ""):
                try:
                    owner_ann[(pid, trial)] = int(c1)
                except ValueError:
                    pass

    print(f"Owner annotations loaded: {len(owner_ann)} trials with choice_1")

    all_results = {"CCD": [], "Owner": []}

    for output_dir, annotations, ann_trial_col, dataset_label in [
        (CCD_OUTPUT, ccd_ann, "trial_number", "CCD"),
        (OWNER_OUTPUT, owner_ann, "trial", "Owner"),
    ]:
        if not output_dir.exists():
            continue

        for subj_dir in sorted(output_dir.iterdir()):
            if not subj_dir.is_dir() or subj_dir.name.startswith("_"):
                continue

            subj_id = subj_dir.name.split("_")[0]

            for trial_dir in sorted(subj_dir.iterdir()):
                if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                    continue

                trial_num = int(trial_dir.name.replace("trial_", ""))

                # Try mapping: trial_0 → annotation trial 1
                ann_c1 = annotations.get((subj_id, trial_num + 1))
                if ann_c1 is None:
                    ann_c1 = annotations.get((subj_id, trial_num))

                if ann_c1 is None:
                    continue

                # Load automated first_choice from optimal_path_deviation.csv
                opt_path = trial_dir / "optimal_path_deviation.csv"
                if not opt_path.exists():
                    continue

                opt_rows = load_csv(opt_path)
                if not opt_rows:
                    continue

                auto_fc = opt_rows[0].get("first_choice", "")
                if auto_fc.startswith("target_"):
                    try:
                        auto_fc_num = int(auto_fc.replace("target_", ""))
                    except ValueError:
                        continue
                else:
                    continue

                status = "MATCH" if ann_c1 == auto_fc_num else "MISMATCH"
                all_results[dataset_label].append({
                    "subject": subj_dir.name,
                    "trial": trial_dir.name,
                    "ann_choice_1": ann_c1,
                    "auto_first_choice": auto_fc_num,
                    "status": status,
                })

    # Print summary and make plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (dataset_label, results) in enumerate(all_results.items()):
        if not results:
            continue

        match = sum(1 for r in results if r["status"] == "MATCH")
        total = len(results)
        pct = 100 * match / total if total > 0 else 0

        print(f"\n{dataset_label} First-Choice Validation:")
        print(f"  Match: {match}/{total} ({pct:.1f}%)")

        # Confusion matrix 4x4
        cm_data = np.zeros((4, 4), dtype=int)
        for r in results:
            i = r["ann_choice_1"] - 1
            j = r["auto_first_choice"] - 1
            if 0 <= i < 4 and 0 <= j < 4:
                cm_data[i, j] += 1

        ax = axes[idx]
        im = ax.imshow(cm_data, cmap="Blues")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(["T1", "T2", "T3", "T4"])
        ax.set_yticklabels(["T1", "T2", "T3", "T4"])
        ax.set_xlabel("Automated First Choice")
        ax.set_ylabel("Annotated choice_1")
        ax.set_title(f"{dataset_label}: Annotated vs Automated\n({match}/{total} match, {pct:.0f}%)")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm_data[i, j]), ha="center", va="center", fontsize=12,
                       color="white" if cm_data[i, j] > total/8 else "black")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_first_choice_validation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 04_first_choice_validation.png")

    return all_results


# ═══════════════════════════════════════════════════════════════
#  PART 5: Per-Subject Summary with Quality Grades
# ═══════════════════════════════════════════════════════════════
def per_subject_quality_summary(ccd_traces, ccd_labels, owner_traces, owner_labels):
    """Grade each subject/trial and produce summary visualization."""
    print("\n" + "=" * 70)
    print("  PART 5: Per-Subject Quality Summary")
    print("=" * 70)

    # ── CCD: per-subject stacked traces ──
    # Group by subject
    subj_groups = defaultdict(list)
    for (xs, zs), label in zip(ccd_traces, ccd_labels):
        subj = label.split("/")[0]
        trial = label.split("/")[1]
        subj_groups[subj].append((xs, zs, trial))

    subjects = sorted(subj_groups.keys())
    n_subj = len(subjects)
    n_cols = 5
    n_rows = (n_subj + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_subj > 1 else [axes]

    target_positions = {
        1: (0.796, 1.625), 2: (0.509, 2.326),
        3: (-0.109, 2.791), 4: (-0.797, 3.049)
    }
    target_colors = {1: "red", 2: "orange", 3: "green", 4: "blue"}

    for i, subj in enumerate(subjects):
        ax = axes[i]
        trials = subj_groups[subj]
        cmap_trials = cm.get_cmap("viridis", len(trials))

        for j, (xs, zs, trial_name) in enumerate(sorted(trials, key=lambda t: t[2])):
            ax.plot(xs, zs, alpha=0.6, linewidth=1, color=cmap_trials(j))
            ax.scatter(xs[0], zs[0], c="green", s=20, zorder=5)
            ax.scatter(xs[-1], zs[-1], c="red", s=20, zorder=5)

        for tid, (tx, tz) in target_positions.items():
            ax.scatter(tx, tz, c=target_colors[tid], marker="*", s=100, zorder=10, edgecolors="black")

        ax.set_title(subj.split("_")[0], fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for i in range(len(subjects), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("CCD Baby Traces: Per-Subject Overview (green=start, red=end, stars=targets)", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_CCD_per_subject_traces.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 05_CCD_per_subject_traces.png")

    # ── Owner: per-subject stacked traces ──
    subj_groups_owner = defaultdict(list)
    for (xs, zs), label in zip(owner_traces, owner_labels):
        parts = label.split("/")
        subj = parts[0]
        trial = parts[1]
        subj_groups_owner[subj].append((xs, zs, trial))

    subjects_owner = sorted(subj_groups_owner.keys())
    n_subj_o = len(subjects_owner)
    n_rows_o = (n_subj_o + n_cols - 1) // n_cols

    fig, axes = plt.subplots(max(n_rows_o, 1), n_cols, figsize=(5 * n_cols, 4 * max(n_rows_o, 1)))
    axes = axes.flatten() if n_subj_o > 1 else [axes]

    for i, subj in enumerate(subjects_owner):
        ax = axes[i]
        trials = subj_groups_owner[subj]

        # Separate annotated vs extra trials
        for j, (xs, zs, trial_name) in enumerate(sorted(trials, key=lambda t: t[2])):
            trial_num = int(trial_name.replace("trial_", ""))
            color = "steelblue" if trial_num < 12 else "lightgray"
            alpha = 0.6 if trial_num < 12 else 0.3
            ax.plot(xs, zs, alpha=alpha, linewidth=1, color=color)
            if trial_num < 12:
                ax.scatter(xs[0], zs[0], c="green", s=15, zorder=5)
                ax.scatter(xs[-1], zs[-1], c="red", s=15, zorder=5)

        ax.set_title(subj.split("_")[0], fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    for i in range(len(subjects_owner), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Owner Dog Traces: Per-Subject (blue=annotated trials, gray=extra trials)", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_owner_per_subject_traces.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 05_owner_per_subject_traces.png")


# ═══════════════════════════════════════════════════════════════
#  PART 6: Optimal Path Deviation Analysis
# ═══════════════════════════════════════════════════════════════
def analyze_optimal_path_deviation():
    """Analyze optimal path deviation values and compare with ground truth trimming."""
    print("\n" + "=" * 70)
    print("  PART 6: Optimal Path Deviation Analysis")
    print("=" * 70)

    # Load CCD annotations for ground truth
    ccd_ann = {}
    if CCD_ANNOTATIONS.exists():
        for r in load_csv(CCD_ANNOTATIONS):
            if r.get("excluded", "").upper() == "Y":
                continue
            pid = r.get("participant_id", "").strip()
            try:
                trial = int(r.get("trial_number", "0"))
            except ValueError:
                continue
            c1 = r.get("choice_1", "NA").strip()
            if c1 not in ("NA", ""):
                try:
                    ccd_ann[(pid, trial)] = int(c1)
                except ValueError:
                    pass

    all_max_devs = []  # (max_dev, label, auto_target, ann_target, match)
    approach_fracs = []  # fraction of frames in approach phase

    for subj_dir in sorted(CCD_OUTPUT.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith("_"):
            continue
        subj_id = subj_dir.name.split("_")[0]

        for trial_dir in sorted(subj_dir.iterdir()):
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue

            trial_num = int(trial_dir.name.replace("trial_", ""))
            opt_csv = trial_dir / "optimal_path_deviation.csv"
            if not opt_csv.exists():
                continue

            rows = load_csv(opt_csv)
            if not rows:
                continue

            auto_fc = rows[0].get("first_choice", "")
            if auto_fc.startswith("target_"):
                auto_fc_num = int(auto_fc.replace("target_", ""))
            else:
                continue

            # Get annotation
            ann_c1 = ccd_ann.get((subj_id, trial_num + 1))
            if ann_c1 is None:
                ann_c1 = ccd_ann.get((subj_id, trial_num))

            # Compute max deviation for the auto-selected target
            dev_col = f"opt_path_dev_target_{auto_fc_num}"
            devs = []
            approach_count = 0
            total_count = 0
            for r in rows:
                total_count += 1
                if r.get("approach_phase", "") == "approach":
                    approach_count += 1
                try:
                    d = float(r.get(dev_col, "nan"))
                    if not math.isnan(d):
                        devs.append(d)
                except (ValueError, TypeError):
                    pass

            if devs:
                max_dev = max(devs)
                match = "MATCH" if ann_c1 == auto_fc_num else ("MISMATCH" if ann_c1 else "NO_ANN")
                label = f"{subj_dir.name}/{trial_dir.name}"
                all_max_devs.append((max_dev, label, auto_fc_num, ann_c1, match))

                if total_count > 0:
                    approach_fracs.append(approach_count / total_count)

    print(f"Analyzed {len(all_max_devs)} trials with optimal path deviation")
    devs_arr = np.array([d[0] for d in all_max_devs])
    print(f"  Max deviation: mean={np.mean(devs_arr):.3f}, median={np.median(devs_arr):.3f}, "
          f"max={np.max(devs_arr):.3f}, min={np.min(devs_arr):.3f}")
    print(f"  > 1m: {np.sum(devs_arr > 1)}, > 2m: {np.sum(devs_arr > 2)}, > 5m: {np.sum(devs_arr > 5)}")

    matched_devs = [d[0] for d in all_max_devs if d[4] == "MATCH"]
    mismatched_devs = [d[0] for d in all_max_devs if d[4] == "MISMATCH"]
    print(f"  When auto matches annotation: mean={np.mean(matched_devs):.3f}, n={len(matched_devs)}")
    if mismatched_devs:
        print(f"  When auto mismatches annotation: mean={np.mean(mismatched_devs):.3f}, n={len(mismatched_devs)}")

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Max deviation histogram
    ax = axes[0, 0]
    ax.hist(devs_arr, bins=50, color="steelblue", alpha=0.7)
    ax.axvline(np.median(devs_arr), color="red", linestyle="--", label=f"median={np.median(devs_arr):.3f}")
    ax.set_title("Distribution of Max Optimal Path Deviation (all CCD trials)")
    ax.set_xlabel("Max Deviation (m)")
    ax.set_ylabel("Count")
    ax.legend()

    # Deviation by match status
    ax = axes[0, 1]
    if matched_devs and mismatched_devs:
        ax.hist(matched_devs, bins=30, alpha=0.6, color="green", label=f"Match (n={len(matched_devs)})")
        ax.hist(mismatched_devs, bins=30, alpha=0.6, color="red", label=f"Mismatch (n={len(mismatched_devs)})")
        ax.set_title("Max Deviation: Matching vs Mismatching First Choice")
        ax.set_xlabel("Max Deviation (m)")
        ax.legend()

    # Approach phase fraction
    ax = axes[1, 0]
    if approach_fracs:
        ax.hist(approach_fracs, bins=30, color="darkorange", alpha=0.7)
        ax.axvline(np.median(approach_fracs), color="red", linestyle="--",
                  label=f"median={np.median(approach_fracs):.2f}")
        ax.set_title("Fraction of Frames in 'Approach' Phase")
        ax.set_xlabel("Fraction")
        ax.legend()

    # Scatter: max deviation vs total frames
    ax = axes[1, 1]
    # Color by match status
    for status, color, marker in [("MATCH", "green", "o"), ("MISMATCH", "red", "x"), ("NO_ANN", "gray", "s")]:
        subset = [(d[0], d[1]) for d in all_max_devs if d[4] == status]
        if subset:
            ax.scatter([d[0] for d in subset], range(len(subset)),
                      c=color, marker=marker, s=30, alpha=0.5, label=f"{status} (n={len(subset)})")
    ax.set_xlabel("Max Deviation (m)")
    ax.set_ylabel("Trial index")
    ax.set_title("Max Deviation per Trial (colored by first-choice match)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_optimal_path_deviation_analysis.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 06_optimal_path_deviation_analysis.png")

    # Top outliers
    sorted_devs = sorted(all_max_devs, key=lambda x: -x[0])
    print(f"\n  Top 15 largest max deviations:")
    for max_dev, label, auto_t, ann_t, match in sorted_devs[:15]:
        print(f"    {label}: {max_dev:.3f}m (auto=T{auto_t}, ann=T{ann_t}, {match})")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Part 1: Target consistency
    all_targets = analyze_target_consistency()

    # Part 2: All traces overlaid
    ccd_traces, ccd_labels, ccd_outliers, owner_traces, owner_labels, owner_outliers = analyze_all_traces()

    # Part 3: Pointing arm validation
    arm_results = validate_pointing_arm()

    # Part 4: First-choice validation
    fc_results = validate_first_choice()

    # Part 5: Per-subject summary
    per_subject_quality_summary(ccd_traces, ccd_labels, owner_traces, owner_labels)

    # Part 6: Optimal path deviation analysis
    analyze_optimal_path_deviation()

    print(f"\n{'='*70}")
    print(f"  ALL ANALYSES COMPLETE")
    print(f"  Output saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
