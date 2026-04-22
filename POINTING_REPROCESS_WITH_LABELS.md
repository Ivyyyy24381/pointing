# Post-Processing Pipeline

All post-processing uses `batch_postprocess.py`. It works entirely from saved output data -- no raw video frames needed.

---

## Quick Start

```bash
# 1. See status of all trials
python batch_postprocess.py scan /path/to/study_output

# 2. Fix targets (find fallback from distribution, apply to bad trials) + reprocess
python batch_postprocess.py fix-targets /path/to/study_output --reprocess

# 3. Reprocess pointing with arm overrides
python batch_postprocess.py reprocess /path/to/study_output --arm-csv arms.csv

# 4. Reprocess with dog depth filter (only pointing when dog is near start)
python batch_postprocess.py reprocess /path/to/study_output --dog-depth-max 2.5
```

---

## Commands

### `scan` - Check status of all trials

```bash
python batch_postprocess.py scan /path/to/study_output
```

Shows a table for every trial/camera:
- How many targets were detected
- Target depth range
- Which arm was detected
- Whether skeleton data and CSV exist
- Flags trials with missing targets

### `fix-targets` - Fix missing or bad targets

```bash
# Preview what would be fixed (no changes)
python batch_postprocess.py fix-targets /path/to/study_output --dry-run

# Fix targets
python batch_postprocess.py fix-targets /path/to/study_output

# Fix targets AND reprocess pointing in one step
python batch_postprocess.py fix-targets /path/to/study_output --reprocess

# Fix + reprocess with arm overrides
python batch_postprocess.py fix-targets /path/to/study_output --reprocess --arm-csv arms.csv
```

**How it works:**

1. Scans all trials for each camera (cam1, cam2, cam3)
2. Collects every trial where all 4 targets were detected
3. Computes **median position** for each target across all good trials (robust to depth noise)
4. Prints per-target position and standard deviation
5. Saves reference to `<study_output>/reference_targets/<cam>_targets.json`
6. Replaces targets in trials that have:
   - Fewer than 4 targets detected
   - More than 4 targets detected
   - Target positions far from the reference (bad depth)
7. Backs up original files as `.bak`

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--max-depth-diff` | 1.0m | Exclude trials where target depth spread exceeds this |
| `--max-position-diff` | 1.0m | Replace targets if position differs from reference by more than this |
| `--dry-run` | off | Preview changes without modifying files |
| `--reprocess` | off | Also reprocess pointing analysis for fixed trials |
| `--arm-csv` | none | Arm override CSV for reprocessing |

### `reprocess` - Reprocess pointing from saved data

```bash
# Reprocess all trials
python batch_postprocess.py reprocess /path/to/study_output

# Reprocess with arm override CSV
python batch_postprocess.py reprocess /path/to/study_output --arm-csv arms.csv

# Reprocess specific trial/camera
python batch_postprocess.py reprocess /path/to/study_output --trial trial_2 --camera cam1

# Reprocess with dog depth filter (only keep frames where dog is near start)
python batch_postprocess.py reprocess /path/to/study_output --dog-depth-max 2.5

# Combine: arm overrides + dog depth filter
python batch_postprocess.py reprocess /path/to/study_output --arm-csv arms.csv --dog-depth-max 2.5
```

**What it does (fast - no MediaPipe re-run):**
1. Loads existing `skeleton_2d.json` (already extracted landmarks)
2. Loads `target_detections_cam_frame.json` (fixed or original)
3. *(Optional)* Filters frames by dog depth -- only keeps frames where dog is close to starting position
4. Recomputes arm vectors with the specified arm (left/right)
5. Runs pointing analysis (ray-ground intersection, distance to targets)
6. Applies Kalman smoothing
7. Exports CSV + regenerates all plots

### `generate-csv` - Create arm override CSV template

```bash
python batch_postprocess.py generate-csv /path/to/study_output
```

Creates a CSV with columns:

| Column | Description |
|--------|-------------|
| `trial` | Trial name (trial_1, trial_2, etc.) |
| `camera` | Camera name (cam1, cam2) |
| `detected_arm` | Auto-detected pointing arm |
| `num_targets` | How many targets were detected |
| `override_arm` | **Edit this** -> `left`, `right`, or `skip` |
| `reprocess` | **Edit this** -> `yes` to reprocess, `no` to skip |

Example CSV:
```csv
trial,camera,detected_arm,num_targets,override_arm,reprocess
trial_1,cam1,right,4,,no
trial_2,cam1,left,4,right,yes
trial_3,cam1,unknown,2,left,yes
trial_4,cam1,right,4,skip,no
```

---

## Typical Workflows

### Fix everything in one pass

```bash
# Step 1: Check what needs fixing
python batch_postprocess.py scan /path/to/study_output

# Step 2: Fix targets + reprocess pointing
python batch_postprocess.py fix-targets /path/to/study_output --reprocess
```

### Fix targets, then manually review arms

```bash
# Step 1: Fix targets only
python batch_postprocess.py fix-targets /path/to/study_output

# Step 2: Generate arm CSV
python batch_postprocess.py generate-csv /path/to/study_output

# Step 3: Edit CSV to correct arms
# vim study_output_postprocess.csv

# Step 4: Reprocess with arm overrides
python batch_postprocess.py reprocess /path/to/study_output --arm-csv study_output_postprocess.csv
```

### Reprocess just one trial

```bash
python batch_postprocess.py reprocess /path/to/study_output --trial trial_5 --camera cam1
```

---

## Output Files Updated

When reprocessing, these files are updated:

| File | Description |
|------|-------------|
| `pointing_hand.json` | Updated pointing arm info |
| `processed_gesture.csv` | Recomputed pointing analysis |
| `2d_pointing_trace.png` | Regenerated pointing trace plot |
| `distance_to_targets_timeseries.png` | Distance to each target over time |
| `distance_to_targets_summary.png` | Summary bar chart |
| `pointing_accuracy_comparison.png` | Accuracy heatmap |

When fixing targets, these are also updated:

| File | Description |
|------|-------------|
| `target_detections_cam_frame.json` | Replaced with reference targets |
| `target_detections_cam_frame.json.bak` | Backup of original |
| `reference_targets/<cam>_targets.json` | Saved reference per camera |

---

## Reference Target Selection

The fallback reference targets are computed using **median positions** across all trials where 4 targets were successfully detected:

- For each camera (cam1, cam2, cam3), collects all trials with 4 valid targets
- Excludes trials where depth spread between targets is too large (default >1m)
- Excludes trials with zero/negative depth values
- Computes median x, y, z position for each of the 4 targets
- Reports standard deviation to show detection stability
- Saves to `reference_targets/<cam>_targets.json` for future use

This is more robust than using any single trial, as depth noise from individual frames gets averaged out.

---

## Dog Depth Filtering

During a trial, the human points at targets and the dog eventually moves toward them. The pointing data is only meaningful **before the dog moves** -- while it's still at the starting position.

The `--dog-depth-max` option filters out frames where the dog has already moved past the threshold depth:

```bash
# Only analyze pointing when dog depth < 2.5m (still at start)
python batch_postprocess.py reprocess /path/to/study_output --dog-depth-max 2.5
```

**How it works:**
1. Loads `dog_detection_results.json` for each trial/camera
2. Gets the dog's nose Z-depth (distance from camera) per frame
3. Only keeps skeleton frames where dog depth is below the threshold
4. Runs pointing analysis only on those filtered frames

**Choosing the right threshold:**
- Use `scan` to see the depth range of targets for each camera
- The dog starts near the human and moves toward the targets
- Set `--dog-depth-max` to a value between the dog's starting depth and the target depth
- The threshold depends on camera position (cam1, cam2, cam3 see different depths)

**Note:** If there's no dog detection data for a trial, those frames are excluded (can't confirm dog is still at start).
