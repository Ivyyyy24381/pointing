# Reprocessing Pointing Analysis with Arm Labels

Quick guide for reprocessing pointing data when the automatic arm detection is incorrect.

## Quick Reference

### Step 1: Generate CSV Template

```bash
python batch_process_study.py /path/to/study_folder --generate-arm-csv
```

This creates `<study_name>_arm_overrides.csv` with columns:

| Column | Description |
|--------|-------------|
| `study` | Bag/study name |
| `trial` | Trial number (trial_1, trial_2, etc.) |
| `camera` | Camera name (cam1, cam2) |
| `detected_arm` | What was auto-detected |
| `override_arm` | **Edit this** → `left`, `right`, or `skip` |
| `reprocess` | **Edit this** → `yes` to reprocess, `no` to skip |

### Step 2: Edit the CSV

Open in Excel or text editor:

```csv
study,trial,camera,detected_arm,override_arm,reprocess
BDL396_study1,trial_1,cam1,right,,no
BDL396_study1,trial_2,cam1,left,right,yes
BDL396_study1,trial_3,cam1,unknown,left,yes
BDL396_study1,trial_4,cam1,right,skip,no
```

**Instructions:**
1. Review `detected_arm` column
2. Set `override_arm` to `left` or `right` where detection is wrong
3. Set `reprocess` to `yes` for trials you want to reprocess
4. Set `override_arm` to `skip` to skip a trial entirely

### Step 3: Reprocess

```bash
# Fast: Only reprocess pointing (uses existing skeleton data)
python batch_process_study.py /path/to/study_folder --pointing-only --arm-csv arms.csv

# Specific trial only
python batch_process_study.py /path/to/study_folder --pointing-only --arm-csv arms.csv --trial trial_2

# Full reprocess with arm override (re-runs skeleton detection)
python batch_process_study.py /path/to/study_folder --arm-csv arms.csv
```

---

## Command Options

| Option | Description |
|--------|-------------|
| `--generate-arm-csv` | Generate CSV template from existing data |
| `--pointing-only` | Skip skeleton detection, only reprocess pointing (fast) |
| `--arm-csv <path>` | Path to arm override CSV file |
| `--trial <name>` | Process only specific trial (e.g., `trial_2`) |
| `--cameras <cam1 cam2>` | Process only specific cameras |

---

## What `--pointing-only` Does

The `--pointing-only` mode is much faster because it:

1. **Loads existing `skeleton_2d.json`** instead of re-running MediaPipe
2. **Recomputes arm vectors** with the new arm (left/right)
3. **Regenerates pointing analysis** (CSV + plots)
4. **Skips subject (dog) detection**

Use this when you only need to fix the pointing arm, not re-run the full pipeline.

---

## Example Workflow

```bash
# 1. Generate the CSV template
python batch_process_study.py /data/BDL396_study1 --generate-arm-csv

# 2. Edit the CSV (set override_arm and reprocess=yes for trials to fix)
# vim /data/BDL396_study1_arm_overrides.csv

# 3. Reprocess only the trials marked for reprocessing
python batch_process_study.py /data/BDL396_study1 --pointing-only --arm-csv /data/BDL396_study1_arm_overrides.csv

# 4. Check the updated plots
ls /data/BDL396_study1_output/trial_*/cam1/*.png
```

---

## Output Files Updated

When reprocessing, these files are updated:

| File | Description |
|------|-------------|
| `skeleton_2d.json` | Updated with new arm vectors |
| `pointing_hand.json` | Updated pointing arm info |
| `processed_gesture.csv` | Recomputed pointing analysis |
| `2d_pointing_trace.png` | Regenerated pointing trace plot |
| `distance_to_targets_timeseries.png` | Regenerated distance plot |
| `distance_to_targets_summary.png` | Regenerated summary plot |
| `pointing_accuracy_comparison.png` | Regenerated accuracy heatmap |
