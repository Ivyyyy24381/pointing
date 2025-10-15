# 📁 Dataset Restructuring and Naming Guide

This document explains the conversion from the **old processed data format** to the **new structured trial-based format**, including the temporary folder layout used during conversion.

---

## 🧱 1. Old Processed Structure

Each trial folder was named using simple integers:

1/
2/
3/
…

Within each trial folder:
- **Color images** were named:

*_Color_0679.png

- **Depth images** were named:

*_Depth_0679.png

- If a “depth” image was actually an RGB visualization (rainbow-colored), it should have been renamed to:

*_Depth_Color_0679.png

This inconsistent naming caused downstream processing issues.

---

## 🧪 2. Temporary Processing Structure

During conversion, files are renamed and copied into a temporary folder:

.processing_trial/
color/
Color_000001.png
depth/
Depth_000001.png

- The **user loads the front-view camera** color and depth streams.  
- Conversion scripts ensure standardized naming and sequential numbering.  
- This folder is used as a clean staging area for downstream processing.

---

## 🆕 3. New Structured Format

Each trial is renamed and stored with **semantic names**:

- `trial_0`: Calibration trial
- `trial_1`, `trial_2`, ...: Actual data collection trials

Example layout:

trial_0/
cam1/
color/
frame_000001.png
depth/
frame_000001.png
cam2/
color/
depth/
trial_1/
cam1/
…

### Key points:
- File names are **zero-padded sequential frame numbers** (`frame_000001.png`), ensuring alignment between color and depth.
- Depth data is now stored as `.png` files instead of `.npy` arrays.
- `.npy` depth arrays are automatically converted to `.png` during restructuring for consistency.
- Multiple cameras (e.g., `cam1`, `cam2`) are supported per trial.

---

## 🚀 Conversion Pipeline

1. **Input:** Old folders (e.g., `1/`, `2/`) with `*_Color_####.png` and `*_Depth_####.png` files.  
2. **Temporary stage:** Files copied to `trial` with cleaned sequential names.  
3. **Output:** Trials organized under `trial_1`, `trial_2`, etc., with proper `cam*/color` and `cam*/depth` subfolders.

---

## 📝 Notes

- Make sure to rename any rainbow-colored depth images to `*_Depth_Color_####.png` before conversion.  
- Ensure both color and depth frame counts match per camera for synchronization.  
- You can easily loop through trials programmatically since the folder structure is consistent.
