
## Point production Data Processing
**installation instruction:**

```
# check out the puppylab branch
git clone --branch puppy-lab --single-branch git@github.com:Ivyyyy24381/pointing.git

# set up environment

conda create -n point_production python=3.12
conda activate point_production

pip install -r requirements.txt
conda install --yes --file conda_requirements.txt $ or run this if using conda
```

### folder structure
for each trial, the folder should contain:
```

trial_folder/
├── Color/
│   └── ... (individual color frames, likely extracted)
├── Color.mp4
├── Depth/
│   └── ... (individual depth-aligned color frames)
├── Depth_Color/
│   └── ... (individual depth frames)
├── Depth_Color.mp4
├── Depth.mp4
├── gesture_data.csv                    # output from gesture detection
├── output.mp4                          # visualization of results

```

if the a specific trial is missing gesture_data.csv, go to step 0:
### step 0 [Optional]：(run if a trial is missing gesture_data.csv)
```
python batch_point_production.py
Example input:
- base path: /Users/ivy/Library/CloudStorage/GoogleDrive-xiao_he@brown.edu/Shared drives/pointing_production/
- subject folder: BDL202_Dee-Dee_front (if leave blank, process all subjects all trials)
- trial number: 2 (if leave blank, process all trials for designated subject)
```

### step 1: 
```
python GUI/skeleton_gui.py
```
- load the Color.mp4 video in a trial
- trim the valid pointing frames
- click process selection to get the pointing data

Output will produce the following files:
```
├── 2d_pointing_trace.png
├── fig/
│   └── ... (figures or plots)
├── processed_gesture_data.csv
```




**Command:**

live stream mode: python gesture_detection.py --mode live

realsense stream mode: python gesture_detection.py --mode rs

local video mode: python gesture_detection.py --mode video


