# Step 4: Pointing Calculation & Analysis

Calculate 3D pointing vectors and analyze distances to targets.

## Features

- ✅ **2D to 3D Conversion**: Convert skeleton keypoints to 3D using depth
- ✅ **Pointing Vector Calculation**: Compute pointing direction from arm pose
- ✅ **Distance Analysis**: Calculate distances to all targets
- ✅ **CSV Export**: Results in analyzable format

## Quick Start

### Process Single Trial

```bash
# Run full analysis pipeline
python -c "
from step4_calculation import PointingCalculator, compute_distances_to_targets

# Step 1: Calculate 3D pointing vectors
calc = PointingCalculator(fx=615, fy=615, cx=320, cy=240)
calc.process_trial(
    skeleton_file='trial_output/trial_1/cam1/skeleton_2d.json',
    depth_folder='trial_input/trial_1/cam1/depth',
    output_file='trial_output/trial_1/cam1/pointing_results.json'
)

# Step 2: Compute distances
from step4_calculation.distance_calculator import analyze_pointing_trial
analyze_pointing_trial(
    pointing_file='trial_output/trial_1/cam1/pointing_results.json',
    targets_file='trial_output/trial_1/cam1/target_detections.json',
    output_file='trial_output/trial_1/cam1/analysis_results.csv'
)
"
```

### Command Line

```bash
# Compute distances
python step4_calculation/distance_calculator.py \
  --pointing trial_output/trial_1/cam1/pointing_results.json \
  --targets trial_output/trial_1/cam1/target_detections.json \
  --output trial_output/trial_1/cam1/analysis_results.csv
```

## Output Format

### pointing_results.json
```json
{
  "frame_000001": {
    "frame": 1,
    "wrist_3d": {"x": 0.5, "y": 0.3, "z": 1.2},
    "elbow_3d": {"x": 0.4, "y": 0.5, "z": 1.1},
    "shoulder_3d": {"x": 0.3, "y": 0.6, "z": 1.0},
    "pointing_vector": [0.5, -0.3, 0.2],
    "pointing_arm": "right"
  }
}
```

### analysis_results.csv
```csv
frame,wrist_x,wrist_y,wrist_z,pointing_arm,distance_target_1,distance_target_2,distance_target_3,distance_target_4,closest_target,closest_distance
1,0.5,0.3,1.2,right,2.5,1.8,3.1,2.9,target_2,1.8
```

## Workflow

1. **Input**: 
   - `skeleton_2d.json` (Step 2)
   - `depth/*.npy` (Step 0)
   - `target_detections.json` (Step 0)

2. **Process**:
   - Convert 2D keypoints to 3D
   - Calculate pointing vectors
   - Compute distances to targets

3. **Output**:
   - `pointing_results.json`
   - `analysis_results.csv`

## Next Steps

Use `analysis_results.csv` for:
- Statistical analysis
- Plotting trajectories
- Determining pointing accuracy
- Behavioral coding
