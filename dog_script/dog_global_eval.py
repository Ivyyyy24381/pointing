import pandas as pd
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import json
import re

root_dir = "dog_data/BDL244_Hannah_side/"
dog_match = re.search(r'(BDL\d+)', root_dir)
dog_id = dog_match.group(1) if dog_match else None
metadata_csv = 'PVP_Comprehension_Data.csv'
pvp_df = pd.read_csv(metadata_csv)
meta_row = pvp_df[(pvp_df['participant_id'] == dog_id)]
dog_name = meta_row.iloc[0]['dog_name'] if 'dog_name' in meta_row.columns else "Unknown"

side_view = True
if side_view:
    target_json = 'target_coords_side.json'
else:
    target_json = 'target_coords_front.json'

# Load target coordinates
with open(target_json, 'r') as f:
    target_data = json.load(f)

target_points = []
for entry in target_data:
    if entry['label'].startswith("target"):
        target_points.append([entry['x'], entry['y'], entry['z']])
target_points = np.array(target_points).reshape(-1, 3)

trace_3d_per_target = {1: [], 2: [], 3: [], 4: []}
trace_2d_per_target = {1: [], 2: [], 3: [], 4: []}
color_per_target = {1: [], 2: [], 3: [], 4: []}
# For mapping each trial's dataframe to its target (for extracting trial_number)
trial_dfs_per_target = {1: [], 2: [], 3: [], 4: []}

all_csvs = []
smoothing_factor = 5

# List all folders in root_dir
all_folders = glob.glob(os.path.join(root_dir, "*"))

for folder in all_folders:
    # Extract just the folder name (e.g., "1" from ".../1/")
    folder_name = os.path.basename(folder)
    if folder_name.isdigit():   # Only if folder is a number
        csv_path = os.path.join(folder, "processed_dog_result_table_with_metadata.csv")
        if os.path.exists(csv_path):
            all_csvs.append(csv_path)

print("CSV files found:", all_csvs)

all_data = []
for csv in all_csvs:
    df = pd.read_csv(csv)
    df['source_csv'] = csv  # Keep track of where it came from
    all_data.append(df)

global_df = pd.concat(all_data, ignore_index=True)
global_df.to_csv(root_dir+"global_cleaned_data.csv", index=False)
print("✅ Saved global_cleaned_data.csv with", len(global_df), "rows.")


dogs = global_df['dog_id'].unique()

for dog in dogs:
    df_dog = global_df[global_df['dog_id'] == dog]

    for trial in df_dog['trial_number'].unique():
        df_trial = df_dog[df_dog['trial_number'] == trial]

        trial_meta = meta_row[meta_row['trial_number'] == trial]
        if not trial_meta.empty:
            assigned_location = int(trial_meta.iloc[0]['location'])
        else:
            assigned_location = None

        # Prepare the rainbow color map based on time progression
        time = df_trial['time_sec'].values
        norm = plt.Normalize(time.min(), time.max())
        colors = cm.rainbow(norm(time))

        # Compute smoothed rolling mean for x, y, z
        smoothed_x = df_trial['trace3d_x'].rolling(window=smoothing_factor, min_periods=1, center=True).mean().values
        smoothed_y = df_trial['trace3d_y'].rolling(window=smoothing_factor, min_periods=1, center=True).mean().values
        smoothed_z = df_trial['trace3d_z'].rolling(window=smoothing_factor, min_periods=1, center=True).mean().values

        # 2D plot (x vs z) with rainbow color
        smoothed_x_2d = df_trial['trace3d_x'].rolling(window=smoothing_factor, min_periods=1, center=True).mean().values
        smoothed_z_2d = df_trial['trace3d_z'].rolling(window=smoothing_factor, min_periods=1, center=True).mean().values

        if assigned_location in [1, 2, 3, 4] and not df_trial.empty:
            trial_trace_3d = np.vstack([smoothed_x, smoothed_y, smoothed_z]).T
            trial_trace_2d = np.vstack([smoothed_x_2d, smoothed_z_2d]).T
            trace_3d_per_target[assigned_location].append(trial_trace_3d)
            trace_2d_per_target[assigned_location].append(trial_trace_2d)
            color_per_target[assigned_location].append(colors)
            # Store the df for this trial for label extraction
            trial_dfs_per_target[assigned_location].append(df_trial)

for tgt in [1, 2, 3, 4]:
    if trace_3d_per_target[tgt]:
        # Plot 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # --- Prepare fixed colors for splines using tab10/tab20 colormap ---
        n_trials = len(trace_3d_per_target[tgt])
        tab_cmap = plt.get_cmap('tab10' if n_trials <= 10 else 'tab20')
        spline_colors = [tab_cmap(i % tab_cmap.N) for i in range(n_trials)]

        for idx, trial_3d in enumerate(trace_3d_per_target[tgt]):
            time = np.arange(len(trial_3d))
            norm = plt.Normalize(time.min(), time.max())
            colors = cm.rainbow(norm(time))
            # Rainbow scatter for original data points
            ax.scatter(trial_3d[:,0], trial_3d[:,1], trial_3d[:,2], c=colors, marker='o')

        # --- Insert 3D spline visualization ---
        from scipy.interpolate import splprep, splev
        # To avoid duplicate legend entries, only add label for the spline, not for the scatter
        for idx, trial_3d in enumerate(trace_3d_per_target[tgt]):
            if trial_3d.shape[0] >= 3:
                try:
                    tck, u = splprep([trial_3d[:, 0], trial_3d[:, 1], trial_3d[:, 2]], s=0.5)
                    unew = np.linspace(0, 1.0, 300)
                    out = splev(unew, tck)
                    # Extract actual trial_number from the corresponding dataframe
                    if idx < len(trial_dfs_per_target[tgt]):
                        trial_df = trial_dfs_per_target[tgt][idx]
                        if 'trial_number' in trial_df.columns and not trial_df.empty:
                            trial_number = trial_df['trial_number'].iloc[0]
                        else:
                            trial_number = idx + 1
                    else:
                        trial_number = idx + 1
                    # Plot the entire spline with a fixed color for this trial, add label for legend
                    ax.plot(out[0], out[1], out[2], color=spline_colors[idx], linewidth=2, alpha=0.8, label=f'Trial {trial_number}')
                except Exception as e:
                    print(f"3D Spline fitting failed for a trial in target {tgt}: {e}")
        # --- End 3D spline visualization ---

        for i, point in enumerate(target_points):
            ax.scatter(point[0], point[1], point[2], c='grey', marker='s', s=80)
            ax.text(point[0], point[1], point[2], f"Target {i+1}", color='black', fontsize=10, weight='bold')
            if i+1 == tgt:
                # Draw a translucent circle for the current target
                circle = plt.Circle((point[0], point[1]), 0.2, color='orange', alpha=0.3)
                ax.add_patch(circle)
                art3d.pathpatch_2d_to_3d(circle, z=point[2], zdir="z")

        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_zlabel("Z (meters)")
        # Only show unique legend entries (avoid duplicate trial labels)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title(f"Combined 3D Trace for Target {tgt} - {dog_id} {dog_name}")
        plt.savefig(root_dir+f"global_trace_target_{tgt}_3d.png")
        plt.close()

        # Plot 2D
        plt.figure()
        n_trials_2d = len(trace_2d_per_target[tgt])
        tab_cmap_2d = plt.get_cmap('tab10' if n_trials_2d <= 10 else 'tab20')
        spline_colors_2d = [tab_cmap_2d(i % tab_cmap_2d.N) for i in range(n_trials_2d)]
        for idx, trial_2d in enumerate(trace_2d_per_target[tgt]):
            time = np.arange(len(trial_2d))
            norm = plt.Normalize(time.min(), time.max())
            colors = cm.rainbow(norm(time))
            # Rainbow scatter for original data points
            plt.scatter(trial_2d[:,0], trial_2d[:,1], c=colors, marker='o')

        # --- Insert 2D spline visualization ---
        for idx, trial_2d in enumerate(trace_2d_per_target[tgt]):
            if trial_2d.shape[0] >= 3:
                # Remove duplicate consecutive points
                unique_2d = [(trial_2d[0,0], trial_2d[0,1])]
                for xi, zi in zip(trial_2d[1:,0], trial_2d[1:,1]):
                    if not np.allclose([xi, zi], unique_2d[-1]):
                        unique_2d.append((xi, zi))
                # Fallback if too few unique points after duplicate removal
                if len(unique_2d) >= 2:
                    x_unique, z_unique = zip(*unique_2d)
                else:
                    # Fallback: use original points without duplicate removal
                    print(f"Not enough unique points after duplicate removal for trial {idx+1} in target {tgt}, using original points.")
                    x_unique = trial_2d[:, 0]
                    z_unique = trial_2d[:, 1]
                spline_degree_2d = min(3, len(x_unique) - 1)
                try:
                    tck_2d, u_2d = splprep([x_unique, z_unique], s=0.5, k=spline_degree_2d)
                    unew_2d = np.linspace(0, 1.0, 300)
                    out_2d = splev(unew_2d, tck_2d)
                    if idx < len(trial_dfs_per_target[tgt]):
                        trial_df = trial_dfs_per_target[tgt][idx]
                        if 'trial_number' in trial_df.columns and not trial_df.empty:
                            trial_number = trial_df['trial_number'].iloc[0]
                        else:
                            trial_number = idx + 1
                    else:
                        trial_number = idx + 1
                    plt.plot(out_2d[0], out_2d[1], color=spline_colors_2d[idx], linewidth=2, alpha=0.8, label=f'Trial {trial_number}')
                except Exception as e:
                    if idx < len(trial_dfs_per_target[tgt]):
                        trial_df = trial_dfs_per_target[tgt][idx]
                        if 'trial_number' in trial_df.columns and not trial_df.empty:
                            trial_number = trial_df['trial_number'].iloc[0]
                        else:
                            trial_number = idx + 1
                    else:
                        trial_number = idx + 1
                    print(f"2D Spline fitting failed for trial {idx+1} in target {tgt}, drawing simple line instead: {e}")
                    print(f"Fallback line X coords: {x_unique}")
                    print(f"Fallback line Z coords: {z_unique}")
                    plt.plot(x_unique, z_unique, color=spline_colors_2d[idx], linewidth=2, alpha=0.8, label=f'Trial {trial_number}')
                    plt.scatter(x_unique, z_unique, c='red', marker='x', s=50)
        # --- End 2D spline visualization ---

        for i, point in enumerate(target_points):
            plt.scatter(point[0], point[2], c='grey', marker='s', s=80)
            plt.text(point[0], point[2] + 0.05, f"Target {i+1}", color='black', fontsize=10, weight='bold')
            if i+1 == tgt:
                circle = plt.Circle((point[0], point[2]), 0.1, color='red', alpha=0.3)
                plt.gca().add_patch(circle)

        plt.xlabel("X (meters)")
        plt.ylabel("Z (meters)")
        # Only show unique legend entries (avoid duplicate trial labels)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title(f"Combined 2D Trace for Target {tgt} - {dog_id} {dog_name}")
        plt.savefig(root_dir+f"global_trace_target_{tgt}_2d.png")
        plt.close()