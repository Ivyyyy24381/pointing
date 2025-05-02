import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
from scipy.spatial.transform import Rotation as R

def align_points_to_ground_plane(points, plane_normal):
    z_axis = np.array([0, 0, 1])
    if np.allclose(plane_normal, z_axis):
        return points

    rotation_vector = np.cross(plane_normal, z_axis)
    if np.linalg.norm(rotation_vector) == 0:
        return points

    angle = np.arccos(np.clip(np.dot(plane_normal, z_axis) / (np.linalg.norm(plane_normal) * np.linalg.norm(z_axis)), -1.0, 1.0))
    rotation = R.from_rotvec(rotation_vector / np.linalg.norm(rotation_vector) * angle)
    return rotation.apply(points)

metadata_csv = 'PVP_Comprehension_Data.csv'
threshold = 0.15
smoothing_window = 3
def process_distance_data(csv_path, start_time):
    dog_match = re.search(r'(BDL\d+)', csv_path)
    dog_id = dog_match.group(1) if dog_match else None
    trial_number = int(os.path.basename(os.path.dirname(csv_path)))

    df = pd.read_csv(csv_path)
    distance_cols = [col for col in df.columns if col.endswith("_r")]

    # Smooth once only
    for col in distance_cols:
        df[col] = df[col].rolling(window=smoothing_window, min_periods=1).mean()

    pvp_df = pd.read_csv(metadata_csv)
    meta_row = pvp_df[
        (pvp_df['participant_id'] == dog_id) &
        (pvp_df['trial_number'] == trial_number)
    ]
    dog_name = meta_row.iloc[0]['dog_name'] if 'dog_name' in meta_row.columns else "Unknown"

    if meta_row.empty:
        raise ValueError("No matching trial in metadata!")

    target_location = int(meta_row.iloc[0]['location'])
    distance_col = f'target_{target_location}_r'

    smoothed = df[distance_col]
    below = smoothed < threshold
    drop_start = None
    drop_end = None

    for i in range(1, len(below)):
        if below.iloc[i] and not below.iloc[i - 1]:
            drop_start = df['time_sec'].iloc[i]
        if drop_start and not below.iloc[i] and below.iloc[i - 1]:
            drop_end = df['time_sec'].iloc[i]
            break  # Stop at the first crossing up

    if drop_start and drop_end:
        end_time_auto = drop_end
    elif drop_start:
        end_time_auto = df['time_sec'].iloc[-1]
    else:
        end_time_auto = df['time_sec'].max()

    df = df[(df['time_sec'] >= start_time) & (df['time_sec'] <= end_time_auto)].copy()

    # valid_search flag removed
    df['touched_target'] = 0

    # Loop over target distances correctly
    for target_num in range(1, 5):  # Assuming targets 1 to 4
        col = f"target_{target_num}_r"
        if col in df.columns:
            below_mask = df[col] < threshold
            df.loc[below_mask, 'touched_target'] = target_num

    df['dog_id'] = dog_id
    df['dog_name'] = dog_name
    df['trial_number'] = trial_number

    # Reorder columns: dog_id, dog_name, trial_number first
    first_cols = ['dog_id', 'dog_name', 'trial_number', 'touched_target']
    other_cols = [col for col in df.columns if col not in first_cols]
    df = df[first_cols + other_cols]

    updated_csv_path = csv_path.replace("_table.csv", "_table_with_metadata.csv")
    df.to_csv(updated_csv_path, index=False)
    print(f"Updated CSV saved: {updated_csv_path}")

    return updated_csv_path, dog_id, dog_name, trial_number, target_location


def plot_distance_data(csv_path, dog_name):
    from scipy.interpolate import make_interp_spline
    df = pd.read_csv(csv_path)
    time_col = df["time_sec"]
    distance_cols = [col for col in df.columns if col.endswith("_r")]

    dog_match = re.search(r'(BDL\d+)', csv_path)
    dog_id = dog_match.group(1) if dog_match else None
    trial_number = int(os.path.basename(os.path.dirname(csv_path)))

    fig, ax = plt.subplots()

    for col in distance_cols:
        smoothed = df[col]  # Already smoothed from earlier step
        mask = df.index >= 0  # Always true now since no trimming

        if "human" in col:
            ax.plot(time_col[mask], smoothed[mask], 'x', label=col.replace("_r", ""), color="gray")
        else:
            ax.plot(time_col[mask], smoothed[mask], 'o', label=col.replace("_r", ""))

            below_threshold = smoothed < threshold
            for i in range(1, len(below_threshold)):
                if below_threshold.iloc[i] and not below_threshold.iloc[i - 1]:
                    ax.annotate(f" {col.replace('_r', '')}",
                                (time_col.iloc[i], smoothed.iloc[i]),
                                textcoords="offset points", xytext=(0, 5), ha='center')

            # Rolling weighted average instead of spline
            smoothed_avg = smoothed.rolling(window=3, min_periods=1).mean()

            # Plot raw data dots again with lower alpha
            ax.plot(time_col[mask], smoothed[mask], 'o', color=ax.get_lines()[-1].get_color(), markersize=4, alpha=0.4)

            # Plot smoothed line
            ax.plot(time_col[mask], smoothed_avg[mask], linestyle='solid',
                    color=ax.get_lines()[-1].get_color(), alpha=0.6)

    ax.axhline(y=threshold, color='black', linestyle='dashed', label='Threshold')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Distance (meters)")
    ax.set_title(f"Dog {dog_name} ({dog_id}) Trial {trial_number} - Distance to Targets Over Time")
    ax.legend()
    ax.grid(True)
    output_plot = csv_path.replace("_table_with_metadata.csv", "_distance_plot.png")
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.close()

from mpl_toolkits.mplot3d import Axes3D

def plot_trace(csv_path, dog_name, dog_id, trial_number = None, side_view = True):
    if side_view:
        target_json = 'target_coords_side.json'
    else:
        target_json = 'target_coords_front.json'
    df = pd.read_csv(csv_path)

    # Use the trimmed df already
    x = df['trace3d_x']
    y = df['trace3d_y']
    z = df['trace3d_z']

    # Remove invalid (NaN or 0,0,0) points
    valid_mask = ~( (x==0) & (y==0) & (z==0) )
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

    import json

    # Load target coordinates
    with open(target_json, 'r') as f:
        target_data = json.load(f)

    target_points = []
    for entry in target_data:
        if entry['label'].startswith("target"):
            target_points.append([entry['x'], entry['y'], entry['z']])
    target_points = np.array(target_points).reshape(-1, 3)

    human_pos = None
    for entry in target_data:
        if entry['label'] == 'human':
            human_pos = np.array([entry['x'], entry['y'], entry['z']])
            break

    # Commented out plane fitting and alignment code
    # if target_points.shape[0] >= 3:
    #     # Fit plane: ax + by + cz + d = 0
    #     A = np.c_[target_points[:, 0], target_points[:, 1], np.ones(target_points.shape[0])]
    #     C, _, _, _ = np.linalg.lstsq(A, target_points[:, 2], rcond=None)
    #     C[2] += 0.5
    #     # z = C[0]*x + C[1]*y + C[2]
    #
    #     plane_normal = np.array([-C[0], -C[1], 1])
    #     plane_normal = plane_normal / np.linalg.norm(plane_normal)
    #
    #     # Align trace points
    #     trace_points = np.vstack([x.values, y.values, z.values]).T
    #     aligned_trace = align_points_to_ground_plane(trace_points, plane_normal)
    #     x_aligned, y_aligned, z_aligned = aligned_trace[:, 0], aligned_trace[:, 1], aligned_trace[:, 2]
    #
    #     # Update dataframe with aligned trace (replace original columns)
    #     df['trace3d_x'] = x_aligned
    #     df['trace3d_y'] = y_aligned
    #     df['trace3d_z'] = z_aligned
    #
    #     # Save cleaned CSV
    #     cleaned_csv = csv_path.replace("_table_with_metadata.csv", "_cleaned_data.csv")
    #     df.to_csv(cleaned_csv, index=False)
    #     print(f"Cleaned CSV with aligned coordinates saved to: {cleaned_csv}")
    # else:
    #     x_aligned, y_aligned, z_aligned = x, y, z

    x_aligned, y_aligned, z_aligned = x, y, z

    #######################
    ## Plot 3D trace
    #######################
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Prepare points and smooth with rolling window
    x_smooth = pd.Series(x_aligned).rolling(window=3, min_periods=1).mean().values
    y_smooth = pd.Series(y_aligned).rolling(window=3, min_periods=1).mean().values
    z_smooth = pd.Series(z_aligned).rolling(window=3, min_periods=1).mean().values
    
    # Fit spline
    from scipy.interpolate import splprep, splev
    tck, u = splprep([x_smooth, z_smooth, y_smooth], s=0.5)
    unew = np.linspace(0, 1.0, 300)
    out = splev(unew, tck)

    # Rainbow color along spline
    cmap = plt.cm.rainbow
    for i in range(len(unew)-1):
        color = cmap(unew[i])
        ax.plot(out[0][i:i+2], out[1][i:i+2], out[2][i:i+2], color=color, linewidth=2)

    # Plot original data points as dots with colors
    colors = [cmap(u) for u in np.linspace(0, 1.0, len(x_aligned))]
    for i in range(len(x_aligned)-1):
        ax.plot(x_aligned[i:i+2], z_aligned[i:i+2], y_aligned[i:i+2], 'o', color=colors[i])

    # Plot target points as cubes (marker='s')
    ax.scatter(target_points[:, 0], target_points[:, 2], target_points[:, 1], c='grey', marker='s', s=100, label='Targets')
    # Label each target with its ID
    for i, entry in enumerate(target_data):
        if entry['label'].startswith("target"):
            ax.text(entry['x'], entry['z'], entry['y'] + 0.05, entry['label'], color='black')

    # Mark and label human position
    # if human_pos is not None:
    #     ax.scatter(human_pos[0], human_pos[2], human_pos[1], c='red', marker='^', s=120, label='Human')
    #     ax.text(human_pos[0], human_pos[2], human_pos[1] + 0.05, 'Human', color='red')

    # Plot target locations
    # Use first available trial's target coordinates for consistency
    df_trial = df
    for target_num in range(1, 5):
        tx = df_trial.get(f'target_{target_num}_x')
        ty = df_trial.get(f'target_{target_num}_y')
        tz = df_trial.get(f'target_{target_num}_z')
        if tx is not None and ty is not None and tz is not None:
            target_x = tx.dropna().iloc[0] if not tx.dropna().empty else None
            target_y = ty.dropna().iloc[0] if not ty.dropna().empty else None
            target_z = tz.dropna().iloc[0] if not tz.dropna().empty else None
            if target_x is not None and target_y is not None and target_z is not None:
                # For 3D plot
                ax.scatter(target_x, target_y, target_z, c='grey', marker='s', s=80)
                ax.text(target_x, target_y, target_z, f'Target {target_num}', color='black')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_zlabel('Y (meters)')
    ax.set_title(f"3D Trace - {dog_id} {dog_name} - Trial {trial_number}")
    if not side_view:
        ax.set_xlim([-1, 2])
        ax.set_ylim([2.0, 4])
        ax.set_zlim([0, 2])
        ax.view_init(elev=45, azim=135)
    else:
        ax.view_init(elev=-145, azim=60)
    ax.legend()
    fig.tight_layout()
    plot_path_3d = csv_path.replace("_table_with_metadata.csv", "_trace3d.png")
    plt.savefig(plot_path_3d)
    plt.close()
    print(f"3D trace plot with aligned coordinates saved: {plot_path_3d}")

    #######################
    ## Plot 2D trace (top down)
    #######################
    fig2, ax2 = plt.subplots()

    # Plot original data points as dots with colors
    for i in range(len(x_aligned)-1):
        ax2.plot(x_aligned[i:i+2], z_aligned[i:i+2], 'o', color=colors[i])

    # Remove NaNs before fitting
    valid_2d = ~np.isnan(x_smooth) & ~np.isnan(z_smooth)
    x_smooth_2d = x_smooth[valid_2d]
    z_smooth_2d = z_smooth[valid_2d]

    if len(x_smooth_2d) >= 3 and len(np.unique(x_smooth_2d)) >= 2 and len(np.unique(z_smooth_2d)) >= 2:
        try:
            tck_2d, u_2d = splprep([x_smooth_2d, z_smooth_2d], s=0.5)
            unew_2d = np.linspace(0, 1.0, 300)
            out_2d = splev(unew_2d, tck_2d)
            for i in range(len(unew_2d)-1):
                color = cmap(unew_2d[i])
                ax2.plot(out_2d[0][i:i+2], out_2d[1][i:i+2], color=color, linewidth=2)
        except Exception as e:
            print(f"Skipping 2D spline fitting due to error: {e}")
    else:
        print("Not enough unique valid points for spline fitting in 2D. Skipping spline.")

    # Plot original data points as dots with colors on top of spline
    for i in range(len(x_aligned)-1):
        ax2.plot(x_aligned[i:i+2], z_aligned[i:i+2], 'o', color=colors[i])

    # Add target points
    ax2.scatter(target_points[:, 0], target_points[:, 2], c='grey', marker='s', s=100, label='Targets')
    # Label each target with its ID in 2D plot
    for i, entry in enumerate(target_data):
        if entry['label'].startswith("target"):
            ax2.text(entry['x'], entry['z'] + 0.05, entry['label'], color='black')

    # Mark and label human position in 2D
    # if human_pos is not None:
    #     ax2.scatter(human_pos[0], human_pos[2], c='red', marker='^', s=120, label='Human')
    #     ax2.text(human_pos[0], human_pos[2] + 0.05, 'Human', color='red')

    # Plot target locations
    for target_num in range(1, 5):
        tx = df_trial.get(f'target_{target_num}_x')
        ty = df_trial.get(f'target_{target_num}_y')
        tz = df_trial.get(f'target_{target_num}_z')
        if tx is not None and ty is not None and tz is not None:
            target_x = tx.dropna().iloc[0] if not tx.dropna().empty else None
            target_y = ty.dropna().iloc[0] if not ty.dropna().empty else None
            target_z = tz.dropna().iloc[0] if not tz.dropna().empty else None
            if target_x is not None and target_y is not None and target_z is not None:
                # For 2D plot
                ax2.scatter(target_x, target_z, c='grey', marker='s', s=80)
                ax2.text(target_x, target_z + 0.05, f'Target {target_num}', color='black')

    ax2.legend()
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Z (meters)')
    ax2.set_title(f"2D Trace (Top View) - {dog_name}")
    ax2.grid(True)

    fig2.tight_layout()
    plot_path_2d = csv_path.replace("_table_with_metadata.csv", "_trace2d.png")
    plt.savefig(plot_path_2d)
    plt.close()
    print(f"2D trace plot with aligned coordinates saved: {plot_path_2d}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    trial_number = 1
    start_time = 3
    parser.add_argument("--csv", type=str, default = f'dog_data/BDL204_Waffles-CAM2/{trial_number}/processed_dog_result_table.csv', help="Path to the processed dog CSV file")
    parser.add_argument("--start", type=float, default=start_time, help="Start time in seconds")
    parser.add_argument("--side_view", action="store_true", help="Use side view to select config JSON")
    args = parser.parse_args()

    csv_path = args.csv

    # First process the data to generate the valid_search column and trim time window
    # updated_csv_path, dog_id, dog_name, trial_number, target_location = process_distance_data(csv_path, start_time=args.start)
    dog_name = 'Waffles'
    dog_id = 'BDL204'
    updated_csv_path = f'dog_data/BDL204_Waffles-CAM2/{trial_number}/processed_dog_result_table_with_metadata.csv'
    # Then plot the processed data
    plot_distance_data(updated_csv_path, dog_name)
    plot_trace(updated_csv_path, dog_name, dog_id, trial_number, side_view=True)