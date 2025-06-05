import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import os
import re
import pandas as pd
from scipy.spatial.transform import Rotation as R

def regenerate_trace3d_from_csv(csv_path, fx, fy, cx, cy, depth_folder, output_csv_path=None):
    def get_valid_depth(depth_frame, u, v, patch_size=5):
        h, w = depth_frame.shape
        u, v = int(u), int(v)
        half = patch_size // 2
        patch = depth_frame[max(0, v - half):min(h, v + half + 1), max(0, u - half):min(w, u + half + 1)]
        valid = patch[patch > 0]
        return np.median(valid) if len(valid) > 0 else None

    def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        return X, Y, Z

    def interpolate_trace(trace_3d):
        trace_array = np.array(trace_3d, dtype=np.float32)
        for dim in range(3):
            col = trace_array[:, dim]
            nans = np.isnan(col)
            not_nans = ~nans
            if np.sum(not_nans) >= 2:
                col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
            trace_array[:, dim] = col
        return trace_array.tolist()

    df = pd.read_csv(csv_path)
    trace_3d = []

    for i, row in df.iterrows():
        frame_idx = int(row["frame_index"]) if "frame_index" in row else i
        u = row.get("nose_x", np.nan)
        v = row.get("nose_y", np.nan)

        if pd.isna(u) or pd.isna(v):
            trace_3d.append([np.nan, np.nan, np.nan])
            continue

        depth_file = os.path.join(depth_folder, f"_Depth_Color_{frame_idx:04d}.raw")
        if not os.path.exists(depth_file):
            trace_3d.append([np.nan, np.nan, np.nan])
            continue

        with open(depth_file, 'rb') as f:
            raw = np.frombuffer(f.read(), dtype=np.uint16).reshape((720, 1280))  # adjust to your resolution
            depth_frame = raw / 1000.0

        depth = get_valid_depth(depth_frame, u, v)
        if depth is None:
            trace_3d.append([np.nan, np.nan, np.nan])
        else:
            x, y, z = pixel_to_3d(u, v, depth, fx, fy, cx, cy)
            trace_3d.append([x, y, z])

    trace_3d = interpolate_trace(trace_3d)
    df["trace3d_x"], df["trace3d_y"], df["trace3d_z"] = zip(*trace_3d)

    if output_csv_path is None:
        output_csv_path = csv_path.replace(".csv", "_trace3d_updated.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"Trace3D regenerated and saved to {output_csv_path}")
    return output_csv_path

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
threshold = .35
smoothing_window = 3
def process_distance_data(csv_path, start_time, end_time=None):
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
    last_below_time = None
    # Improved: Find the last continuous block below threshold
    for i in range(1, len(below)):
        if below.iloc[i] and not below.iloc[i - 1]:
            drop_start = df['time_sec'].iloc[i]
        if below.iloc[i]:
            last_below_time = df['time_sec'].iloc[i]
    # Set drop_end to last time below threshold (end of last continuous block)
    if drop_start and last_below_time:
        drop_end = last_below_time
    elif drop_start:
        drop_end = df['time_sec'].iloc[-1]
    else:
        drop_end = df['time_sec'].max()

    print(f"Auto-trim Start: {start_time}, Auto-trim End: {drop_end}")

    if end_time is None:
        end_time = drop_end

    df = df[(df['time_sec'] >= start_time) & (df['time_sec'] <= end_time)].copy()

    # Remove outlier distances > 3.5 meters
    distance_mask = (df[distance_cols] <= 3.5).all(axis=1)
    df = df[distance_mask]



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
            smoothed_avg = smoothed.rolling(window=smoothing_window, min_periods=1).mean()

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
    output_plot = os.path.splitext(csv_path)[0] + "_distance_plot.png"
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
    # if 'time_sec' in df.columns:
    #     df = df[(df['time_sec'] >= args.start) & (df['time_sec'] <= args.end)].copy()
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
    x_smooth = pd.Series(x_aligned).rolling(window=smoothing_window, min_periods=1).mean().values
    y_smooth = pd.Series(y_aligned).rolling(window=smoothing_window, min_periods=1).mean().values
    z_smooth = pd.Series(z_aligned).rolling(window=smoothing_window, min_periods=1).mean().values

    from scipy.interpolate import splprep, splev
    cmap = plt.cm.rainbow
    # Remove invalid (nan) points for spline fitting
    valid_3d = ~np.isnan(x_smooth) & ~np.isnan(y_smooth) & ~np.isnan(z_smooth)
    x_smooth_valid = x_smooth[valid_3d]
    y_smooth_valid = y_smooth[valid_3d]
    z_smooth_valid = z_smooth[valid_3d]
    num_points = len(x_smooth_valid)
    # Sample a subset of valid points for spline fitting (minimum 5 or n//5)
    if num_points >= 5:
        sample_n = max(5, num_points // 5)
        rng = np.random.default_rng(42)
        idxs = np.linspace(0, num_points - 1, sample_n).astype(int)
        xs_sample = x_smooth_valid[idxs]
        ys_sample = y_smooth_valid[idxs]
        zs_sample = z_smooth_valid[idxs]
        spline_degree = min(3, len(xs_sample) - 1)
        try:
            tck, u = splprep([xs_sample, zs_sample, ys_sample], s=0.5, k=spline_degree)
            unew = np.linspace(0, 1.0, 300)
            out = splev(unew, tck)
            # Rainbow color along spline
            for i in range(len(unew)-1):
                color = cmap(unew[i])
                ax.plot(out[0][i:i+2], out[1][i:i+2], out[2][i:i+2], color=color, linewidth=2)
        except Exception as e:
            print(f"Skipping 3D spline fitting due to error: {e}")
    else:
        print("Not enough valid points for 3D spline fitting. Skipping spline.")

    # Plot original data points as rainbow-colored dots
    colors = [cmap(u) for u in np.linspace(0, 1.0, len(x_aligned))]
    for i in range(len(x_aligned)):
        ax.plot([x_aligned.iloc[i]], [z_aligned.iloc[i]], [y_aligned.iloc[i]], 'o', color=colors[i])

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
    plot_path_3d =os.path.splitext(csv_path)[0]+"_trace_3d.png"
    plt.savefig(plot_path_3d)
    plt.close()
    print(f"3D trace plot with aligned coordinates saved: {plot_path_3d}")

    #######################
    ## Plot 2D trace (top down)
    #######################
    fig2, ax2 = plt.subplots()

    # Plot original data points as rainbow-colored dots
    for i in range(len(x_aligned)):
        ax2.plot([x_aligned.iloc[i]], [z_aligned.iloc[i]], 'o', color=colors[i])

    # Remove NaNs before fitting
    valid_2d = ~np.isnan(x_smooth) & ~np.isnan(z_smooth)
    x_smooth_2d = x_smooth[valid_2d]
    z_smooth_2d = z_smooth[valid_2d]
    num_points_2d = len(x_smooth_2d)
    # Sample a subset of valid points for spline fitting (minimum 5 or n//5)
    if num_points_2d >= 5:
        sample_n_2d = max(5, num_points_2d // 5)
        idxs_2d = np.linspace(0, num_points_2d - 1, sample_n_2d).astype(int)
        xs2d_sample = x_smooth_2d[idxs_2d]
        zs2d_sample = z_smooth_2d[idxs_2d]
        spline_degree_2d = min(3, len(xs2d_sample) - 1)
        try:
            tck_2d, u_2d = splprep([xs2d_sample, zs2d_sample], s=0.5, k=spline_degree_2d)
            unew_2d = np.linspace(0, 1.0, 300)
            out_2d = splev(unew_2d, tck_2d)
            for i in range(len(unew_2d)-1):
                color = cmap(unew_2d[i])
                ax2.plot(out_2d[0][i:i+2], out_2d[1][i:i+2], color=color, linewidth=2)
        except Exception as e:
            print(f"Skipping 2D spline fitting due to error: {e!r}")
    else:
        print("Not enough valid points for 2D spline fitting. Skipping spline.")

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
    ax2.set_xlim([-2, 1])
    ax2.set_ylim([0.5,3.5])
    ax2.set_title(f"2D Trace (Top View) - {dog_name}")
    ax2.grid(True)

    fig2.tight_layout()
    plot_path_2d = os.path.splitext(csv_path)[0]+ "_trace2d.png"
    
    plt.savefig(plot_path_2d)
    plt.close()
    print(f"2D trace plot with aligned coordinates saved: {plot_path_2d}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    dog_name = 'Kenzi'
    dog_id = 'BDL127'
    trial_number = 10
    start_time = 3
    end_time = 4.5
    parser.add_argument("--csv", type=str, default = f'dog_data/{dog_id}_{dog_name}_side/{trial_number}/processed_dog_result_table.csv', help="Path to the processed dog CSV file")
    parser.add_argument("--start", type=float, default=start_time, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=end_time, help="Start time in seconds")
    parser.add_argument("--side_view", action="store_true", help="Use side view to select config JSON")
    args = parser.parse_args()

    csv_path = args.csv

    # First process the data to generate the valid_search column and trim time window
    updated_csv_path, dog_id, dog_name, trial_number, target_location = process_distance_data(csv_path, start_time=args.start, end_time = end_time)
    fx, fy, cx, cy = 643.2109985351562, 642.4398803710938, 641.45458984375, 359.393798828125  # ← your intrinsics
    depth_folder = f"dog_data/{dog_id}_{dog_name}_side/{trial_number}/Depth_Color"
    
    # updated_csv_path = regenerate_trace3d_from_csv(updated_csv_path, fx, fy, cx, cy, depth_folder)
    
    # updated_csv_path = f'dog_data/{dog_id}_{dog_name}_side/{trial_number}/processed_dog_result_table_with_metadata.csv'
    # Then plot the processed data
    plot_distance_data(updated_csv_path, dog_name)
    plot_trace(updated_csv_path, dog_name, dog_id, trial_number, side_view=True)