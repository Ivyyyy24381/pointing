"""
2D Pointing Trace Visualization

Plots ground intersection points of pointing vectors over time.
Based on legacy gesture_data_process.py visualization.

Uses FIXED axis ranges for consistent comparison across trials.

Experiment setup (based on config/targets.yaml):
- Targets arranged in a CURVED ARC
- Target 1 & 2 (+X): to human's LEFT (camera's right)
- Target 3 & 4 (-X): to human's RIGHT (camera's left)
- Targets at Z: ~2.6-2.9m depth (arc curves away from camera in middle)
- Dog: CENTER of arc (inside the curve)
- Human: OUTSIDE of the curve (pointing toward targets/dog)

Camera coordinate frame:
- +X: camera's right (human's left when facing camera)
- +Y: down
- +Z: into scene (depth)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fixed axis ranges based on config/targets.yaml and typical experiment setup
# X: targets range from -1.06m to +1.16m, with margin
#    +X = camera's right = human's left (targets 1, 2)
#    -X = camera's left = human's right (targets 3, 4)
# Z: targets at ~2.6-2.9m in an arc
DEFAULT_X_RANGE = (-1.5, 1.5)   # Normal orientation: -X left, +X right
DEFAULT_Z_RANGE = (2.0, 5.0)    # Targets at ~2.8m, extended range for human/dog movement


def plot_2d_pointing_trace(analyses_dict: Dict, targets: List[Dict],
                           human_position: List[float],
                           output_path: Path,
                           trial_name: str = "",
                           fixed_xlim: Optional[Tuple[float, float]] = None,
                           fixed_zlim: Optional[Tuple[float, float]] = None,
                           use_fixed_axes: bool = True):
    """
    Plot 2D ground intersection points of pointing vectors.

    Args:
        analyses_dict: Dictionary of {frame_key: analysis_dict} from pointing analysis
        targets: List of target dictionaries with 'position_m' key (already transformed)
        human_position: [x, y, z] position of human center (transformed)
        output_path: Path to save the PNG file
        trial_name: Name for the plot title
        fixed_xlim: Optional fixed X-axis limits (x_max, x_min) - reversed for camera view
        fixed_zlim: Optional fixed Z-axis limits (z_min, z_max)
        use_fixed_axes: If True, use fixed axis ranges for consistent comparison

    Creates a 2D plot (X vs Z) showing:
    - Target positions as black X markers
    - Human position as gray circle
    - Ground intersection points for each vector, with alpha increasing over time
    """

    # Color mapping for each vector type
    color_map = {
        'eye_to_wrist': 'r',           # Red
        'shoulder_to_wrist': 'g',       # Green
        'elbow_to_wrist': 'b',          # Blue
        'nose_to_wrist': 'm'            # Magenta
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    # Collect all X and Z coordinates for automatic axis limits
    all_x = []
    all_z = []

    # Plot targets
    for target in targets:
        # Support both formats: legacy 'position_m' or new 'x', 'y', 'z'
        if 'position_m' in target:
            x, _, z = target['position_m']
        elif 'x' in target and 'y' in target and 'z' in target:
            x, y, z = target['x'], target['y'], target['z']
        else:
            continue  # Skip invalid targets

        ax.scatter(x, z, c='black', marker='x', s=200, linewidths=3)
        all_x.append(x)
        all_z.append(z)

        # Get target label
        target_label = target.get('id', target.get('label', 'target'))
        ax.text(x, z + 0.15, f"{target_label}", fontsize=11, color='black',
               ha='center', fontweight='bold')

    # Plot human center
    if human_position:
        x, _, z = human_position
        ax.scatter(x, z, c='gray', marker='o', s=300, edgecolors='black', linewidths=2)
        ax.text(x, z - 0.2, "Human", fontsize=11, color='gray',
               ha='center', fontweight='bold')
        all_x.append(x)
        all_z.append(z)

    # Sort analyses by frame number for proper alpha progression
    sorted_frames = sorted(analyses_dict.items(),
                          key=lambda item: int(item[0].split('_')[-1]))
    num_frames = len(sorted_frames)

    # Plot ground intersections with alpha increasing by frame
    for idx, (frame_key, analysis) in enumerate(sorted_frames):
        # Alpha from 0.1 to ~1.0 as frames progress (matches legacy code)
        alpha = 0.1 + 0.9 * (idx / num_frames)

        for vec_key, color in color_map.items():
            intersection_key = f'{vec_key}_ground_intersection'

            if intersection_key in analysis:
                intersection = analysis[intersection_key]

                # Check if valid (not None values)
                if (intersection and
                    len(intersection) == 3 and
                    intersection[0] is not None):

                    x, _, z = intersection
                    ax.plot(x, z, marker='.', color=color, alpha=alpha,
                           markersize=8)
                    all_x.append(x)
                    all_z.append(z)

    # Add legend
    legend_handles = [
        mpatches.Patch(color=color, label=vec_key.replace('_', ' '))
        for vec_key, color in color_map.items()
    ]
    legend_handles.append(mpatches.Patch(color='black', label='Targets'))
    legend_handles.append(mpatches.Patch(color='gray', label='Human'))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=10,
             framealpha=0.9)

    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z (m) - depth', fontsize=12, fontweight='bold')

    # Set axis limits
    if use_fixed_axes:
        # Use fixed ranges for consistent comparison across trials
        xlim = fixed_xlim if fixed_xlim is not None else DEFAULT_X_RANGE
        zlim = fixed_zlim if fixed_zlim is not None else DEFAULT_Z_RANGE
        ax.set_xlim(xlim)
        ax.set_ylim(zlim)
    elif all_x and all_z:
        # Dynamic axis limits using percentiles to ignore outliers
        x_min, x_max = np.percentile(all_x, 5), np.percentile(all_x, 95)
        z_min, z_max = np.percentile(all_z, 5), np.percentile(all_z, 95)

        # Also include target positions (always show targets)
        for target in targets:
            if 'position_m' in target:
                tx, _, tz = target['position_m']
            elif 'x' in target and 'z' in target:
                tx, tz = target['x'], target['z']
            else:
                continue
            x_min, x_max = min(x_min, tx), max(x_max, tx)
            z_min, z_max = min(z_min, tz), max(z_max, tz)

        # Add 10% margin on each side
        x_range = x_max - x_min
        z_range = z_max - z_min
        x_margin = max(0.2, x_range * 0.1)
        z_margin = max(0.2, z_range * 0.1)

        # Normal X orientation: -X left (human's right), +X right (human's left)
        ax.set_xlim([x_min - x_margin, x_max + x_margin])
        ax.set_ylim([z_min - z_margin, z_max + z_margin])
    else:
        # Fallback to default fixed ranges
        ax.set_xlim(DEFAULT_X_RANGE)
        ax.set_ylim(DEFAULT_Z_RANGE)

    # Compute and display statistics for noise assessment
    stats_text = ""
    if all_x and all_z:
        x_std = np.std(all_x)
        z_std = np.std(all_z)
        x_range_actual = np.ptp(all_x)
        z_range_actual = np.ptp(all_z)
        n_points = len(all_x)

        # Count outliers (points outside typical range based on config/targets.yaml)
        x_outliers = sum(1 for x in all_x if x < -1.5 or x > 1.5)
        z_outliers = sum(1 for z in all_z if z < 2.0 or z > 5.0)

        stats_text = (f"N={n_points} | X: std={x_std:.2f}m, range={x_range_actual:.2f}m | "
                     f"Z: std={z_std:.2f}m, range={z_range_actual:.2f}m")
        if x_outliers > 0 or z_outliers > 0:
            stats_text += f" | Outliers: {x_outliers + z_outliers}"

    # Add info text at bottom
    info_text = "Fixed axes" if use_fixed_axes else "Auto axes"
    if stats_text:
        info_text = f"{info_text} | {stats_text}"
    ax.text(0.5, 0.02, info_text, transform=ax.transAxes,
            fontsize=8, color='gray', alpha=0.8, ha='center')

    title = f'{trial_name} 2D Ground Intersection Points of Pointing Vectors' if trial_name else '2D Ground Intersection Points of Pointing Vectors'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Saved 2D pointing trace: {output_path}")
