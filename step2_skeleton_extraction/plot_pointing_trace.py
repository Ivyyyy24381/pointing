"""
2D Pointing Trace Visualization

Plots ground intersection points of pointing vectors over time.
Based on legacy gesture_data_process.py visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List


def plot_2d_pointing_trace(analyses_dict: Dict, targets: List[Dict],
                           human_position: List[float],
                           output_path: Path,
                           trial_name: str = ""):
    """
    Plot 2D ground intersection points of pointing vectors.

    Args:
        analyses_dict: Dictionary of {frame_key: analysis_dict} from pointing analysis
        targets: List of target dictionaries with 'position_m' key (already transformed)
        human_position: [x, y, z] position of human center (transformed)
        output_path: Path to save the PNG file
        trial_name: Name for the plot title

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
    ax.set_ylabel('Z (m)', fontsize=12, fontweight='bold')

    # Set axis limits automatically based on data with 10% margin
    if all_x and all_z:
        import numpy as np
        x_min, x_max = np.min(all_x), np.max(all_x)
        z_min, z_max = np.min(all_z), np.max(all_z)

        # Add 10% margin on each side
        x_range = x_max - x_min
        z_range = z_max - z_min
        x_margin = max(0.2, x_range * 0.1)  # At least 20cm margin
        z_margin = max(0.2, z_range * 0.1)

        ax.set_xlim([x_max + x_margin, x_min - x_margin])  # Reversed X for camera view
        ax.set_ylim([z_min - z_margin, z_max + z_margin])
    else:
        # Fallback to default if no data
        ax.set_xlim([1.5, -1.5])
        ax.set_ylim([-2, 2.5])

    title = f'{trial_name} 2D Ground Intersection Points of Pointing Vectors' if trial_name else '2D Ground Intersection Points of Pointing Vectors'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Saved 2D pointing trace: {output_path}")
