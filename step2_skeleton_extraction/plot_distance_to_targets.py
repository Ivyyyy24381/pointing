"""
Distance to Target Visualization.

Plots the distance from each pointing representation's ground intersection
to each target over time. Helps analyze pointing accuracy and consistency.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Color mapping for each pointing representation
VECTOR_COLORS = {
    'eye_to_wrist': '#e41a1c',           # Red
    'shoulder_to_wrist': '#377eb8',       # Blue
    'elbow_to_wrist': '#4daf4a',          # Green
    'nose_to_wrist': '#984ea3',           # Purple
    'head_orientation': '#ff7f00',        # Orange
}

# Line styles for filtered vs raw
LINE_STYLES = {
    'raw': '-',
    'filtered': '--',
}


def plot_distance_to_targets(analyses_dict: Dict,
                             targets: List[Dict],
                             output_path: Path,
                             trial_name: str = "",
                             show_filtered: bool = True,
                             figsize: Tuple = (14, 10)):
    """
    Plot distance from each pointing representation to each target over time.

    Creates a multi-panel plot:
    - One subplot per target
    - Multiple lines per subplot (one for each pointing representation)
    - Shows both raw and filtered (if available) trajectories

    Args:
        analyses_dict: Dictionary of {frame_key: analysis_dict}
        targets: List of target dictionaries
        output_path: Path to save the PNG file
        trial_name: Name for the plot title
        show_filtered: If True, also plot filtered trajectories (dashed lines)
        figsize: Figure size tuple
    """

    if not analyses_dict or not targets:
        print("⚠️ No data to plot")
        return

    # Sort frames by frame number
    sorted_frames = sorted(analyses_dict.items(),
                          key=lambda item: int(item[0].split('_')[-1]))

    frame_numbers = [int(k.split('_')[-1]) for k, _ in sorted_frames]
    num_frames = len(frame_numbers)
    num_targets = len(targets)

    # Create subplots: one row per target
    fig, axes = plt.subplots(num_targets, 1, figsize=figsize, sharex=True)
    if num_targets == 1:
        axes = [axes]

    # Pointing representations to plot
    vec_names = ['eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist',
                 'nose_to_wrist', 'head_orientation']

    # Extract distances for each target and representation
    for target_idx, (ax, target) in enumerate(zip(axes, targets)):
        target_label = target.get('id', target.get('label', f'Target {target_idx + 1}'))

        for vec_name in vec_names:
            color = VECTOR_COLORS.get(vec_name, 'gray')

            # Raw distances
            dist_key = f'{vec_name}_dist_to_target_{target_idx + 1}'
            distances = []

            for _, analysis in sorted_frames:
                if analysis and dist_key in analysis:
                    dist = analysis[dist_key]
                    distances.append(dist if dist is not None else np.nan)
                else:
                    distances.append(np.nan)

            # Plot raw distances
            if not all(np.isnan(distances)):
                ax.plot(frame_numbers, distances,
                       color=color, linestyle=LINE_STYLES['raw'],
                       linewidth=1.5, alpha=0.7,
                       label=f'{vec_name.replace("_", " ")} (raw)')

            # Filtered distances (if available)
            if show_filtered:
                # Check if filtered data exists
                filtered_key = f'{vec_name}_ground_intersection_filtered'
                has_filtered = any(
                    analysis and filtered_key in analysis
                    for _, analysis in sorted_frames
                )

                if has_filtered:
                    # Need to recompute distances from filtered intersections
                    filtered_distances = []
                    for _, analysis in sorted_frames:
                        if analysis and filtered_key in analysis:
                            intersection = analysis[filtered_key]
                            if intersection and intersection[0] is not None:
                                # Compute distance to target
                                if 'x' in target and 'z' in target:
                                    target_pos = np.array([target['x'], target.get('y', 0), target['z']])
                                elif 'position_m' in target:
                                    target_pos = np.array(target['position_m'])
                                else:
                                    filtered_distances.append(np.nan)
                                    continue

                                int_pos = np.array(intersection)
                                dist = np.linalg.norm(int_pos - target_pos)
                                filtered_distances.append(dist)
                            else:
                                filtered_distances.append(np.nan)
                        else:
                            filtered_distances.append(np.nan)

                    if not all(np.isnan(filtered_distances)):
                        ax.plot(frame_numbers, filtered_distances,
                               color=color, linestyle=LINE_STYLES['filtered'],
                               linewidth=2.0, alpha=1.0,
                               label=f'{vec_name.replace("_", " ")} (filtered)')

        # Subplot formatting
        ax.set_ylabel(f'Distance to {target_label} (m)', fontsize=10)
        ax.set_ylim(0, 5)  # Fixed Y range for consistency
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add horizontal lines at common thresholds
        ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=1.0, color='orange', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=2.0, color='red', linestyle=':', alpha=0.5, linewidth=1)

        # Add threshold annotations on first subplot only
        if target_idx == 0:
            ax.text(frame_numbers[0], 0.5, ' 0.5m', fontsize=8, color='green',
                   va='center', alpha=0.7)
            ax.text(frame_numbers[0], 1.0, ' 1.0m', fontsize=8, color='orange',
                   va='center', alpha=0.7)
            ax.text(frame_numbers[0], 2.0, ' 2.0m', fontsize=8, color='red',
                   va='center', alpha=0.7)

    # Only show legend on first subplot
    axes[0].legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    # Common X axis label
    axes[-1].set_xlabel('Frame Number', fontsize=11, fontweight='bold')

    # Title
    title = f'{trial_name} - Distance to Targets Over Time' if trial_name else 'Distance to Targets Over Time'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Saved distance to targets plot: {output_path}")


def plot_distance_summary(analyses_dict: Dict,
                          targets: List[Dict],
                          output_path: Path,
                          trial_name: str = "",
                          figsize: Tuple = (12, 8)):
    """
    Plot summary statistics of distance to each target for each pointing representation.

    Creates a bar chart showing mean distance ± std for each vector type per target.

    Args:
        analyses_dict: Dictionary of {frame_key: analysis_dict}
        targets: List of target dictionaries
        output_path: Path to save the PNG file
        trial_name: Name for the plot title
        figsize: Figure size tuple
    """
    from typing import Tuple

    if not analyses_dict or not targets:
        print("⚠️ No data to plot")
        return

    vec_names = ['eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist',
                 'nose_to_wrist', 'head_orientation']
    num_targets = len(targets)

    # Compute statistics for each vector-target combination
    stats = {}  # {vec_name: {target_idx: {'mean': x, 'std': y}}}

    for vec_name in vec_names:
        stats[vec_name] = {}
        for target_idx in range(num_targets):
            dist_key = f'{vec_name}_dist_to_target_{target_idx + 1}'
            distances = []

            for _, analysis in analyses_dict.items():
                if analysis and dist_key in analysis:
                    dist = analysis[dist_key]
                    if dist is not None and not np.isnan(dist):
                        distances.append(dist)

            if distances:
                stats[vec_name][target_idx] = {
                    'mean': np.mean(distances),
                    'std': np.std(distances),
                    'min': np.min(distances),
                    'count': len(distances)
                }
            else:
                stats[vec_name][target_idx] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'count': 0
                }

    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(num_targets)
    width = 0.15
    offsets = np.linspace(-2*width, 2*width, len(vec_names))

    for i, vec_name in enumerate(vec_names):
        means = [stats[vec_name][t]['mean'] for t in range(num_targets)]
        stds = [stats[vec_name][t]['std'] for t in range(num_targets)]
        color = VECTOR_COLORS.get(vec_name, 'gray')

        bars = ax.bar(x + offsets[i], means, width,
                     yerr=stds, capsize=3,
                     label=vec_name.replace('_', ' '),
                     color=color, alpha=0.8)

    # Labels and formatting
    ax.set_ylabel('Mean Distance to Target (m)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Target', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.get('id', t.get('label', f'T{i+1}'))
                       for i, t in enumerate(targets)])

    # Add horizontal reference lines
    ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5)
    ax.axhline(y=1.0, color='orange', linestyle=':', alpha=0.5)
    ax.axhline(y=2.0, color='red', linestyle=':', alpha=0.5)

    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_ylim(0, max(3, ax.get_ylim()[1]))
    ax.grid(True, axis='y', alpha=0.3)

    title = f'{trial_name} - Mean Distance to Targets by Pointing Method' if trial_name else 'Mean Distance to Targets by Pointing Method'
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Saved distance summary plot: {output_path}")


def plot_best_representation_analysis(analyses_dict: Dict,
                                       targets: List[Dict],
                                       output_path: Path,
                                       trial_name: str = "",
                                       figsize: Tuple = (10, 8)):
    """
    Analyze which pointing representation is most accurate for each target.

    Creates a heatmap showing which method achieves lowest mean distance to each target.

    Args:
        analyses_dict: Dictionary of {frame_key: analysis_dict}
        targets: List of target dictionaries
        output_path: Path to save the PNG file
        trial_name: Name for the plot title
        figsize: Figure size tuple
    """
    from typing import Tuple

    if not analyses_dict or not targets:
        print("⚠️ No data to plot")
        return

    vec_names = ['eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist',
                 'nose_to_wrist', 'head_orientation']
    vec_labels = [v.replace('_', '\n') for v in vec_names]
    num_targets = len(targets)

    # Compute mean distances
    distance_matrix = np.zeros((len(vec_names), num_targets))

    for i, vec_name in enumerate(vec_names):
        for j in range(num_targets):
            dist_key = f'{vec_name}_dist_to_target_{j + 1}'
            distances = []

            for _, analysis in analyses_dict.items():
                if analysis and dist_key in analysis:
                    dist = analysis[dist_key]
                    if dist is not None and not np.isnan(dist):
                        distances.append(dist)

            if distances:
                distance_matrix[i, j] = np.mean(distances)
            else:
                distance_matrix[i, j] = np.nan

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Mask NaN values
    masked_matrix = np.ma.masked_invalid(distance_matrix)

    im = ax.imshow(masked_matrix, cmap='RdYlGn_r', aspect='auto',
                  vmin=0, vmax=3)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, label='Mean Distance (m)')

    # Set ticks
    ax.set_xticks(np.arange(num_targets))
    ax.set_yticks(np.arange(len(vec_names)))
    ax.set_xticklabels([t.get('id', t.get('label', f'T{i+1}'))
                       for i, t in enumerate(targets)])
    ax.set_yticklabels(vec_labels)

    # Add text annotations
    for i in range(len(vec_names)):
        for j in range(num_targets):
            value = distance_matrix[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > 1.5 else 'black'
                ax.text(j, i, f'{value:.2f}m',
                       ha='center', va='center', color=text_color, fontsize=10)

    # Highlight best (minimum) for each target
    for j in range(num_targets):
        col = distance_matrix[:, j]
        if not np.all(np.isnan(col)):
            best_idx = np.nanargmin(col)
            ax.add_patch(plt.Rectangle((j-0.5, best_idx-0.5), 1, 1,
                                       fill=False, edgecolor='blue',
                                       linewidth=3))

    ax.set_xlabel('Target', fontsize=11, fontweight='bold')
    ax.set_ylabel('Pointing Representation', fontsize=11, fontweight='bold')

    title = f'{trial_name} - Pointing Accuracy Comparison' if trial_name else 'Pointing Accuracy Comparison'
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Saved best representation analysis: {output_path}")

    # Print summary
    print("\n📊 Best pointing representation per target:")
    for j, target in enumerate(targets):
        target_label = target.get('id', target.get('label', f'Target {j+1}'))
        col = distance_matrix[:, j]
        if not np.all(np.isnan(col)):
            best_idx = np.nanargmin(col)
            best_vec = vec_names[best_idx]
            best_dist = col[best_idx]
            print(f"  {target_label}: {best_vec} ({best_dist:.2f}m)")
