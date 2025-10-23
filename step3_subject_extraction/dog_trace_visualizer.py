#!/usr/bin/env python3
"""
Dog 2D Trace Visualizer

Creates a top-down view (X-Z plane) of the dog's movement trace with:
- Rainbow colored dots showing the dog's path over time
- Target locations as gray squares
- Grid and axis labels
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class DogTraceVisualizer:
    """Visualize dog movement trace in 2D top-down view."""

    def __init__(self):
        """Initialize visualizer."""
        pass

    def create_trace_plot(self,
                         dog_results_path: Path,
                         targets_path: Optional[Path],
                         output_image_path: Path,
                         title: str = "2D Trace (Top View) - Dog",
                         figsize: Tuple[int, int] = (10, 8)):
        """
        Create 2D top-down trace visualization.

        Args:
            dog_results_path: Path to dog_detection_results.json
            targets_path: Optional path to target_detections_cam_frame.json
            output_image_path: Path to save the output image
            title: Plot title
            figsize: Figure size (width, height)
        """
        # Load dog results
        with open(dog_results_path) as f:
            dog_data = json.load(f)

        if not dog_data:
            print("⚠️ No dog detection data found")
            return

        # Load targets if available
        targets = []
        if targets_path and targets_path.exists():
            with open(targets_path) as f:
                targets_json = json.load(f)
                if isinstance(targets_json, list):
                    targets = targets_json
                else:
                    targets = targets_json.get('targets', [])

        # Extract trace points (X, Z coordinates) from dog keypoints_3d
        # ONLY use points with valid depth data (3D coordinates)
        trace_points = []
        frame_keys = sorted(dog_data.keys())

        for frame_key in frame_keys:
            dog_result = dog_data[frame_key]
            keypoints_3d = dog_result.get('keypoints_3d')

            # Only use 3D keypoints with valid depth data
            if keypoints_3d and len(keypoints_3d) > 0:
                # Use nose (index 0) for trace - it typically has the most reliable depth data
                kp = keypoints_3d[0]

                if len(kp) >= 3:
                    x, y, z = kp[0], kp[1], kp[2]
                    # Check if this is a valid 3D point (not all zeros)
                    if abs(x) > 0.001 or abs(y) > 0.001 or abs(z) > 0.001:
                        # Top-down view: X-Z plane (X is left-right, Z is depth/forward-back)
                        trace_points.append([x, z])

        if not trace_points:
            print("⚠️ No valid 3D trace points found (depth data missing)")
            return

        trace_points = np.array(trace_points)

        # Remove outliers using IQR method
        if len(trace_points) > 4:
            # Calculate IQR for X and Z separately
            q1_x = np.percentile(trace_points[:, 0], 25)
            q3_x = np.percentile(trace_points[:, 0], 75)
            iqr_x = q3_x - q1_x

            q1_z = np.percentile(trace_points[:, 1], 25)
            q3_z = np.percentile(trace_points[:, 1], 75)
            iqr_z = q3_z - q1_z

            # Define outlier bounds (1.5 * IQR is standard)
            lower_x = q1_x - 1.5 * iqr_x
            upper_x = q3_x + 1.5 * iqr_x
            lower_z = q1_z - 1.5 * iqr_z
            upper_z = q3_z + 1.5 * iqr_z

            # Filter outliers
            mask = (
                (trace_points[:, 0] >= lower_x) & (trace_points[:, 0] <= upper_x) &
                (trace_points[:, 1] >= lower_z) & (trace_points[:, 1] <= upper_z)
            )

            n_outliers = len(trace_points) - np.sum(mask)
            if n_outliers > 0:
                print(f"   Removed {n_outliers} outlier points from trace")

            trace_points = trace_points[mask]

            if len(trace_points) == 0:
                print("⚠️ No valid trace points after outlier removal")
                return

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Plot targets as gray squares
        if targets:
            target_xs = []
            target_zs = []
            target_labels = []

            for target in targets:
                x = target.get('x', 0)
                z = target.get('z', 0)
                label = target.get('label', 'target')

                if x != 0 or z != 0:  # Valid target
                    target_xs.append(x)
                    target_zs.append(z)
                    target_labels.append(label)

            # Plot target positions
            ax.scatter(target_xs, target_zs,
                      c='gray', marker='s', s=200,
                      label='Targets', zorder=5,
                      edgecolors='black', linewidths=1.5)

            # Add target labels
            for i, (x, z, label) in enumerate(zip(target_xs, target_zs, target_labels)):
                ax.annotate(label,
                           xy=(x, z),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=10,
                           fontweight='bold')

        # Create rainbow colors for the trace (from blue to red over time)
        n_points = len(trace_points)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_points))

        # Fit a smooth curve through the trace points
        from scipy.interpolate import make_interp_spline

        # Create parameter t for parametric spline
        t = np.arange(n_points)

        # Fit smooth spline using UnivariateSpline with smoothing (doesn't pass through all points)
        try:
            # Need at least 4 points for smoothing spline
            if n_points >= 4:
                from scipy.interpolate import UnivariateSpline

                # Use UnivariateSpline which allows smoothing (s parameter)
                # s controls smoothing: 0 = interpolate through all points, higher = smoother but less exact
                # Good smoothing value is typically between n_points and n_points * sqrt(2*n_points)
                smoothing_factor = n_points * 2.0  # Adjust this for more/less smoothing

                # Fit smoothing splines for X and Z separately
                spl_x = UnivariateSpline(t, trace_points[:, 0], s=smoothing_factor, k=3)
                spl_z = UnivariateSpline(t, trace_points[:, 1], s=smoothing_factor, k=3)

                # Create smooth parameter values for plotting
                t_smooth = np.linspace(0, n_points - 1, n_points * 20)
                x_smooth = spl_x(t_smooth)
                z_smooth = spl_z(t_smooth)

                # Create rainbow colors for smooth curve
                colors_smooth = plt.cm.rainbow(np.linspace(0, 1, len(t_smooth)))

                # Draw smooth curve with gradient colors
                for i in range(len(t_smooth) - 1):
                    ax.plot([x_smooth[i], x_smooth[i+1]],
                           [z_smooth[i], z_smooth[i+1]],
                           color=colors_smooth[i], alpha=0.7, linewidth=3.0, zorder=2)
            else:
                # Not enough points for spline, draw straight lines
                for i in range(n_points - 1):
                    ax.plot([trace_points[i, 0], trace_points[i+1, 0]],
                           [trace_points[i, 1], trace_points[i+1, 1]],
                           color=colors[i], alpha=0.7, linewidth=2.5, zorder=2)
        except Exception as e:
            print(f"⚠️ Spline fitting failed: {e}, using linear interpolation")
            # Fallback to straight lines
            for i in range(n_points - 1):
                ax.plot([trace_points[i, 0], trace_points[i+1, 0]],
                       [trace_points[i, 1], trace_points[i+1, 1]],
                       color=colors[i], alpha=0.7, linewidth=2.5, zorder=2)

        # Plot trace as colored dots ON TOP of the curve
        ax.scatter(trace_points[:, 0], trace_points[:, 1],
                  c=colors, s=80, alpha=0.9, zorder=4,
                  edgecolors='white', linewidths=0.5)

        # Set labels and title
        ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z (meters)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Set axis limits based on target locations with margin
        if targets:
            target_xs = [t.get('x', 0) for t in targets if (t.get('x', 0) != 0 or t.get('z', 0) != 0)]
            target_zs = [t.get('z', 0) for t in targets if (t.get('x', 0) != 0 or t.get('z', 0) != 0)]

            if target_xs and target_zs:
                # Calculate range based on targets
                min_x = min(target_xs)
                max_x = max(target_xs)
                min_z = min(target_zs)
                max_z = max(target_zs)

                # Add margin (20% of range or at least 0.5m)
                x_range = max_x - min_x
                z_range = max_z - min_z
                x_margin = max(x_range * 0.2, 0.5)
                z_margin = max(z_range * 0.2, 0.5)

                ax.set_xlim(min_x - x_margin, max_x + x_margin)
                ax.set_ylim(min_z - z_margin, max_z + z_margin)

        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Add legend
        if targets:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # Add colorbar to show time progression
        sm = plt.cm.ScalarMappable(cmap='rainbow',
                                   norm=plt.Normalize(vmin=0, vmax=n_points-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Frame Progression', fontsize=10, fontweight='bold')

        # Tight layout
        plt.tight_layout()

        # Save figure
        plt.savefig(output_image_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        print(f"✅ Saved 2D trace visualization: {output_image_path}")
        print(f"   Trace points: {n_points}")
        print(f"   Targets: {len(targets)}")

    def create_combined_trace_plot(self,
                                  dog_results_path: Path,
                                  human_results_path: Optional[Path],
                                  targets_path: Optional[Path],
                                  output_image_path: Path,
                                  title: str = "2D Trace (Top View) - Dog & Human"):
        """
        Create combined 2D trace showing both dog and human movement.

        Args:
            dog_results_path: Path to dog_detection_results.json
            human_results_path: Path to skeleton_2d.json
            targets_path: Path to target_detections_cam_frame.json
            output_image_path: Path to save output image
            title: Plot title
        """
        # Load dog results
        with open(dog_results_path) as f:
            dog_data = json.load(f)

        # Load human results if available
        human_data = {}
        if human_results_path and human_results_path.exists():
            with open(human_results_path) as f:
                human_data = json.load(f)

        # Load targets
        targets = []
        if targets_path and targets_path.exists():
            with open(targets_path) as f:
                targets_json = json.load(f)
                if isinstance(targets_json, list):
                    targets = targets_json
                else:
                    targets = targets_json.get('targets', [])

        # Extract dog trace - ONLY use points with valid depth data
        dog_trace = []
        for frame_key in sorted(dog_data.keys()):
            dog_result = dog_data[frame_key]
            keypoints_3d = dog_result.get('keypoints_3d')
            if keypoints_3d and len(keypoints_3d) > 0:
                # Use nose (index 0) - most reliable for depth
                kp = keypoints_3d[0]
                if len(kp) >= 3:
                    x, y, z = kp[0], kp[1], kp[2]
                    # Only include if we have valid depth data (not all zeros)
                    if abs(x) > 0.001 or abs(y) > 0.001 or abs(z) > 0.001:
                        dog_trace.append([x, z])

        # Extract human trace - ONLY use points with valid depth data
        human_trace = []
        for frame_key in sorted(human_data.keys()):
            human_result = human_data[frame_key]
            landmarks_3d = human_result.get('landmarks_3d')
            if landmarks_3d and len(landmarks_3d) > 0:
                # Use nose (index 0)
                kp = landmarks_3d[0]
                if len(kp) >= 3:
                    x, y, z = kp[0], kp[1], kp[2]
                    # Only include if we have valid depth data (not all zeros)
                    if abs(x) > 0.001 or abs(y) > 0.001 or abs(z) > 0.001:
                        human_trace.append([x, z])

        # Remove outliers from dog trace using IQR method
        if dog_trace and len(dog_trace) > 4:
            dog_trace = np.array(dog_trace)
            q1_x = np.percentile(dog_trace[:, 0], 25)
            q3_x = np.percentile(dog_trace[:, 0], 75)
            iqr_x = q3_x - q1_x
            q1_z = np.percentile(dog_trace[:, 1], 25)
            q3_z = np.percentile(dog_trace[:, 1], 75)
            iqr_z = q3_z - q1_z

            lower_x = q1_x - 1.5 * iqr_x
            upper_x = q3_x + 1.5 * iqr_x
            lower_z = q1_z - 1.5 * iqr_z
            upper_z = q3_z + 1.5 * iqr_z

            mask = (
                (dog_trace[:, 0] >= lower_x) & (dog_trace[:, 0] <= upper_x) &
                (dog_trace[:, 1] >= lower_z) & (dog_trace[:, 1] <= upper_z)
            )
            dog_trace = dog_trace[mask]

        # Remove outliers from human trace using IQR method
        if human_trace and len(human_trace) > 4:
            human_trace = np.array(human_trace)
            q1_x = np.percentile(human_trace[:, 0], 25)
            q3_x = np.percentile(human_trace[:, 0], 75)
            iqr_x = q3_x - q1_x
            q1_z = np.percentile(human_trace[:, 1], 25)
            q3_z = np.percentile(human_trace[:, 1], 75)
            iqr_z = q3_z - q1_z

            lower_x = q1_x - 1.5 * iqr_x
            upper_x = q3_x + 1.5 * iqr_x
            lower_z = q1_z - 1.5 * iqr_z
            upper_z = q3_z + 1.5 * iqr_z

            mask = (
                (human_trace[:, 0] >= lower_x) & (human_trace[:, 0] <= upper_x) &
                (human_trace[:, 1] >= lower_z) & (human_trace[:, 1] <= upper_z)
            )
            human_trace = human_trace[mask]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Plot targets
        if targets:
            target_xs = [t.get('x', 0) for t in targets if (t.get('x', 0) != 0 or t.get('z', 0) != 0)]
            target_zs = [t.get('z', 0) for t in targets if (t.get('x', 0) != 0 or t.get('z', 0) != 0)]
            target_labels = [t.get('label', 'target') for t in targets if (t.get('x', 0) != 0 or t.get('z', 0) != 0)]

            ax.scatter(target_xs, target_zs,
                      c='gray', marker='s', s=200,
                      label='Targets', zorder=5,
                      edgecolors='black', linewidths=1.5)

            for x, z, label in zip(target_xs, target_zs, target_labels):
                ax.annotate(label, xy=(x, z), xytext=(5, 5),
                           textcoords='offset points', fontsize=10, fontweight='bold')

        # Plot dog trace (rainbow)
        if dog_trace:
            dog_trace = np.array(dog_trace)
            n_dog = len(dog_trace)
            colors_dog = plt.cm.rainbow(np.linspace(0, 1, n_dog))

            ax.scatter(dog_trace[:, 0], dog_trace[:, 1],
                      c=colors_dog, s=50, alpha=0.8, zorder=3,
                      edgecolors='none', label='Dog')

            for i in range(n_dog - 1):
                ax.plot([dog_trace[i, 0], dog_trace[i+1, 0]],
                       [dog_trace[i, 1], dog_trace[i+1, 1]],
                       color=colors_dog[i], alpha=0.5, linewidth=1.5, zorder=2)

        # Plot human trace (green gradient)
        if human_trace:
            human_trace = np.array(human_trace)
            n_human = len(human_trace)
            colors_human = plt.cm.Greens(np.linspace(0.3, 1, n_human))

            ax.scatter(human_trace[:, 0], human_trace[:, 1],
                      c=colors_human, s=50, alpha=0.8, zorder=4,
                      edgecolors='none', label='Human')

            for i in range(n_human - 1):
                ax.plot([human_trace[i, 0], human_trace[i+1, 0]],
                       [human_trace[i, 1], human_trace[i+1, 1]],
                       color=colors_human[i], alpha=0.5, linewidth=1.5, zorder=3)

        # Labels and formatting
        ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z (meters)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Set axis limits based on target locations with margin
        if targets:
            target_xs = [t.get('x', 0) for t in targets if (t.get('x', 0) != 0 or t.get('z', 0) != 0)]
            target_zs = [t.get('z', 0) for t in targets if (t.get('x', 0) != 0 or t.get('z', 0) != 0)]

            if target_xs and target_zs:
                min_x = min(target_xs)
                max_x = max(target_xs)
                min_z = min(target_zs)
                max_z = max(target_zs)

                x_range = max_x - min_x
                z_range = max_z - min_z
                x_margin = max(x_range * 0.2, 0.5)
                z_margin = max(z_range * 0.2, 0.5)

                ax.set_xlim(min_x - x_margin, max_x + x_margin)
                ax.set_ylim(min_z - z_margin, max_z + z_margin)

        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(output_image_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        print(f"✅ Saved combined 2D trace: {output_image_path}")


def main():
    """Test the visualizer."""
    import argparse

    parser = argparse.ArgumentParser(description='Create dog 2D trace visualization')
    parser.add_argument('--dog-results', type=str, required=True,
                       help='Path to dog_detection_results.json')
    parser.add_argument('--human-results', type=str,
                       help='Path to skeleton_2d.json (optional)')
    parser.add_argument('--targets', type=str,
                       help='Path to target_detections_cam_frame.json')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image path')
    parser.add_argument('--combined', action='store_true',
                       help='Create combined dog+human trace')
    parser.add_argument('--title', type=str, default='2D Trace (Top View)',
                       help='Plot title')

    args = parser.parse_args()

    visualizer = DogTraceVisualizer()

    if args.combined and args.human_results:
        visualizer.create_combined_trace_plot(
            dog_results_path=Path(args.dog_results),
            human_results_path=Path(args.human_results),
            targets_path=Path(args.targets) if args.targets else None,
            output_image_path=Path(args.output),
            title=args.title
        )
    else:
        visualizer.create_trace_plot(
            dog_results_path=Path(args.dog_results),
            targets_path=Path(args.targets) if args.targets else None,
            output_image_path=Path(args.output),
            title=args.title
        )


if __name__ == '__main__':
    main()
