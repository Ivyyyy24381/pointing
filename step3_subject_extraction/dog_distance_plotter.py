"""
Dog Distance to Targets Plotter
Generates time-series plot showing Euclidean distance from dog to each target over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


class DogDistancePlotter:
    """Creates distance vs time plots for dog-to-target distances."""

    def __init__(self):
        self.target_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    def create_distance_plot(self, csv_path: Path, output_image_path: Path,
                            title: str = "Dog Distance to Targets Over Time"):
        """
        Create a plot showing Euclidean distance from dog to each target over time.

        Args:
            csv_path: Path to processed_dog_result_table.csv
            output_image_path: Where to save the output PNG
            title: Plot title
        """
        # Read CSV
        df = pd.read_csv(csv_path)

        # Extract time and distances
        time_sec = df['time_sec'].values

        # Check which target columns exist
        target_columns = []
        for i in range(1, 5):
            col_name = f'target_{i}_r'
            if col_name in df.columns:
                target_columns.append(col_name)

        if not target_columns:
            print("⚠️ No target distance columns found in CSV")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each target's distance
        for idx, col_name in enumerate(target_columns):
            target_num = int(col_name.split('_')[1])
            distances = df[col_name].values

            # Remove any NaN, zero, or invalid values (indicates missing depth data)
            valid_mask = ~np.isnan(distances) & (distances > 0.01)
            valid_time = time_sec[valid_mask]
            valid_dist = distances[valid_mask]

            # Remove outliers using IQR method
            if len(valid_dist) > 4:
                q1 = np.percentile(valid_dist, 25)
                q3 = np.percentile(valid_dist, 75)
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outlier_mask = (valid_dist >= lower_bound) & (valid_dist <= upper_bound)
                n_outliers = len(valid_dist) - np.sum(outlier_mask)

                if n_outliers > 0:
                    print(f"   Target {target_num}: Removed {n_outliers} outlier points")

                valid_time = valid_time[outlier_mask]
                valid_dist = valid_dist[outlier_mask]

            if len(valid_dist) > 0:
                # Fit smoothed trend line using UnivariateSpline
                if len(valid_dist) >= 4:
                    from scipy.interpolate import UnivariateSpline

                    # Sort by time (should already be sorted, but ensure it)
                    sort_idx = np.argsort(valid_time)
                    sorted_time = valid_time[sort_idx]
                    sorted_dist = valid_dist[sort_idx]

                    # Use smoothing spline to get trend line (not passing through all points)
                    smoothing_factor = len(sorted_dist) * 2.0  # Adjust for more/less smoothing
                    try:
                        spline = UnivariateSpline(sorted_time, sorted_dist, s=smoothing_factor, k=3)

                        # Generate smooth curve
                        time_smooth = np.linspace(sorted_time.min(), sorted_time.max(), len(sorted_time) * 10)
                        dist_smooth = spline(time_smooth)

                        # Plot smoothed trend line
                        ax.plot(time_smooth, dist_smooth,
                               color=self.target_colors[idx],
                               linewidth=2.5,
                               alpha=0.8,
                               label=f'Target {target_num}',
                               zorder=2)
                    except:
                        # Fallback to direct line if smoothing fails
                        ax.plot(sorted_time, sorted_dist,
                               color=self.target_colors[idx],
                               linewidth=2,
                               alpha=0.7,
                               label=f'Target {target_num}')
                else:
                    # Not enough points for smoothing, use direct line
                    ax.plot(valid_time, valid_dist,
                           color=self.target_colors[idx],
                           linewidth=2,
                           alpha=0.7,
                           label=f'Target {target_num}')

                # Plot scatter points on top
                ax.scatter(valid_time, valid_dist,
                          color=self.target_colors[idx],
                          s=30,
                          alpha=0.5,
                          edgecolors='white',
                          linewidths=0.5,
                          zorder=3)

        # Formatting
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Euclidean Distance (meters)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set reasonable y-axis limits
        all_distances = []
        for col_name in target_columns:
            distances = df[col_name].values
            valid_dist = distances[~np.isnan(distances) & (distances > 0.01)]
            if len(valid_dist) > 0:
                all_distances.extend(valid_dist)

        if all_distances:
            min_dist = min(all_distances)
            max_dist = max(all_distances)
            margin = (max_dist - min_dist) * 0.1
            ax.set_ylim(max(0, min_dist - margin), max_dist + margin)

        plt.tight_layout()
        plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved distance plot: {output_image_path}")
        print(f"   Targets plotted: {len(target_columns)}")
        print(f"   Time range: {time_sec.min():.2f}s - {time_sec.max():.2f}s")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dog_distance_plotter.py <csv_path> [output_path]")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = csv_path.parent / "dog_distance_to_targets.png"

    plotter = DogDistancePlotter()
    plotter.create_distance_plot(csv_path, output_path)
