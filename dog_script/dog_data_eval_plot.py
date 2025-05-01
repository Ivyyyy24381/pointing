import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_distance_to_targets(csv_path, start_time=None, end_time=None):
    df = pd.read_csv(csv_path)
    if start_time is not None and end_time is not None:
        df = df[(df["time_sec"] >= start_time) & (df["time_sec"] <= end_time)]

    time_col = df["time_sec"]
    fig, ax = plt.subplots()
    for col in df.columns:
        if col.endswith("_r"):
            smoothed = df[col].rolling(window=5, min_periods=1).mean()
            if "human" in col:
                ax.plot(time_col, smoothed, label=col.replace("_r", ""), color="gray", linestyle="--")
            else:
                ax.plot(time_col, smoothed, label=col.replace("_r", ""))
                threshold = 0.15
                below_threshold = smoothed < threshold
                for i in range(1, len(below_threshold)):
                    if below_threshold.iloc[i] and not below_threshold.iloc[i - 1]:
                        time_val = time_col.iloc[i]
                        dist_val = smoothed.iloc[i]
                        ax.annotate(f" {col.replace('_r', '')}", (time_val, dist_val),
                                    textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color="black")
    ax.axhline(y=threshold, color='black', linestyle='dashed', linewidth=1, label='Threshold (0.15 m)')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Distance (meters)")
    ax.set_title("Dog Distance to Targets Over Time")
    ax.legend()
    ax.grid(True)
    out_path = csv_path.replace("_table.csv", "_distance_comparison_from_csv.png")
    plt.savefig(out_path)
    print(f"Saved distance comparison plot to: {out_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to the processed dog CSV file")
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--side_view", action="store_true", help="Use side view to select config JSON")
    args = parser.parse_args()

    csv_path = args.csv
    # plot_trace_from_csv(csv_path, start_time=args.start, end_time=args.end, side_view=args.side_view)
    plot_distance_to_targets(csv_path, start_time=args.start, end_time=args.end)