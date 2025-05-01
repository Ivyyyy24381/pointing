import sys
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading
from queue import Queue, Empty
from collections import deque
import time
sys.path.append('code/')
from bag_to_video import run_rs_convert, run_ffmpeg_convert, concat_videos
from batch_split_bag import generate_and_run_commands
from shutil import copy2
import shutil

# Add smoothing using a deque
def moving_average(signal, window_size=10):
    window = deque(maxlen=window_size)
    for val in signal:
        window.append(val)
        yield sum(window) / len(window)
        
class RedirectOutput:
    """Class to redirect stdout and stderr to a tkinter Text widget."""
    def __init__(self, text_widget, queue):
        self.text_widget = text_widget
        self.queue = queue

    def write(self, message):
        self.queue.put(message)

    def flush(self):
        pass


class RosbagSlicerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rosbag Slicer")
        
        # Setup Queue for handling stdout and stderr
        self.queue = Queue()
        self.update_terminal()
        
        # Rosbag file path
        self.rosbag_path = None
        self.csv_path = None

        # Checkbox for automatic slicing
        self.auto_slice_var = tk.BooleanVar()
        self.auto_slice_checkbox = tk.Checkbutton(root, text="Automatic Slice", variable=self.auto_slice_var)
        self.auto_slice_checkbox.pack()

        # Label and button for rosbag file
        self.rosbag_label = tk.Label(root, text="Load Rosbag File:")
        self.rosbag_label.pack()
        self.rosbag_button = tk.Button(root, text="Upload Rosbag", command=self.load_rosbag)
        self.rosbag_button.pack()

        # Optional Start time (default 0)
        self.start_time_label = tk.Label(root, text="Start Time (Optional, default 0):")
        self.start_time_label.pack()
        self.start_time_entry = tk.Entry(root)
        self.start_time_entry.insert(0, "0")
        self.start_time_entry.pack()

        # Optional End time (default 2000)
        self.end_time_label = tk.Label(root, text="End Time (Optional, default full video):")
        self.end_time_label.pack()
        self.end_time_entry = tk.Entry(root)
        self.end_time_entry.insert(0, "15000")
        self.end_time_entry.pack()

        # CSV file label and button
        self.csv_label = tk.Label(root, text="Upload CSV for Batch Processing (Optional):")
        self.csv_label.pack()
        self.csv_button = tk.Button(root, text="Upload CSV", command=self.load_csv)
        self.csv_button.pack()

        # Clear Upload button
        self.clear_button = tk.Button(root, text="Clear Uploads", command=self.clear_uploads)
        self.clear_button.pack()

        # Process button
        self.process_button = tk.Button(root, text="Process", command=self.start_processing)
        self.process_button.pack()

        # Status label
        self.status_label = tk.Label(root, text="")
        self.status_label.pack()

        # Progress Bar
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack()

        # Terminal output block
        self.terminal_output = tk.Text(root, height=10, state='disabled', wrap='word')
        self.terminal_output.pack(fill=tk.BOTH, expand=True)

        # Redirect stdout and stderr to the Text widget
        sys.stdout = RedirectOutput(self.terminal_output, self.queue)
        sys.stderr = RedirectOutput(self.terminal_output, self.queue)

    def load_rosbag(self):
        self.rosbag_path = filedialog.askopenfilename(filetypes=[("Rosbag files", "*.bag")])
        if self.rosbag_path:
            self.status_label.config(text=f"Loaded Rosbag: {os.path.basename(self.rosbag_path)}")

    def load_csv(self):
        self.csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.csv_path:
            self.status_label.config(text=f"Loaded CSV: {os.path.basename(self.csv_path)}")
        if self.rosbag_path:
            self.status_label.config(text=f"Loaded Rosbag: {os.path.basename(self.rosbag_path)} and CSV: {os.path.basename(self.csv_path)}")

    def clear_uploads(self):
        self.rosbag_path = None
        self.csv_path = None
        self.status_label.config(text="Uploads cleared.")

    def start_processing(self):
        threading.Thread(target=self.process_rosbag).start()

    def process_rosbag(self):
        if not self.rosbag_path:
            messagebox.showerror("Error", "Please upload a rosbag file!")
            return

        start_time = int(self.start_time_entry.get() or "0")
        end_time = int(self.end_time_entry.get() or "15000")

        self.progress_bar['value'] = 0

        if self.csv_path:
            # Mode 1: CSV slicing mode
            print("Processing in CSV slicing mode...")
            self.process_batch_split(self.rosbag_path, self.csv_path)
        elif self.auto_slice_var.get():
            # Mode 2: Automatic slicing mode
            print("Processing in automatic slicing mode...")
            output_folder = self.process_single_rosbag(self.rosbag_path, start_time, end_time)
            self.automatic_slicing_policy(output_folder)
        else:
            # Mode 3: No input (default) - slice entire bag based on start/end time
            print("Processing entire bag with no external input...")
            self.process_single_rosbag(self.rosbag_path, start_time, end_time)

    def automatic_slicing_policy(self, output_folder):
        if output_folder:
            try:
                # Step 1: Generate full video
                self.status_label.config(text="Generating full video for analysis...")
                
                color_video_path = os.path.join(output_folder, 'Color.mp4')
                depth_video_path = os.path.join(output_folder, 'Depth.mp4')

                # Step 2: Analyze videos for split points
                self.status_label.config(text="Analyzing video for split points...")
                split_points = self.find_split_points(color_video_path, depth_video_path)

                # Step 3: Update CSV with identified splits
                self.status_label.config(text="Updating CSV with split points...")
                csv_path = os.path.join(output_folder, "auto_splits.csv")
                self.update_csv_with_splits(csv_path, split_points)
                
                # Step 4: Automatically run batch split using the generated CSV
                self.status_label.config(text="Running batch split with identified splits...")
                self.process_batch_split(self.rosbag_path, csv_path, h=0)

                # messagebox.showinfo("Automatic Slicing", f"Splits identified and batch process completed using {csv_path}.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def find_split_points(self, color_video_path, depth_video_path):
        cap_color = cv2.VideoCapture(color_video_path)
        cap_depth = cv2.VideoCapture(depth_video_path)
        red_found = False
        frame_idx = 0
        split_points = []
        previous_frames = []  # To store the previous 10 frames
        red_frame_time = 0
        green_frame_time = 0

        while cap_color.isOpened() and cap_depth.isOpened():
            ret_color, frame_color = cap_color.read()
            ret_depth, frame_depth = cap_depth.read()

            # Break the loop if no more frames are available
            if not ret_color and not ret_depth:

                print(f"End of video reached at frame index {frame_idx}")
                if red_found:
                    # Use the last valid frame as the green_frame_time (i.e., end)
                    green_frame_time = frame_idx - 1  # -1 because we're now past the last valid frame
                    split_points.append((red_frame_time, green_frame_time))
                    print(f"No green frame after red. Final split added: ({red_frame_time}, {green_frame_time})")
                    red_found = False
                break

            # Skip the frame if either color or depth frame is not readable
            if not ret_color or not ret_depth or frame_color is None or frame_depth is None:
                print(f"Warning: Skipping unreadable frame at index {frame_idx}")
                frame_idx += 1
                continue

            # Check the shape of frame_color to ensure it's valid
            if frame_color.shape[2] != 3:  # Ensure we have a color frame with 3 channels
                print(f"Error: Skipping frame with invalid color shape at index {frame_idx}")
                frame_idx += 1
                continue

            try:
                # Calculate average red, green, and depth values
                red_channel = int(np.mean(frame_color[:, :, 2]))  # Red channel
                green_channel = int(np.mean(frame_color[:, :, 1]))  # Green channel
                depth_channel = int(np.mean(frame_depth))  # Depth channel

                print(f"Frame {frame_idx}: red: {red_channel}, green: {green_channel}, depth: {depth_channel}")
                import matplotlib.pyplot as plt

                # Show color frame with overlay text
                overlay = frame_color.copy()
                cv2.putText(overlay, f"Frame: {frame_idx} R:{red_channel} G:{green_channel} D:{int(depth_channel)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("Color", overlay)
                cv2.imshow("Depth", frame_depth.astype(np.uint8))  # Scale if needed

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error processing frame at index {frame_idx}: {e}")
                frame_idx += 1
                continue

            # Store the last 10 frames to compare
            previous_frames.append((red_channel, green_channel, depth_channel))
            if len(previous_frames) > 5:
                previous_frames.pop()
            # Check for depth < 5 as the start marker (red found when depth < 5 and red is dominant)
            

            if depth_channel < 5:
                if not red_found and red_channel > green_channel:
                    red_frame_time = frame_idx  # Convert frame index to time in seconds
                    red_found = True

                    print(f"Red frame detected at frame {red_frame_time}.")
                    time.sleep(1)                
                elif red_found and green_channel > red_channel:
                    green_frame_time = frame_idx  # Convert frame index to time in seconds
                    time.sleep(1)
                    split_points.append((red_frame_time, green_frame_time))  # Add split point
                    red_found = False  # Reset for the next red-green pair
                    print(f"Green frame detected at frame {green_frame_time} seconds. Split added.")

                    
            frame_idx += 1
        cv2.destroyAllWindows()
        cap_color.release()
        cap_depth.release()

        # Return split points with only the timestamps (ignore 'red'/'green' tag)
        print(split_points)
        return split_points

    def update_csv_with_splits(self, csv_path, split_intervals):
        df = pd.DataFrame(split_intervals, columns=['start_seconds', 'end_seconds'])
        df.to_csv(csv_path, index=False)

    def process_single_rosbag(self, rosbag_path, start_time, end_time):
        output_folder = os.path.dirname(rosbag_path)
        
        if output_folder:
            try:
                self.progress_bar['maximum'] = 100
                self.status_label.config(text="Running rs-convert...")
                self.progress_bar['value'] = 30
                intr, recorded_fps, duration, total_frames = run_rs_convert(rosbag_path, output_folder, start_time, end_time)

                self.status_label.config(text="Converting images to videos...")
                self.progress_bar['value'] = 60
                frate = total_frames * 2 / duration
                run_ffmpeg_convert(output_folder, framerate = frate)

                self.status_label.config(text="Concatenating videos...")
                self.progress_bar['value'] = 90
                color_video_path = os.path.join(output_folder, 'Color.mp4') 
                depth_video_path = os.path.join(output_folder, 'Depth.mp4') 
                concat_videos(color_video_path, depth_video_path, os.path.join(output_folder, "output.mp4"))

                self.progress_bar['value'] = 100
                messagebox.showinfo("Success", f"Video processed and saved in {output_folder}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        return output_folder



    def copy_frame_range(self, color_dir, depth_dir, depth_color_dir, out_dir, start_frame, end_frame):
        os.makedirs(os.path.join(out_dir, "Color"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "Depth"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "Depth_Color"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "Raw"), exist_ok=True)

        start_frame = int(start_frame)
        end_frame = int(end_frame)

        for idx in range(start_frame, end_frame):
            color_src = os.path.join(color_dir, f"_Color_{idx:04d}.png")
            depth_src = os.path.join(depth_dir, f"_Depth_{idx:04d}.png")
            depth_color_src = os.path.join(depth_color_dir, f"_Depth_Color_{idx:04d}.png")

            # RAW files (adjust extensions/names as needed)
            color_raw_src = os.path.join(color_dir, f"_Color_{idx:04d}.raw")
            depth_raw_src = os.path.join(depth_dir, f"_Depth_{idx:04d}.raw")
            depth_color_raw_src = os.path.join(depth_color_dir, f"_Depth_Color_{idx:04d}.raw")

            # Destinations
            color_dst = os.path.join(out_dir, "Color", f"_Color_{idx:04d}.png")
            depth_dst = os.path.join(out_dir, "Depth", f"_Depth_{idx:04d}.png")
            depth_color_dst = os.path.join(out_dir, "Depth_Color", f"_Depth_Color_{idx:04d}.png")

            color_raw_dst = os.path.join(out_dir, "Color",  f"_Color_{idx:04d}.raw")
            depth_raw_dst = os.path.join(out_dir, "Depth",  f"_Depth_{idx:04d}.raw")
            depth_color_raw_dst = os.path.join(out_dir, "Depth_Color",f"_Depth_Color_{idx:04d}.raw")
            # Copy images
            if os.path.exists(color_src):
                shutil.copy2(color_src, color_dst)
            if os.path.exists(depth_src):
                shutil.copy2(depth_src, depth_dst)
            if os.path.exists(depth_color_src):
                shutil.copy2(depth_color_src, depth_color_dst)

            # Copy raw files
            if os.path.exists(color_raw_src):
                shutil.copy2(color_raw_src, color_raw_dst)
            if os.path.exists(depth_raw_src):
                shutil.copy2(depth_raw_src, depth_raw_dst)
            if os.path.exists(depth_raw_src):
                shutil.copy2(depth_color_raw_src, depth_color_raw_dst)



    def process_batch_split(self, rosbag_path, csv_path, h=0):
        output_folder = os.path.dirname(rosbag_path)
        
        if output_folder:
            try:
                df = pd.read_csv(csv_path, header=h)
                total_rows = len(df)
                self.progress_bar['maximum'] = total_rows

                # Step 1: Pre-extract all frames (full rosbag)
                print("Extracting full rosbag frames using rs-convert...")
                intr, recorded_fps, duration, total_frames = run_rs_convert(rosbag_path, output_folder, 0, 15000)  # Process from 0s to max
                frate = total_frames * 2 / duration
                color_dir = os.path.join(output_folder, 'Color/')
                depth_dir = os.path.join(output_folder, 'Depth/')
                depth_color_dir = os.path.join(output_folder, 'Depth_Color/')

                # Step 2: Loop through CSV, slice frames
                for i, row in df.iterrows():
                    start_frame = int(row['start_seconds'])  # assuming seconds == frames
                    end_frame = int(row['end_seconds'])
                    index = i + 1
                    out_path = os.path.join(output_folder, str(index))
                    os.makedirs(out_path, exist_ok=True)

                    self.copy_frame_range(color_dir, depth_dir, depth_color_dir, out_path, start_frame, end_frame)

                    run_ffmpeg_convert(out_path, framerate = frate)

                    color_video_path = os.path.join(out_path, "Color.mp4")
                    depth_video_path = os.path.join(out_path, "Depth.mp4")
                    concat_videos(color_video_path, depth_video_path, os.path.join(out_path, "output.mp4"))

                    self.progress_bar['value'] += 1

            except Exception as e:
                messagebox.showerror("Error", str(e))

    def update_terminal(self):
        """Update the terminal output with the last two messages from the queue."""
        try:
            messages = []  # List to store the last two messages
            while not self.queue.empty():
                message = self.queue.get_nowait()
                messages.append(message)
                if len(messages) > 5:  # Keep only the last two messages
                    messages.pop(0)
            
            if messages:
                self.terminal_output.config(state='normal')
                self.terminal_output.delete(1.0, tk.END)  # Clear previous messages
                self.terminal_output.insert(tk.END, "".join(messages))  # Display the last two messages
                self.terminal_output.see(tk.END)
                self.terminal_output.config(state='disabled')
        except Empty:
            pass
        self.root.after(100, self.update_terminal)  # Continuously check for new messages in the queue

# Main
if __name__ == "__main__":
    root = tk.Tk()
    app = RosbagSlicerGUI(root)
    root.mainloop()