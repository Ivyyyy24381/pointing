from GUI.bag_slice_gui import RosbagSlicerGUI
import tkinter as tk
root = tk.Tk()
app = RosbagSlicerGUI(root)
color_video_path = "/home/xhe71/Desktop/dog_data/BDL204_Waffle/Color.mp4"
depth_video_path = "/home/xhe71/Desktop/dog_data/BDL204_Waffle/Depth.mp4"
split_points = app.find_split_points(color_video_path, depth_video_path)


# Step 3: Update CSV with identified splits
self.status_label.config(text="Updating CSV with split points...")
csv_path = os.path.join(output_folder, "auto_splits.csv")
self.update_csv_with_splits(csv_path, split_points)

# Step 4: Automatically run batch split using the generated CSV
self.status_label.config(text="Running batch split with identified splits...")
self.process_batch_split(self.rosbag_path, csv_path, h=0)
