from tqdm import tqdm
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import yaml
import os 
import pandas as pd
import sys
sys.path.append('./')  
sys.path.append('gesture')# Adjust this path based on your project structure
from gesture_data_process import GestureDataProcessor
from batch_point_production import run_gesture_detection
class VideoTrimmerGUI:
    INTRINSICS_PATH = "config/camera_config.yaml"
    TARGETS_PATH = "config/targets.yaml"
    INTRINSICS = yaml.safe_load(open(INTRINSICS_PATH, 'r'))
    TARGETS = yaml.safe_load(open(TARGETS_PATH, 'r'))
    def process_selection(self):
        print(f"Processing video: {self.video_path}")
        print(f"Start frame: {self.start_frame}")
        print(f"End frame: {self.end_frame}")

        import shutil
        from tqdm import tqdm
        import os

        root_path = self.video_path.rsplit('/', 1)[0]
        color_video_path = os.path.join(root_path, 'Color.mp4')
        data_path = os.path.join(root_path, "gesture_data.csv")
        if not os.path.exists(data_path):
            # gesture_processor = PointingGestureDetector().run_video(color_video_path)
            base_path, subject_name, trial_no, _ = self.video_path.rsplit('/', 3)
            run_gesture_detection(base_path, subject_folder=subject_name, trial_id = trial_no)
            
        Gesture_data_processor = GestureDataProcessor(data_path)
        trimmed_data = Gesture_data_processor.trim_data(Gesture_data_processor, start_frame=self.start_frame, end_frame=self.end_frame)

        # Remove existing processed_gesture_data.csv before processing (if needed)
        processed_csv_path = os.path.join(root_path, "processed_gesture_data.csv")
        if os.path.exists(processed_csv_path):
            os.remove(processed_csv_path)

        Gesture_data_processor.process_data(trimmed_data)

        run_gesture_detection
        tqdm.write(f"✅ Processed trimmed data saved to {root_path}/processed_gesture_data.csv")

        # Reset video and proceed to next
        if hasattr(self, "video_batch") and hasattr(self, "batch_index"):
            self.cap.release()
            self.canvas.delete("all")
            self.batch_index += 1
            if self.batch_index < len(self.video_batch):
                self.load_video_path(self.video_batch[self.batch_index])
            else:
                tqdm.write("🎉 All videos processed.")
                self.root.quit()
        else:
            if self.progress:
                self.progress.update(1)
            self.next_video()
    def __init__(self, root):
        self.root = root
        self.root.title("Video Trimmer")

        self.video_path = None
        self.cap = None
        self.frame_pos = 0
        self.total_frames = 0
        self.start_frame = 0
        self.end_frame = 0
        self.video_queue = []
        self.progress = None

        self.canvas = tk.Canvas(root, width=640, height=480, highlightthickness=2)
        self.canvas.pack()

        self.load_btn = tk.Button(root, text="Load Video", command=self.load_video)
        self.load_btn.pack()

        self.frame_slider = tk.Scale(root, from_=0, to=0, orient=tk.HORIZONTAL, label="Frame", command=self.update_frame, length=800)
        self.frame_slider.pack()

        self.mark_start_btn = tk.Button(root, text="Mark Start", command=self.mark_start_frame)
        self.mark_start_btn.pack()

        self.mark_end_btn = tk.Button(root, text="Mark End", command=self.mark_end_frame)
        self.mark_end_btn.pack()
        self.process_btn = tk.Button(root, text="Process Selection", command=self.process_selection)
        self.process_btn.pack()

        self.root.bind("<Right>", self.next_frame)
        self.root.bind("<Left>", self.prev_frame)

    def set_video_queue(self, video_list):
        self.video_queue = video_list
        if self.video_queue:
            self.progress = tqdm(total=len(self.video_queue), desc="Processing Videos")
            self.load_video_path(self.video_queue.pop(0))

    def next_video(self):
        if self.video_queue:
            next_path = self.video_queue.pop(0)
            self.load_video_path(next_path)
        else:
            if self.progress:
                self.progress.close()
            self.root.quit()
    
    def load_video(self, video_path=None):
        if video_path:
            self.video_path = video_path
        else:
            self.video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if not self.video_path:
            return
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_pos = 0
        self.start_frame = 0
        self.end_frame = self.total_frames - 1
        self.frame_slider.config(to=self.total_frames - 1)
        self.frame_slider.set(0)
        self.show_frame()

    def show_frame(self):
        if self.cap:
            self.frame_pos = self.frame_slider.get()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                self.tk_image = self.cv_to_tk(frame)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
                self.root.title(f"Video Trimmer - Frame {self.frame_pos}/{self.total_frames}")
                if self.start_frame <= self.frame_pos <= self.end_frame:
                    self.canvas.config(highlightbackground="green")
                else:
                    self.canvas.config(highlightbackground="red")

    def cv_to_tk(self, frame):
        import PIL.Image, PIL.ImageTk
        img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return PIL.ImageTk.PhotoImage(image=img)

    def next_frame(self, event=None):
        if self.frame_pos < self.total_frames - 1:
            self.frame_pos += 1
            self.frame_slider.set(self.frame_pos)
            self.show_frame()

    def prev_frame(self, event=None):
        if self.frame_pos > 0:
            self.frame_pos -= 1
            self.frame_slider.set(self.frame_pos)
            self.show_frame()

    def update_frame(self, val):
        self.frame_pos = int(val)
        self.show_frame()

    def mark_start_frame(self):
        self.start_frame = self.frame_pos
        self.mark_start_btn.config(text=f"Mark Start (Frame {self.start_frame})")

    def mark_end_frame(self):
        self.end_frame = self.frame_pos
        self.mark_end_btn.config(text=f"Mark End (Frame {self.end_frame})")

    def load_video_path(self, path):
        """Call this externally to load a video from a given path."""
        self.load_video(video_path=path)
        
if __name__ == "__main__":
    import sys
    import glob
    import os
    from urllib.parse import unquote

    root = tk.Tk()
    app = VideoTrimmerGUI(root)

    if len(sys.argv) > 1:
        # Join all arguments into a single path string
        raw_input = " ".join(sys.argv[1:])
        clean_input = unquote(raw_input.strip('"'))

        if os.path.isdir(clean_input):
            video_files = sorted(glob.glob(os.path.join(clean_input, "*", "Color.mp4")))
            print(f"🎞️ Found {len(video_files)} videos.")
            # Set video_batch and batch_index, and load the first video
            app.video_batch = sorted(video_files)
            app.batch_index = 0
            app.load_video_path(app.video_batch[0])
        elif os.path.isfile(clean_input):
            app.load_video_path(clean_input)

    root.mainloop()
    # After GUI loop, concatenate all processed_gesture_data.csv files
    import pandas as pd

    if len(sys.argv) > 1:
        root_folder = unquote(" ".join(sys.argv[1:]).strip('"'))
        if os.path.isdir(root_folder):
            all_data = []
            for subdir, _, _ in os.walk(root_folder):
                csv_path = os.path.join(subdir, "processed_gesture_data.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        path_parts = os.path.normpath(subdir).split(os.sep)
                        if len(path_parts) >= 2:
                            df["dog"] = path_parts[-2]
                            df["trial"] = path_parts[-1]
                        else:
                            df["dog"] = "unknown"
                            df["trial"] = os.path.basename(subdir)
                        all_data.append(df)
                    except Exception as e:
                        print(f"❌ Failed to read {csv_path}: {e}")
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                # Reorder columns to put 'dog' and 'trial' first
                cols = combined_df.columns.tolist()
                if 'dog' in cols and 'trial' in cols:
                    cols.remove('dog')
                    cols.remove('trial')
                    combined_df = combined_df[['dog', 'trial'] + cols]
                
                # Sort by 'trial' column (converted to numeric if possible)
                try:
                    combined_df['trial_numeric'] = pd.to_numeric(combined_df['trial'], errors='coerce')
                    combined_df = combined_df.sort_values(by=['trial_numeric', 'frame']).drop(columns=['trial_numeric'])
                except Exception as e:
                    print(f"⚠️ Failed to sort by trial: {e}")
                save_path = os.path.join(root_folder, f"{path_parts[-2]}_gesture_data.csv")
                combined_df.to_csv(save_path, index=False)
                print(f"✅ Saved combined data to {save_path}")
            else:
                print("⚠️ No processed_gesture_data.csv files found.")
