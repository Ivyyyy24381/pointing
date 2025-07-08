import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import yaml
import os 
import pandas as pd
import sys
sys.path.append('./')  
sys.path.append('visualize')# Adjust this path based on your project structure
from gesture_data_process import GestureDataProcessor
class VideoTrimmerGUI:
    INTRINSICS_PATH = "config/camera_config.yaml"
    TARGETS_PATH = "config/targets.yaml"
    INTRINSICS = yaml.safe_load(open(INTRINSICS_PATH, 'r'))
    TARGETS = yaml.safe_load(open(TARGETS_PATH, 'r'))
    def process_selection(self):
        print(f"Processing video: {self.video_path}")
        print(f"Start frame: {self.start_frame}")
        print(f"End frame: {self.end_frame}")
        
        root_path = self.video_path.rsplit('/', 1)[0]
        data_path = os.path.join(root_path, "gesture_data.csv")
        Gesture_data_processor = GestureDataProcessor(data_path)
        trimmed_data = Gesture_data_processor.trim_data(Gesture_data_processor, start_frame=self.start_frame, end_frame=self.end_frame)  # Example frame range
        Gesture_data_processor.process_data(trimmed_data)
        messagebox.showinfo("Success", f"Processed trimmed data saved to {root_path}/processed_gesture_data.csv")
        
        
        
    def __init__(self, root):
        self.root = root
        self.root.title("Video Trimmer")

        self.video_path = None
        self.cap = None
        self.frame_pos = 0
        self.total_frames = 0
        self.start_frame = 0
        self.end_frame = 0

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

    def load_video(self):
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

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTrimmerGUI(root)
    root.mainloop()