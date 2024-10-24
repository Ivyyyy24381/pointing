import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from gesture_detection import GestureDetector  # Assuming the class in your backend

class SkeletonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Skeleton Tracking GUI")
        self.detector = None
        self.video_path = None
        self.frame_data = []
        self.current_frame = 0

        # Input Mode Selection
        self.input_mode_var = tk.StringVar(value="File")  # Default to "File"
        tk.Label(root, text="Select Input Mode:").pack()
        tk.Radiobutton(root, text="File", variable=self.input_mode_var, value="File").pack(anchor=tk.W)
        tk.Radiobutton(root, text="Webcam", variable=self.input_mode_var, value="Webcam").pack(anchor=tk.W)
        tk.Radiobutton(root, text="Other Stream", variable=self.input_mode_var, value="Stream").pack(anchor=tk.W)

        # Button to upload/select video
        self.upload_button = tk.Button(root, text="Start Video", command=self.start_video)
        self.upload_button.pack()

        # Slider to select frames
        self.slider = tk.Scale(root, from_=0, to=0, orient=tk.HORIZONTAL, length=600, command=self.update_frame)
        self.slider.pack()

        # Matplotlib figure for 3D skeleton plot
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

    def start_video(self):
        mode = self.input_mode_var.get()

        # If File mode, ask for a file
        if mode == "File":
            self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
            if self.video_path:
                self.detector = GestureDetector(mode="file", input_source=self.video_path)
                self.process_video()

        # Webcam mode
        elif mode == "Webcam":
            self.detector = GestureDetector(mode="webcam")
            self.process_video()

        # Other stream (if applicable)
        elif mode == "Stream":
            # Here you can define another input source, for example, a network stream
            stream_url = filedialog.askstring("Stream URL", "Enter the stream URL:")
            if stream_url:
                self.detector = GestureDetector(mode="stream", input_source=stream_url)
                self.process_video()

    def process_video(self):
        if not self.detector:
            return

        # Assuming your detector has a method to process the video and return skeleton tracking data
        self.frame_data = self.detector.process()

        # Set the slider range based on the number of frames
        self.slider.config(to=len(self.frame_data) - 1)

        # Show the first frame
        self.update_frame(0)

    def update_frame(self, frame_idx):
        self.current_frame = int(frame_idx)
        self.ax.clear()

        # Get the 3D data for the selected frame
        skeleton = self.frame_data[self.current_frame]
        xs, ys, zs = skeleton[:, 0], skeleton[:, 1], skeleton[:, 2]

        # Plot the skeleton in 3D
        self.ax.scatter(xs, ys, zs, c='b')
        self.canvas.draw()

# Main
if __name__ == "__main__":
    root = tk.Tk()
    app = SkeletonGUI(root)
    root.mainloop()