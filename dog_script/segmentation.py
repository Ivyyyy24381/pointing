import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import sys
sys.path.append('./thirdparty/sam2')
from sam2.build_sam import build_sam2_video_predictor
class SAM2VideoSegmenter:
    def __init__(self, video_dir):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from PIL import Image
        self.torch = torch
        self.plt = plt
        self.Image = Image
        self.video_dir = video_dir

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"using device: {self.device}")

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        
        sam2_checkpoint = os.path.join(os.getcwd(),"thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)


    @staticmethod
    def show_mask(mask, ax, obj_id=None, random_color=False):
        import numpy as np
        import matplotlib.pyplot as plt
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        import matplotlib.pyplot as plt
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


    def load_video_frames(self):
        # Support for .png input folders: convert PNGs to JPEGs in a temp folder if needed
        import shutil
        png_files = [p for p in os.listdir(self.video_dir) if os.path.splitext(p)[-1].lower() == ".png"]
        jpg_files = [p for p in os.listdir(self.video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
        self.tmp_jpeg_dir = None
        if png_files and not jpg_files:
            # Create temp JPEG folder
            import tempfile
            self.tmp_jpeg_dir = os.path.join(self.video_dir, "_tmp_jpegs")
            os.makedirs(self.tmp_jpeg_dir, exist_ok=True)
            for png_file in png_files:
                src_path = os.path.join(self.video_dir, png_file)
                dst_name = os.path.splitext(png_file)[0] + ".jpg"
                dst_path = os.path.join(self.tmp_jpeg_dir, dst_name)
                img = self.Image.open(src_path).convert("RGB")
                img.save(dst_path, "JPEG")
            self.video_dir = self.tmp_jpeg_dir
        self.frame_names = sorted([
            p for p in os.listdir(self.video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ], key=lambda p: str(os.path.splitext(p)[0]))

    def select_points_on_image(self, frame_path):
        import matplotlib.pyplot as plt
        from PIL import Image
        image = Image.open(frame_path)
        fig, ax = plt.subplots()
        ax.imshow(image)
        coords = []
        labels = []

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                label = 0 if event.button == 3 else 1  # Right-click = background (label 0), left-click = foreground (label 1)
                coords.append((event.xdata, event.ydata))
                labels.append(label)
                color = 'ro' if label == 0 else 'go'
                ax.plot(event.xdata, event.ydata, color)
                fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.title(f"Click to select points on {os.path.basename(frame_path)} (close window when done)")
        plt.show()
        return coords, labels

    def interactive_segmentation(self, frame_idx=None, obj_id=1):
        self.inference_state = self.predictor.init_state(video_path=self.video_dir)
        self.predictor.reset_state(self.inference_state)
        selected_frames = np.linspace(0, len(self.frame_names)-1, num = min(3, len(self.frame_names)))
        all_points = []
        all_labels = []
        for idx in selected_frames:
            idx = int(idx)
            frame_path = os.path.join(self.video_dir, self.frame_names[idx])
            points, labels = self.select_points_on_image(frame_path)
            all_points.extend(points)
            all_labels.extend(labels)

            _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=idx,
                obj_id=obj_id,
                points=np.array(all_points, dtype=np.float32),
                labels=np.array(all_labels, np.int32)
            )

    def propagate_segmentation(self):
        self.video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def visualize_results(self, vis_frame_stride=1):
        # split and get the last folder
        root_dir = os.path.join(os.path.split(os.path.split(self.video_dir)[0])[0])
        output_dir = os.path.join(os.path.split(os.path.split(self.video_dir)[0])[0], 'segmented_color')
        os.makedirs(output_dir, exist_ok=True)
        plt.close("all")
        for out_frame_idx in range(0, len(self.frame_names), vis_frame_stride):
            img_path = os.path.join(self.video_dir, self.frame_names[out_frame_idx])
            image = cv2.imread(img_path)  # BGR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create black canvas
            masked_image = np.zeros_like(image_rgb)

            # Apply mask(s)
            if out_frame_idx in self.video_segments:
                for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
                    out_mask = out_mask.squeeze().astype(bool)
                    
                    masked_image[out_mask] = image_rgb[out_mask]

            # Save result
            fig_path = os.path.join(output_dir, f"vis_frame_{out_frame_idx:06d}.png")
            plt.imsave(fig_path, masked_image)
        
        
        # --- Save video ---
        img_paths = sorted(glob(os.path.join(output_dir, "vis_frame_*.png")))
        if img_paths:
            # Read first image to get frame size
            frame = cv2.imread(img_paths[0])
            height, width, _ = frame.shape
            video_out_dir = os.path.split(output_dir)[0]
            video_path = os.path.join(video_out_dir, "masked_video.mp4")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

            for img_path in img_paths:
                frame = cv2.imread(img_path)
                writer.write(frame)
            writer.release()
            print(f"✅ Saved video to {video_path}")
        else:
            print("⚠️ No masked frames found to save as video.")


if __name__ == "__main__":
    segmenter = SAM2VideoSegmenter("/home/xhe71/Desktop/dog_data/baby/CCD0411_side/1/Color")
    segmenter.load_video_frames()
    segmenter.interactive_segmentation()
    segmenter.propagate_segmentation()
    segmenter.visualize_results()
    # Clean up temp JPEG folder if it was created
    if hasattr(segmenter, "tmp_jpeg_dir") and segmenter.tmp_jpeg_dir is not None:
        import shutil
        try:
            shutil.rmtree(segmenter.tmp_jpeg_dir)
        except Exception as e:
            print(f"Warning: failed to remove temp JPEG dir {segmenter.tmp_jpeg_dir}: {e}")
    if hasattr(segmenter, "segmented_path"):
        import shutil
        try:
            shutil.rmtree(segmenter.tmp_jpeg_dir)
        except Exception as e:
            print(f"Warning: failed to remove temp JPEG dir {segmenter.tmp_jpeg_dir}: {e}")