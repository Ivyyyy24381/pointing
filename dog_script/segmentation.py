import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import sys
import json
sys.path.append('./thirdparty/sam2')
from sam2.build_sam import build_sam2_video_predictor
class SAM2VideoSegmenter:
    def clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM errors."""
        if hasattr(self, 'torch') and self.device.type == "cuda":
            self.torch.cuda.empty_cache()
            allocated = self.torch.cuda.memory_allocated(0) / 1024**3
            cached = self.torch.cuda.memory_reserved(0) / 1024**3
            print(f"🧹 GPU memory cleared: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    
    def __init__(self, video_dir):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # Set memory management environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from PIL import Image
        self.torch = torch
        self.plt = plt
        self.Image = Image
        self.video_dir = video_dir
        self.color_dir = os.path.split(self.video_dir)[0]
        self.max_size = 360

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # Clear GPU cache before initialization
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"using device: {self.device}")
        
        # Print GPU memory info if available
        if self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
            cached_memory = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU Memory: {allocated_memory:.2f}GB allocated, {cached_memory:.2f}GB cached, {total_memory:.2f}GB total")

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
        # Store scaling factors per frame
        self.scale_factors = {}
        self.original_dimensions = {}
        self.processing_dimensions = {}
        # Initialize inference state to None
        self.inference_state = None



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
        """Loads frames, converts to JPEG if needed, downsizes, and samples if too many frames."""
        from PIL import Image
        import shutil
        
        print(f"📁 Loading video frames from {self.video_dir}")
        
        png_files = [p for p in os.listdir(self.video_dir) if p.lower().endswith(".png")]
        jpg_files = [p for p in os.listdir(self.video_dir) if p.lower().endswith((".jpg", ".jpeg"))]
        self.tmp_jpeg_dir = None
        self.scale_factors = {}
        self.original_dimensions = {}
        self.processing_dimensions = {}

        print(f"🖼️ Found {len(png_files)} PNG files, {len(jpg_files)} JPEG files")

        # Convert PNG → downscaled JPEG
        if png_files and not jpg_files:
            print(f"🔄 Converting {len(png_files)} PNG files to resized JPEGs...")
            self.tmp_jpeg_dir = os.path.join(self.video_dir, "_tmp_jpegs")
            os.makedirs(self.tmp_jpeg_dir, exist_ok=True)
            for fname in sorted(png_files):
                src = os.path.join(self.video_dir, fname)
                img = Image.open(src).convert("RGB")
                orig_w, orig_h = img.size
                self.original_dimensions[fname] = (orig_w, orig_h)
                img.thumbnail((self.max_size, self.max_size))  # downscale
                new_w, new_h = img.size
                jpeg_name = fname.replace(".png", ".jpg")
                self.scale_factors[jpeg_name] = (orig_w / new_w, orig_h / new_h)
                self.processing_dimensions[jpeg_name] = (new_w, new_h)
                img.save(os.path.join(self.tmp_jpeg_dir, jpeg_name), "JPEG", quality=90)
            self.video_dir = self.tmp_jpeg_dir
            print(f"✅ Converted and resized frames, scale factors calculated")

        # Collect frame names (after conversion)
        all_frames = sorted([p for p in os.listdir(self.video_dir) if p.lower().endswith((".jpg", ".jpeg"))])
        print(f"🎥 Total frames available: {len(all_frames)}")

        # Implement more aggressive frame sampling for memory management
        # SAM2 loads all frames into GPU memory, so we need to be conservative
        max_frames_for_memory = 200  # Conservative limit for SAM2
        
        if len(all_frames) > max_frames_for_memory:
            # Calculate sampling rate to stay under limit
            sample_rate = max(2, len(all_frames) // max_frames_for_memory)
            print(f"⚠️ {len(all_frames)} frames found. Sampling every {sample_rate} frames for memory management.")
            sampled_frames = all_frames[::sample_rate]
            
            # Ensure we don't exceed the limit
            if len(sampled_frames) > max_frames_for_memory:
                sampled_frames = sampled_frames[:max_frames_for_memory]
                
            # Move sampled frames to temporary folder
            sampled_dir = os.path.join(self.video_dir, "_sampled")
            os.makedirs(sampled_dir, exist_ok=True)
            for fname in sampled_frames:
                shutil.copy(os.path.join(self.video_dir, fname), os.path.join(sampled_dir, fname))
            self.video_dir = sampled_dir
            self.frame_names = sampled_frames
            print(f"🎥 Using {len(sampled_frames)} sampled frames (reduced from {len(all_frames)})")
        elif len(all_frames) > 500:  # Still sample for performance even if memory allows
            sample_rate = 2
            print(f"⚠️ {len(all_frames)} frames found. Sampling every {sample_rate} frames for performance.")
            sampled_frames = all_frames[::sample_rate]
            
            sampled_dir = os.path.join(self.video_dir, "_sampled")
            os.makedirs(sampled_dir, exist_ok=True)
            for fname in sampled_frames:
                shutil.copy(os.path.join(self.video_dir, fname), os.path.join(sampled_dir, fname))
            self.video_dir = sampled_dir
            self.frame_names = sampled_frames
            print(f"🎥 Using {len(sampled_frames)} sampled frames")
        else:
            self.frame_names = all_frames
            print(f"🎥 Using all {len(all_frames)} frames")

        # Default scaling factors for already-resized JPEGs
        for f in self.frame_names:
            if f not in self.scale_factors:
                self.scale_factors[f] = (1.0, 1.0)
                # For existing JPEGs, assume they're already at processing size
                img = Image.open(os.path.join(self.video_dir, f))
                w, h = img.size
                self.processing_dimensions[f] = (w, h)
                self.original_dimensions[f] = (w, h)  # Assume same if no conversion
                
        print(f"📏 Scale factors for {len(self.scale_factors)} frames")
        
        # Debug: Show first few scale factors
        for i, (fname, scale) in enumerate(list(self.scale_factors.items())[:3]):
            print(f"   {fname}: scale={scale}, orig={self.original_dimensions.get(fname.replace('.jpg', '.png'), 'unknown')}, proc={self.processing_dimensions[fname]}")
        
        return len(self.frame_names)
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

    def rescale_mask(self, mask, frame_name):
        """Rescale a low-res mask back to original resolution."""
        sx, sy = self.scale_factors.get(frame_name, (1.0, 1.0))
        if sx != 1.0 or sy != 1.0:
            new_w = int(mask.shape[1] * sx)
            new_h = int(mask.shape[0] * sy)
            return cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        return mask
    

    def interactive_segmentation(self, frame_idx=None, obj_id=1):
        points_file = os.path.join(self.color_dir, "interactive_points.json")

        # Initialize inference state with memory management
        try:
            print(f"🔧 Initializing SAM2 inference state for {self.video_dir}")
            print(f"   Processing {len(self.frame_names)} frames")
            
            # Clear GPU cache before loading frames
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()
                
            self.inference_state = self.predictor.init_state(video_path=self.video_dir)
            self.predictor.reset_state(self.inference_state)
            
            # Check GPU memory after loading
            if self.device.type == "cuda":
                allocated_memory = self.torch.cuda.memory_allocated(0) / 1024**3
                cached_memory = self.torch.cuda.memory_reserved(0) / 1024**3
                print(f"   GPU Memory after loading: {allocated_memory:.2f}GB allocated, {cached_memory:.2f}GB cached")
                
            print(f"✅ SAM2 inference state initialized successfully")
        except self.torch.cuda.OutOfMemoryError as e:
            print(f"❌ GPU out of memory during SAM2 initialization: {e}")
            print(f"   Try reducing the number of frames or using a smaller model")
            print(f"   Current frame count: {len(self.frame_names)}")
            raise
        except Exception as e:
            print(f"❌ Failed to initialize SAM2 inference state: {e}")
            raise

        if os.path.exists(points_file):
            print(f"🔹 Found saved points at {points_file}, loading...")
            with open(points_file, 'r') as f:
                data = json.load(f)
            all_points = [tuple(p) for p in data["points"]]   # convert back to tuples
            all_labels = data["labels"]
            selected_frames_data = data.get("frames", [])     # load frame data
            
            # Convert frame names/indices to proper frame indices for current video
            frame_indices = []
            if selected_frames_data:
                # Check if we have frame names (strings) or indices (integers)
                if isinstance(selected_frames_data[0], str):
                    # Convert frame names to indices in current frame list
                    print(f"🔄 Converting {len(selected_frames_data)} frame names to indices...")
                    # Create a map from original frame names to current sampled indices
                    original_frames = sorted([f.replace('.jpg', '.png') for f in os.listdir(self.color_dir + '/Color') if f.endswith('.png')])
                    
                    for frame_name in set(selected_frames_data):  # Use unique frame names
                        if frame_name in original_frames:
                            original_idx = original_frames.index(frame_name)
                            # Find corresponding index in sampled frames
                            # Since we sampled every 3rd frame, calculate which sampled frame corresponds
                            sampled_idx = original_idx // 3  # Approximate mapping
                            if sampled_idx < len(self.frame_names):
                                frame_indices.append(sampled_idx)
                                print(f"   {frame_name} -> sampled index {sampled_idx}")
                            else:
                                print(f"   ⚠️ {frame_name} -> index {sampled_idx} out of range")
                else:
                    # Already have indices, just use them (with validation)
                    frame_indices = [idx for idx in selected_frames_data if isinstance(idx, int) and 0 <= idx < len(self.frame_names)]
            
            if not frame_indices:
                # Fallback: use a few representative frames
                frame_indices = [0, len(self.frame_names)//2, len(self.frame_names)-1]
                frame_indices = [idx for idx in frame_indices if idx < len(self.frame_names)]
                print(f"⚠️ Using fallback frame indices: {frame_indices}")

            # Scale points for SAM2 processing
            if self.scale_factors:
                # Use average scale factor
                scale_values = list(self.scale_factors.values())
                avg_sx = sum(s[0] for s in scale_values) / len(scale_values)
                avg_sy = sum(s[1] for s in scale_values) / len(scale_values)
                scaled_points = [(p[0] / avg_sx, p[1] / avg_sy) for p in all_points]
            else:
                scaled_points = all_points

            # Re-apply saved clicks to predictor
            for idx in frame_indices:
                try:
                    _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=idx,
                        obj_id=obj_id,
                        points=np.array(scaled_points, dtype=np.float32),
                        labels=np.array(all_labels, np.int32)
                    )
                    print(f"✅ Applied points to frame {idx}")
                except Exception as e:
                    print(f"❌ Failed to apply points to frame {idx}: {e}")
                    print(f"   Points shape: {np.array(scaled_points).shape}")
                    print(f"   Labels shape: {np.array(all_labels).shape}")
                    raise

        else:
            print("🖱️ No saved points found. Please click interactively...")

            selected_frames = np.linspace(0, len(self.frame_names) - 1, num=min(3, len(self.frame_names))).astype(int)
            all_points, all_labels = [], []

            for idx in selected_frames:
                frame_path = os.path.join(self.video_dir, self.frame_names[idx])
                points, labels = self.select_points_on_image(frame_path)
                all_points.extend(points)
                all_labels.extend(labels)

                # Scale points for SAM2 if needed
                frame_name = self.frame_names[idx]
                sx, sy = self.scale_factors.get(frame_name, (1.0, 1.0))
                scaled_points = [(p[0] / sx, p[1] / sy) for p in all_points]

                try:
                    _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=idx,
                        obj_id=obj_id,
                        points=np.array(scaled_points, dtype=np.float32),
                        labels=np.array(all_labels, np.int32)
                    )
                    print(f"✅ Applied points to frame {idx}")
                except Exception as e:
                    print(f"❌ Failed to apply points to frame {idx}: {e}")
                    print(f"   Points shape: {np.array(scaled_points).shape}")
                    print(f"   Labels shape: {np.array(all_labels).shape}")
                    raise

            # ✅ Save as JSON (convert tuples → lists for JSON compatibility)
            if all_points:  # Only save if we have points
                with open(points_file, 'w') as f:
                    json.dump({
                        "frames": selected_frames.tolist(),
                        "points": [list(p) for p in all_points],
                        "labels": all_labels
                    }, f, indent=2)

                print(f"✅ Saved interactive points to {points_file}")
            else:
                print("⚠️ No points were selected, segmentation may not work properly")

    def propagate_segmentation(self):
        if self.inference_state is None:
            print("❌ Cannot propagate segmentation: inference state not initialized")
            return
            
        print("🚀 Starting segmentation propagation...")
        
        # Clear GPU memory before propagation
        self.clear_gpu_memory()
        
        self.video_segments = {}
        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                self.video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                # Move tensors to CPU immediately to free GPU memory
                if hasattr(out_mask_logits, 'cpu'):
                    del out_mask_logits
                    
            print(f"✅ Propagation completed for {len(self.video_segments)} frames")
            
            # Clear GPU memory after propagation
            self.clear_gpu_memory()
            
        except self.torch.cuda.OutOfMemoryError as e:
            print(f"❌ GPU out of memory during propagation: {e}")
            print(f"   Consider reducing frame count further")
            self.clear_gpu_memory()
            raise
        except Exception as e:
            print(f"❌ Segmentation propagation failed: {e}")
            raise
    def save_scale_metadata(self):
        """Save scale factors and dimensions for downstream processing."""
        metadata = {
            "scale_factors": self.scale_factors,
            "original_dimensions": self.original_dimensions,
            "processing_dimensions": self.processing_dimensions,
            "max_size": self.max_size,
            "processing_info": {
                "resized_for_sam2": True,
                "frame_count": len(self.frame_names)
            }
        }
        
        metadata_path = os.path.join(self.color_dir, "sam2_scale_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved scale metadata to {metadata_path}")
        return metadata_path

    def visualize_results(self, vis_frame_stride=1):
        output_dir = os.path.join(self.color_dir, 'segmented_color')
        os.makedirs(output_dir, exist_ok=True)

        for idx in range(0, len(self.frame_names), vis_frame_stride):
            frame_name = self.frame_names[idx].split('.')[0] + '.png'
            img_path = os.path.join(self.color_dir,'Color', frame_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Skipping missing frame: {img_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            H, W = image_rgb.shape[:2]
            masked_image = np.zeros_like(image_rgb)

            if idx in self.video_segments:
                for obj_id, mask in self.video_segments[idx].items():
                    # Ensure mask is 2D boolean
                    mask_full = np.squeeze(mask)
                    if mask_full.ndim == 3:
                        mask_full = mask_full[:, :, 0]

                    # Use proper rescaling method
                    processing_frame_name = self.frame_names[idx]
                    mask_resized = self.rescale_mask(mask_full, processing_frame_name)
                    
                    # Ensure final mask matches original image dimensions
                    if mask_resized.shape != (H, W):
                        mask_resized = cv2.resize(mask_resized.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    mask_resized = mask_resized.astype(bool)

                    # Apply mask to masked_image
                    masked_image[mask_resized] = image_rgb[mask_resized]

            out_path = os.path.join(output_dir, f"vis_frame_{idx:06d}.png")
            plt.imsave(out_path, masked_image)

        # Save video
        img_paths = sorted(glob(os.path.join(output_dir, "vis_frame_*.png")))
        if img_paths:
            frame = cv2.imread(img_paths[0])
            h, w, _ = frame.shape
            out_video = os.path.join(os.path.split(output_dir)[0], "masked_video.mp4")
            writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
            for p in img_paths:
                writer.write(cv2.imread(p))
            writer.release()
            print(f"✅ Saved video to {out_video}")
            
        # Save scale metadata after processing
        self.save_scale_metadata()

if __name__ == "__main__":
    import shutil
    # 🔹 Ask user to provide the base directory interactively
    base_path = input("Enter the base folder path containing all trials: ").strip()

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"❌ The provided path does not exist: {base_path}")


    # Loop through each trial
    for trial in sorted(os.listdir(base_path)):
        trial_path = os.path.join(base_path, trial, "Color")
        if not os.path.isdir(trial_path):
            continue  # Skip non-trial directories

        print(f"\n=== 🐾 Processing Trial: {trial} ===")
        segmenter = SAM2VideoSegmenter(trial_path)
        segmenter.load_video_frames()
        segmenter.interactive_segmentation()   # 🖱️ interactive labeling

        # Optional: you can enable propagation + visualization if desired
        # segmenter.propagate_segmentation()
        # segmenter.visualize_results()

        # ✅ Clean up temp JPEG folder if created
        if hasattr(segmenter, "tmp_jpeg_dir") and segmenter.tmp_jpeg_dir is not None:
            try:
                shutil.rmtree(segmenter.tmp_jpeg_dir)
            except Exception as e:
                print(f"Warning: failed to remove temp JPEG dir {segmenter.tmp_jpeg_dir}: {e}")

    print("✅ All trials processed.")