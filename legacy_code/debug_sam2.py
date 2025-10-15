#!/usr/bin/env python3
"""
Debug script for SAM2 tensor indexing issues
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

# Add SAM2 to path
sys.path.append('./thirdparty/sam2')
from sam2.build_sam import build_sam2_video_predictor


def debug_sam2_initialization(video_dir):
    """Debug SAM2 initialization with detailed logging."""
    print(f"🔍 Debugging SAM2 initialization for: {video_dir}")
    
    # Check directory structure
    if not os.path.exists(video_dir):
        print(f"❌ Video directory does not exist: {video_dir}")
        return False
        
    # List files in directory
    files = os.listdir(video_dir)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📁 Directory contents: {len(files)} total files, {len(image_files)} image files")
    
    if not image_files:
        print("❌ No image files found in directory")
        return False
    
    # Check first few images
    for i, img_file in enumerate(sorted(image_files)[:3]):
        img_path = os.path.join(video_dir, img_file)
        try:
            img = Image.open(img_path)
            print(f"🖼️ {img_file}: {img.size} ({img.mode})")
        except Exception as e:
            print(f"❌ Failed to open {img_file}: {e}")
            return False
    
    # Initialize SAM2
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {device}")
        
        sam2_checkpoint = os.path.join(os.getcwd(), "thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        if not os.path.exists(sam2_checkpoint):
            print(f"❌ SAM2 checkpoint not found: {sam2_checkpoint}")
            return False
            
        print("🚀 Building SAM2 predictor...")
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        print("✅ SAM2 predictor built successfully")
        
        # Try to initialize inference state
        print("🔄 Initializing inference state...")
        inference_state = predictor.init_state(video_path=video_dir)
        print("✅ Inference state initialized successfully")
        
        # Check inference state structure
        print("🔍 Inference state keys:", list(inference_state.keys()))
        
        if "images" in inference_state:
            images = inference_state["images"]
            print(f"📊 Images tensor info:")
            print(f"   Type: {type(images)}")
            if hasattr(images, 'shape'):
                print(f"   Shape: {images.shape}")
            if hasattr(images, 'dtype'):
                print(f"   Dtype: {images.dtype}")
                
            # Try to access first few frames
            try:
                for i in range(min(3, len(image_files))):
                    if hasattr(images, '__getitem__'):
                        frame = images[i]
                        print(f"   Frame {i} shape: {frame.shape if hasattr(frame, 'shape') else 'No shape'}")
                    else:
                        print(f"   Images object not indexable")
                        break
            except Exception as e:
                print(f"❌ Error accessing frame data: {e}")
                print(f"   Images type: {type(images)}")
                if hasattr(images, 'keys'):
                    print(f"   Images keys: {list(images.keys())}")
                return False
        
        # Try to add a simple point
        print("🎯 Testing point addition...")
        try:
            test_points = np.array([[100.0, 100.0]], dtype=np.float32)
            test_labels = np.array([1], dtype=np.int32)
            
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=test_points,
                labels=test_labels
            )
            print("✅ Point addition successful")
            print(f"   Output obj IDs: {out_obj_ids}")
            print(f"   Output mask shape: {out_mask_logits.shape if hasattr(out_mask_logits, 'shape') else 'No shape'}")
            
        except Exception as e:
            print(f"❌ Point addition failed: {e}")
            print(f"   Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        print("🎉 All SAM2 tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ SAM2 initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug SAM2 tensor indexing issues")
    parser.add_argument("--video_dir", type=str, required=True, help="Video directory to test")
    
    args = parser.parse_args()
    
    video_dir = os.path.expanduser(args.video_dir)
    success = debug_sam2_initialization(video_dir)
    
    if success:
        print("\n✅ SAM2 debugging completed successfully")
    else:
        print("\n❌ SAM2 debugging found issues")
        
    return success


if __name__ == "__main__":
    main()