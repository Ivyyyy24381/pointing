from detect_dog_skeleton import detect_dog
from dog_pose_visualize import pose_visualize
import os
import argparse
import matplotlib
from segmentation import SAM2VideoSegmenter
matplotlib.use('Agg')

def process_dog(folder_path, side_view=False):
    for folder_name in sorted(os.listdir(folder_path)):
        folder_full_path = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_full_path) or not folder_name.isdigit():
            continue

        video_path = os.path.join(folder_full_path, 'Color.mp4')
        segmenter = SAM2VideoSegmenter(os.path.join(folder_full_path, 'Color'))
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
        segmented_video_path = os.path.join(folder_full_path, 'masked_video.mp4')
        if os.path.exists(segmented_video_path):
            video_path = segmented_video_path
        detect_dog(video_path)
        

        json_files = [f for f in os.listdir(folder_full_path) if f.endswith('.json') and 'Color' in f]
        if not json_files:
            print(f"No JSON found in {folder_full_path}")
            continue

        json_path = os.path.join(folder_full_path, json_files[0])
        pose_visualize(json_path, side_view=side_view)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default='/home/xhe71/Desktop/dog_data/baby', help="Path to the root dog video dataset directory")
    parser.add_argument("--side_view", action='store_true', help="Flag to indicate if this is the side view")
    args = parser.parse_args()
    # process_dog(os.path.expanduser(args.root_path), side_view=args.side_view)
    process_dog(os.path.expanduser(args.root_path), side_view=True)