import deeplabcut
import cv2
import os
import shutil
import glob
import mediapipe as mp
import json
import numpy as np

# === CONFIGURATION ===

def detect_dog(video_path):
    video_path = os.path.expanduser(video_path)  # Expands ~ to /home/username
    # === Load Pretrained DLC Zoo Model ===
    model_type = 'superanimal_quadruped' 
    output = deeplabcut.video_inference_superanimal([video_path],
                                            model_type,
                                            model_name="hrnet_w32",
                                            detector_name="fasterrcnn_resnet50_fpn_v2",
                                            max_individuals = 1, 
                                            plot_trajectories=True,
                                            batch_size = 4, 
                                            video_adapt = False)    # === Analyze Video ===
    # Define your desired new base name
    new_basename = os.path.splitext(os.path.basename(video_path))[0] + 'masked_video'

    # Get the DLC output directory
    output_dir = os.path.dirname(video_path)

    # Rename DLC output files
    for filetype in ['.mp4', '.json', '.h5']:
        matches = glob.glob(os.path.join(output_dir, f'*{filetype}'))
        for f in matches:
            if 'DeepCut' in f or 'skeleton' in f:  # DLC-named files
                new_name = os.path.join(output_dir, new_basename + filetype)
                shutil.move(f, new_name)
                print(f"Renamed {f} to {new_name}")
    return output


def run_mediapipe_json(video_path):
    video_path = os.path.expanduser(video_path)
    output_json_path = video_path.replace('.mp4', '_skeleton.json')
    output_video_path = video_path.replace('.mp4', '_annotated.mp4')

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)
    mp_drawing = mp.solutions.drawing_utils

    json_data = []
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        keypoints = []
        bbox = []
        if results.pose_landmarks:
            # Draw and extract landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            xs, ys = [], []
            for lm in results.pose_landmarks.landmark:
                x, y = int(lm.x * width), int(lm.y * height)
                xs.append(x)
                ys.append(y)
                keypoints.append([x, y, lm.visibility])

            x0, y0 = max(min(xs), 0), max(min(ys), 0)
            x1, y1 = min(max(xs), width - 1), min(max(ys), height - 1)
            bbox = [x0, y0, x1 - x0, y1 - y0]
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        else:
            # Fallback: compute bbox from non-black pixels
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(thresh)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                bbox = [x, y, w, h]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        out.write(frame)

        json_data.append({
            "bodyparts": [keypoints] if keypoints else [],
            "bboxes": [bbox] if bbox else [],
            "bbox_scores": [1.0] if bbox else []
        })

        frame_idx += 1

    cap.release()
    out.release()
    pose.close()

    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved JSON to {output_json_path}")
    print(f"Saved annotated video to {output_video_path}")
# video_path = '~/Desktop/dog_data/BDL204_Waffle/2/Color.mp4'
# detect_dog(video_path)