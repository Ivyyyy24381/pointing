import deeplabcut
import cv2
import os

# === CONFIGURATION ===

def detect_dog(video_path):
    video_path = os.path.expanduser(video_path)  # Expands ~ to /home/username
    # === Load Pretrained DLC Zoo Model ===
    model_type = 'superanimal_quadruped' 
    output = deeplabcut.video_inference_superanimal([video_path],
                                            model_type,
                                            model_name="hrnet_w32",
                                            detector_name="fasterrcnn_resnet50_fpn_v2",
                                            video_adapt = False)    # === Analyze Video ===
    
# video_path = '~/Desktop/dog_data/BDL204_Waffle/2/Color.mp4'
# detect_dog(video_path)