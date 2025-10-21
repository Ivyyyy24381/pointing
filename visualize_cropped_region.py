#!/usr/bin/env python3
"""
Visualize what the dog detector sees (cropped region).
"""

import cv2
from pathlib import Path

# Load test image
image_path = "/Users/ivy/Downloads/dog_data/BDL049_Star_side_cam/1/Color/_Color_0125.png"
output_path = "/Users/ivy/Downloads/dog_data/BDL049_Star_side_cam/1/cropped_lower_half.png"

image = cv2.imread(image_path)
h, w = image.shape[:2]

# Crop to lower half (same as detector)
crop_ratio = 0.5
y_offset = int(h * (1 - crop_ratio))
cropped = image[y_offset:, :, :]

print(f"Original image: {w}x{h}")
print(f"Cropped region: {cropped.shape[1]}x{cropped.shape[0]}")
print(f"Y offset: {y_offset}")
print(f"Saving to: {output_path}")

cv2.imwrite(output_path, cropped)

print(f"\n✅ Saved cropped region!")
print(f"   This is what DeepLabCut is trying to detect a dog in.")
