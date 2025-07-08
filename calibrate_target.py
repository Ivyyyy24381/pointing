color_path = "/Users/ivy/Library/CloudStorage/GoogleDrive-xiao_he@brown.edu/Shared drives/pointing_production/BDL202_Dee-Dee_front/2/Color/_Color_1032.png"
depth_path = "/Users/ivy/Library/CloudStorage/GoogleDrive-xiao_he@brown.edu/Shared drives/pointing_production/BDL202_Dee-Dee_front/2/Depth_Color/_Depth_Color_1032.raw"
outout_path = "config/targets.yaml"
import cv2
import numpy as np
import matplotlib.pyplot as plt


# display color and depth images
# Read color image
color_img = cv2.imread(color_path)
# Read .raw depth file (16-bit unsigned, known resolution)
depth_width, depth_height = color_img.shape[1], color_img.shape[0]

camera_intrinsics = {
        "fx": 614.52099609375,  # Focal length in x (pixels)
        "fy": 614.460693359375,  # Focal length in y (pixels)
        "cx": depth_width/2,     # Principal point x (pixels) (image width / 2 for 640x480)
        "cy": depth_height/2      # Principal point y (pixels) (image height / 2 for 640x480)
    }

import numpy as np
with open(depth_path, 'rb') as f:
    depth_img = np.frombuffer(f.read(), dtype=np.uint16).reshape((depth_height, depth_width))
# Convert depth image to meters if it's in millimeters
if depth_img.dtype == np.uint16:
    depth_img = depth_img.astype(np.float32) / 1000.0  # Convert mm to m

# Create a visual copy of the depth image to draw on without modifying the actual data
depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
depth_vis = np.uint8(depth_vis)
depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)  # For color drawing

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
plt.title("Color Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(depth_img)
plt.title("Depth Image")
plt.axis('off')

# Select ROIs on the color image
rois = cv2.selectROIs("Select ROIs", color_img, showCrosshair=True)
cv2.destroyAllWindows()

# Draw selected boxes and extract center points and corresponding depths
for i, roi in enumerate(rois):
    x, y, w, h = roi
    center_x = x + w // 2
    center_y = y + h // 2
    depth_value = depth_img[center_y, center_x]

    # Draw rectangle and center on color image
    cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(color_img, (center_x, center_y), 5, (0, 0, 255), -1)

    # Draw corresponding box and center on depth_vis (visual copy)
    cv2.rectangle(depth_vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.circle(depth_vis, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.putText(depth_vis, f"Box {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    print(f"Box {i+1}: Center = ({center_x}, {center_y}), Depth = {depth_value:.3f} meters")

import yaml

targets = []
for i, roi in enumerate(rois):
    x, y, w, h = roi
    center_x = x + w // 2
    center_y = y + h // 2
    depth_value = depth_img[center_y, center_x]
    # Convert pixel coordinates to meters using camera intrinsics
    cx, cy = camera_intrinsics["cx"], camera_intrinsics["cy"]
    fx, fy = camera_intrinsics["fx"], camera_intrinsics["fy"]
    X = (center_x - cx) * depth_value / fx
    Y = (center_y - cy) * depth_value / fy
    Z = depth_value
    targets.append({
        "id": f"target_{i+1}",
        "center": [int(center_x), int(center_y)],
        "depth_m": float(depth_value),
        "position_m": [float(X), float(Y), float(Z)]
    })

# Save to YAML
with open(outout_path, 'w') as f:
    yaml.dump({"targets": targets}, f, sort_keys=False)
print(f"Target data saved to {outout_path}")
