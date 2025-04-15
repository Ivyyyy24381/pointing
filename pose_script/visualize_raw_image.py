import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from matplotlib.widgets import RectangleSelector

import pyrealsense2 as rs

# RealSense intrinsics from bag file
cfg = rs.config()
cfg.enable_device_from_file("/home/xhe71/Desktop/dog_data/BDL204_Waffle/BDL204_Waffles_PVP_01.bag")
pipeline = rs.pipeline()
profile = pipeline.start(cfg)
frames = pipeline.wait_for_frames()
depth_stream = profile.get_stream(rs.stream.depth)
intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
pipeline.stop()

fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy

# Load color and depth raw images
color_path = '/home/xhe71/Desktop/dog_data/BDL204_Waffle/Color/_Color_1401.raw'
depth_path = '/home/xhe71/Desktop/dog_data/BDL204_Waffle/Depth/_Depth_1401.raw'
width = 1280
height = 720

with open(color_path, 'rb') as f:
    color_raw = np.fromfile(f, dtype=np.uint8).reshape((height, width, 3))

with open(depth_path, 'rb') as f:
    depth_raw = np.fromfile(f, dtype=np.uint16).reshape((height, width))

depth_meters = depth_raw.astype(np.float32) / 1000.0
depth_meters[depth_raw == 0] = np.nan  # ignore invalid depth

# Global for rectangle
bbox = []

# Click and # Prepare the figure with two subplots
fig, (ax_color, ax_depth) = plt.subplots(1, 2, figsize=(14, 6))

# Show color and depth grayscale initially
depth_display = ax_depth.imshow(depth_meters, cmap='gray')
color_display = ax_color.imshow(color_raw)

# Add rectangle patches (invisible until drawn)
rect_color = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='lime', facecolor='none')
rect_depth = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
ax_color.add_patch(rect_color)
ax_depth.add_patch(rect_depth)

# Title
ax_color.set_title("Color Image")
ax_depth.set_title("Depth Image (meters)")
for ax in (ax_color, ax_depth):
    ax.axis('off')

# Modified onselect to update rectangles
def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    # Update rectangle visuals
    width = xmax - xmin
    height = ymax - ymin
    rect_color.set_bounds(xmin, ymin, width, height)
    rect_depth.set_bounds(xmin, ymin, width, height)

    # Crop depth and color
    depth_crop = depth_meters[ymin:ymax, xmin:xmax]
    color_crop = color_raw[ymin:ymax, xmin:xmax]

    # Print average RGB
    mean_rgb = np.mean(color_crop, axis=(0, 1))[::-1]  # BGR → RGB
    print(f"Avg color (RGB): R={mean_rgb[0]:.1f}, G={mean_rgb[1]:.1f}, B={mean_rgb[2]:.1f}")

    # Compute average depth
    valid_depths = depth_crop[~np.isnan(depth_crop)]
    if valid_depths.size == 0:
        print("No valid depth values found.")
        return

    avg_depth = np.mean(valid_depths)
    print(f"Average depth (m): {avg_depth:.3f}")

    # Print center pixel info
    center_x = (xmax + xmin) // 2
    center_y = (ymax + ymin) // 2
    center_rgb = color_raw[center_y, center_x][::-1]
    center_depth = depth_meters[center_y, center_x]
    print(f"Center pixel at ({center_x}, {center_y}) → RGB={center_rgb}, Depth={center_depth:.3f} m")

    # Convert pixel region to 3D points
    ys, xs = np.mgrid[ymin:ymax, xmin:xmax]
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    z = depth_crop

    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    Z = z

    valid = ~np.isnan(Z)
    X, Y, Z = X[valid], Y[valid], Z[valid]

    if Z.size == 0:
        print("No valid 3D points.")
        return

    center_3D = np.mean(np.stack([X, Y, Z], axis=1), axis=0)
    print(f"Avg 3D position (m): X={center_3D[0]:.3f}, Y={center_3D[1]:.3f}, Z={center_3D[2]:.3f}")

    fig.canvas.draw_idle()


# RectangleSelector setup
toggle_selector = RectangleSelector(
    ax_color,
    onselect,
    useblit=True,
    button=[1],
    minspanx=5,
    minspany=5,
    spancoords='pixels',
    interactive=True
)

plt.suptitle("Draw box on color image to get matching depth + 3D position")
plt.tight_layout()
plt.show()
