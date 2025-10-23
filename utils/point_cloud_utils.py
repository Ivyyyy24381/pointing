"""Point cloud creation and visualization utilities"""

import numpy as np
import cv2
import open3d as o3d
from typing import Tuple, Optional


def create_point_cloud(color_img: np.ndarray, depth_img: np.ndarray,
                      fx: Optional[float] = None, fy: Optional[float] = None,
                      cx: Optional[float] = None, cy: Optional[float] = None) -> o3d.geometry.PointCloud:
    """
    Create point cloud from color and depth images.

    Args:
        color_img: Color image (H, W, 3) BGR
        depth_img: Depth image (H, W) in meters
        fx, fy: Focal lengths in pixels (auto-estimated if None)
        cx, cy: Principal point (auto-estimated if None)

    Returns:
        Point cloud
    """
    h, w = depth_img.shape

    # Auto-estimate intrinsics if not provided
    if fx is None or fy is None:
        # Common RealSense intrinsics
        if w == 640 and h == 480:
            fx = fy = 615.0
            cx = 320.0
            cy = 240.0
        elif w == 1280 and h == 720:
            fx = fy = 922.5
            cx = 640.0
            cy = 360.0
        elif w == 1920 and h == 1080:
            fx = fy = 1383.75
            cx = 960.0
            cy = 540.0
        else:
            # Generic assumption
            fx = fy = w * 0.96
            cx = w / 2
            cy = h / 2

    if cx is None:
        cx = w / 2
    if cy is None:
        cy = h / 2

    # Create Open3D images
    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image((depth_img * 1000).astype(np.uint16))  # Convert to mm

    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,  # mm to meters
        convert_rgb_to_intensity=False
    )

    # Camera intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=w,
        height=h,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # Flip it (Open3D uses different coordinate system)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    return pcd


def visualize_point_cloud(pcd: o3d.geometry.PointCloud, window_name: str = "Point Cloud",
                         width: int = 1024, height: int = 768) -> None:
    """
    Visualize point cloud in Open3D viewer.

    Args:
        pcd: Point cloud to visualize
        window_name: Window title
        width: Window width
        height: Window height
    """
    o3d.visualization.draw_geometries([pcd],
                                     window_name=window_name,
                                     width=width,
                                     height=height)


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: str) -> None:
    """
    Save point cloud to file.

    Args:
        pcd: Point cloud to save
        output_path: Output file path (.ply, .pcd, etc.)
    """
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"💾 Saved point cloud to: {output_path}")
