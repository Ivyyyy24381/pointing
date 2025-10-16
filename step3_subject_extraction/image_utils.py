"""
Image utilities for lower-half processing
"""
import numpy as np


def crop_to_lower_half(image, crop_ratio=0.5):
    """
    Crop image to lower half for baby detection (avoids detecting adults).

    Args:
        image: Input image (H, W, C)
        crop_ratio: Ratio of image height to keep (0.5 = lower 50%)

    Returns:
        cropped_image: Lower portion of image
        y_offset: Y-offset for coordinate mapping
    """
    height = image.shape[0]
    y_offset = int(height * (1 - crop_ratio))

    cropped = image[y_offset:, :, :]
    return cropped, y_offset


def map_coordinates_from_crop(keypoints, y_offset):
    """
    Map keypoint coordinates from cropped image back to original image.

    Args:
        keypoints: List of keypoints [[x, y, conf], ...]
        y_offset: Y-offset from cropping

    Returns:
        mapped_keypoints: Keypoints in original image coordinates
    """
    mapped = []
    for kp in keypoints:
        if len(kp) >= 2:
            x, y = kp[0], kp[1]
            rest = kp[2:] if len(kp) > 2 else []
            # Add y_offset back to y coordinate
            mapped.append([x, y + y_offset] + rest)
        else:
            mapped.append(kp)
    return mapped


def map_bbox_from_crop(bbox, y_offset):
    """
    Map bounding box from cropped image back to original image.

    Args:
        bbox: [x, y, w, h] in cropped coordinates
        y_offset: Y-offset from cropping

    Returns:
        mapped_bbox: [x, y, w, h] in original coordinates
    """
    if len(bbox) != 4:
        return bbox

    x, y, w, h = bbox
    return [x, y + y_offset, w, h]
