"""
Image utilities for lower-half processing
"""
import numpy as np


def crop_to_lower_half(image, crop_ratio=0.6):
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


def crop_to_upper_half(image, crop_ratio=0.6):
    """
    Crop image to upper half for pointing detection (focuses on upper body).

    Args:
        image: Input image (H, W, C) or (H, W)
        crop_ratio: Ratio of image height to keep (0.6 = upper 60%)

    Returns:
        cropped_image: Upper portion of image
        y_offset: Y-offset for coordinate mapping (always 0 for upper crop)
    """
    height = image.shape[0]
    crop_height = int(height * crop_ratio)

    # Crop from top
    if len(image.shape) == 3:
        cropped = image[:crop_height, :, :]
    else:
        cropped = image[:crop_height, :]

    return cropped, 0  # y_offset is 0 since we crop from top


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
