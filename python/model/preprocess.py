import cv2
import numpy as np
import torch

from typing import Tuple


# Fixed for this project — must match the checkpoint's training normalization.
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SIZE_DIVISOR = 32


def load_image_rgb(image_path: str) -> np.ndarray:
    """Loads an image from disk as RGB, (H, W, 3) uint8."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image(img: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    Matches the original repo's raw-inference pipeline exactly:
    normalize -> pad to a multiple of 32 (bottom/right only) -> to tensor -> add batch dim.

    No resizing happens here, so the calibration matrix (P2) needs NO
    adjustment — padding only extends the canvas, it never moves the
    image origin, so the principal point stays valid as-is.

    Args:
        img: (H, W, 3) uint8 RGB image, as loaded by `load_image_rgb`.

    Returns:
        input_tensor: (1, 3, padded_H, padded_W) float32, normalized, ready for the model.
        ori_img:      (H, W, 3) float32, UNNORMALIZED — original resolution, for visualization.
        pad_shape:    (padded_H, padded_W) — needed by decode.py for the feature-map -> pixel scale factor.
    """
    ori_img = img.astype(np.float32).copy()

    # Normalize
    norm_img = (img.astype(np.float32) - MEAN.reshape(1, 1, -1)) / STD.reshape(1, 1, -1)

    # Pad to a multiple of 32, anchored top-left (zeros added bottom/right only)
    ori_h, ori_w = norm_img.shape[:2]
    padded_h = int(np.ceil(ori_h / SIZE_DIVISOR)) * SIZE_DIVISOR
    padded_w = int(np.ceil(ori_w / SIZE_DIVISOR)) * SIZE_DIVISOR

    canvas = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    canvas[:ori_h, :ori_w, :] = norm_img

    # To tensor, CHW, add batch dim
    input_tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0)

    return input_tensor, ori_img, (padded_h, padded_w)