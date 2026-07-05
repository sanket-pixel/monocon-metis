import numpy as np
import torch
import torch.nn.functional as F

from typing import Dict, List, Tuple


PI = np.pi

# Fixed constants for this project — MonoCon on KITTI, DLA-34 backbone.
TOPK = 30                    # Max objects considered per image
LOCAL_MAXIMUM_KERNEL = 3     # NMS-via-maxpool kernel size on the heatmap
NUM_ALPHA_BINS = 12          # Must match head.py
NUM_KPTS = 9                 # Must match head.py


def get_local_maximum(heatmap: torch.Tensor, kernel: int = LOCAL_MAXIMUM_KERNEL) -> torch.Tensor:
    """
    Cheap NMS: a pixel survives only if it's the max within its own
    `kernel`x`kernel` neighborhood. Suppresses everything else to zero.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    is_peak = (hmax == heatmap).float()
    return heatmap * is_peak


def get_topk_from_heatmap(heatmap: torch.Tensor, k: int = TOPK) -> Tuple[torch.Tensor, ...]:
    """
    Flatten (class, y, x) into one dimension and take the top-k scoring
    pixels overall, then recover which class/row/col each one came from.
    """
    batch, _, height, width = heatmap.shape
    scores, flat_indices = torch.topk(heatmap.view(batch, -1), k)

    classes = torch.div(flat_indices, height * width, rounding_mode='floor')
    pixel_indices = flat_indices % (height * width)
    ys = torch.div(pixel_indices, width, rounding_mode='floor')
    xs = (pixel_indices % width).float()

    return scores, pixel_indices, classes, ys, xs


def gather_at_indices(feat: torch.Tensor, pixel_indices: torch.Tensor) -> torch.Tensor:
    """
    Pull out the feature vector at each of the top-k pixel locations.
    feat: (B, C, H, W) -> (B, K, C), one row per selected object.
    """
    B, C, H, W = feat.shape
    feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, H*W, C)
    idx = pixel_indices.unsqueeze(2).expand(-1, -1, C)      # (B, K, C)
    return feat.gather(1, idx)


def decode_alpha(alpha_cls: torch.Tensor, alpha_offset: torch.Tensor) -> torch.Tensor:
    """
    alpha_cls:    (B, K, NUM_ALPHA_BINS) — which angle bin is most likely
    alpha_offset: (B, K, NUM_ALPHA_BINS) — offset within each bin

    Picks the winning bin, takes that bin's offset, converts
    (bin index + offset) into a single angle in radians, wrapped to [-pi, pi].
    """
    best_bin = alpha_cls.argmax(dim=-1, keepdim=True)                 # (B, K, 1)
    offset = alpha_offset.gather(2, best_bin)                          # (B, K, 1)

    angle_per_bin = (2 * PI) / NUM_ALPHA_BINS
    alpha = (best_bin.float() * angle_per_bin) + offset

    alpha = torch.where(alpha > PI, alpha - 2 * PI, alpha)
    alpha = torch.where(alpha < -PI, alpha + 2 * PI, alpha)
    return alpha


def alpha_to_rotation_y(image_x: torch.Tensor, alpha: torch.Tensor, calib_P2: torch.Tensor) -> torch.Tensor:
    """
    Converts observation angle (alpha, camera-relative) into global
    rotation-y, using the object's image-plane x-position and the
    camera's horizontal focal length / principal point (from P2).

    image_x: (B, K, 1) — projected 3D center x-coordinate, in pixels
    alpha:   (B, K, 1)
    calib_P2: (B, 3, 4)
    """
    focal_x = calib_P2[:, 0:1, 0:1]       # (B, 1, 1)
    principal_x = calib_P2[:, 0:1, 2:3]   # (B, 1, 1)

    rot_y = alpha + torch.atan2(image_x - principal_x, focal_x)

    rot_y = torch.where(rot_y > PI, rot_y - 2 * PI, rot_y)
    rot_y = torch.where(rot_y < -PI, rot_y + 2 * PI, rot_y)
    return rot_y


def image_to_camera_3d(image_xy: torch.Tensor, depth: torch.Tensor, calib_P2_list: List[np.ndarray]) -> torch.Tensor:
    """
    Un-projects a 2D image point + depth back into 3D camera coordinates,
    using the inverse of each image's calibration matrix.

    image_xy: (B, K, 2) pixel coordinates
    depth:    (B, K, 1)
    Returns:  (B, K, 3) camera-space (x, y, z)
    """
    # Homogeneous image-plane point scaled by depth: (u*z, v*z, z)
    scaled_xy = image_xy * depth
    points_2d_scaled = torch.cat([scaled_xy, depth], dim=-1)   # (B, K, 3)

    results = []
    for b_idx, points in enumerate(points_2d_scaled):
        P2 = points.new_tensor(calib_P2_list[b_idx])            # (3, 4)

        viewpad = torch.eye(4, device=points.device)
        viewpad[:3, :4] = P2
        inv_viewpad = torch.inverse(viewpad).T

        homo = torch.cat([points, points.new_ones((points.shape[0], 1))], dim=1)  # (K, 4)
        points_3d = (homo @ inv_viewpad)[:, :3]                 # (K, 3)
        results.append(points_3d.unsqueeze(0))

    return torch.cat(results, dim=0)


def decode_predictions(pred_dict: Dict[str, torch.Tensor],
                       calib_P2_list: List[np.ndarray],
                       img_h: int,
                       img_w: int,
                       score_thres: float = 0.4) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Turns the 9 dense per-pixel head outputs into a per-image list of
    actual detected 3D boxes. Runs entirely on CPU, entirely in eager
    PyTorch — this function is never traced or exported to ONNX.

    Returns three lists (one entry per image in the batch):
        bboxes_2d:  (N, 5)  -> x1, y1, x2, y2, score
        bboxes_3d:  (N, 7)  -> x, y, z, length, height, width, rotation_y
        labels:     (N,)    -> class index (0=Ped, 1=Cyclist, 2=Car)
    """
    heatmap = get_local_maximum(pred_dict['center_heatmap'])
    batch, _, feat_h, feat_w = heatmap.shape

    scores, pixel_indices, labels, ys, xs = get_topk_from_heatmap(heatmap, k=TOPK)

    # --- 2D box ---
    wh = gather_at_indices(pred_dict['wh'], pixel_indices)          # (B, K, 2)
    offset = gather_at_indices(pred_dict['offset'], pixel_indices)  # (B, K, 2)

    center_x = xs + offset[..., 0]
    center_y = ys + offset[..., 1]

    x_scale = img_w / feat_w
    y_scale = img_h / feat_h

    x1 = (center_x - wh[..., 0] / 2.) * x_scale
    y1 = (center_y - wh[..., 1] / 2.) * y_scale
    x2 = (center_x + wh[..., 0] / 2.) * x_scale
    y2 = (center_y + wh[..., 1] / 2.) * y_scale

    bboxes_2d = torch.cat([torch.stack([x1, y1, x2, y2], dim=2), scores.unsqueeze(-1)], dim=-1)

    # --- Orientation (alpha -> rotation_y comes after we know the 3D center) ---
    alpha_cls = gather_at_indices(pred_dict['alpha_cls'], pixel_indices)
    alpha_offset = gather_at_indices(pred_dict['alpha_offset'], pixel_indices)
    alpha = decode_alpha(alpha_cls, alpha_offset)                    # (B, K, 1)

    # --- Depth, with uncertainty folded into the 2D confidence score ---
    depth_out = gather_at_indices(pred_dict['depth'], pixel_indices)  # (B, K, 2)
    depth = depth_out[:, :, 0:1]
    depth_confidence = torch.exp(-depth_out[:, :, 1])
    bboxes_2d[..., -1] = bboxes_2d[..., -1] * depth_confidence

    # --- Projected 3D center, via the keypoint-offset head ---
    kpt_offset = gather_at_indices(pred_dict['center2kpt_offset'], pixel_indices)
    kpt_offset = kpt_offset.view(batch, TOPK, NUM_KPTS * 2)[..., -2:]  # last keypoint = projected 3D center

    kpt_offset[..., 0:1] = (kpt_offset[..., 0:1] + xs.unsqueeze(-1)) * x_scale
    kpt_offset[..., 1:2] = (kpt_offset[..., 1:2] + ys.unsqueeze(-1)) * y_scale
    center_2d = kpt_offset  # (B, K, 2), in original image pixel coordinates

    calib_P2_tensor = torch.stack([torch.from_numpy(p).float() for p in calib_P2_list], dim=0)
    rot_y = alpha_to_rotation_y(center_2d[:, :, 0:1], alpha, calib_P2_tensor)

    center_3d = image_to_camera_3d(center_2d, depth, calib_P2_list)   # (B, K, 3)

    dim = gather_at_indices(pred_dict['dim'], pixel_indices)          # (B, K, 3)
    bboxes_3d = torch.cat([center_3d, dim, rot_y], dim=-1)            # (B, K, 7)

    # KITTI convention fix: model predicts box center at (0.5, 0.5, 0.5)
    # origin, but KITTI labels use bottom-center (0.5, 1.0, 0.5). Shift up
    # by half the object height along the box's own y-axis.
    origin_shift = bboxes_3d.new_tensor((0.5, 1.0, 0.5)) - bboxes_3d.new_tensor((0.5, 0.5, 0.5))
    bboxes_3d[:, :, :3] += bboxes_3d[:, :, 3:6] * origin_shift

    # --- Final confidence filtering, per image (variable object count) ---
    keep = bboxes_2d[..., -1] > score_thres

    ret_2d = [b[m] for b, m in zip(bboxes_2d, keep)]
    ret_3d = [b[m] for b, m in zip(bboxes_3d, keep)]
    ret_labels = [l[m] for l, m in zip(labels, keep)]

    return ret_2d, ret_3d, ret_labels