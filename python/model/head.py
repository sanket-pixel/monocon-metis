import torch
import torch.nn as nn

from typing import Dict
from python.model.attentive_norm import AttnBatchNorm2d


# Fixed constants for this project — MonoCon on KITTI, DLA-34 backbone.
IN_CH = 64          # Channels coming in from the neck
FEAT_CH = 64         # Internal channel width used inside every head
NUM_CLASSES = 3      # Car, Pedestrian, Cyclist
NUM_KPTS = 9         # 8 box corners + 1 projected 3D center
NUM_ALPHA_BINS = 12  # Discretized observation-angle bins

EPS = 1e-12


def make_head(out_channels: int) -> nn.Sequential:
    """
    Standard MonoCon prediction head: one 3x3 conv, one attentive-BN,
    ReLU, then a 1x1 conv down to the target number of output channels.
    Every head below uses exactly this shape.
    """
    return nn.Sequential(
        nn.Conv2d(IN_CH, FEAT_CH, kernel_size=3, padding=1),
        AttnBatchNorm2d(FEAT_CH, num_affine_trans=10, momentum=0.03, eps=0.001),
        nn.ReLU(inplace=True),
        nn.Conv2d(FEAT_CH, out_channels, kernel_size=1))


class MonoConHead(nn.Module):
    """
    Inference-only dense prediction heads for MonoCon.

    Takes the neck's single fused feature map (B, 64, H/4, W/4) and produces
    9 dense per-pixel prediction maps. No losses, no target generation —
    this is pure forward-pass inference.

    All 9 heads run independently in parallel on the same input feature map.
    """

    def __init__(self):
        super().__init__()

        # --- 2D box properties ---
        self.heatmap_head = make_head(NUM_CLASSES)          # per-class object-center heatmap
        self.wh_head = make_head(2)                         # 2D box width, height
        self.offset_head = make_head(2)                     # sub-pixel center offset

        # --- 2D <-> 3D keypoint properties ---
        self.center2kpt_offset_head = make_head(NUM_KPTS * 2)  # center -> each of 9 keypoints
        self.kpt_heatmap_head = make_head(NUM_KPTS)            # per-keypoint heatmap
        self.kpt_heatmap_offset_head = make_head(2)            # sub-pixel keypoint offset

        # --- 3D box properties ---
        self.dim_head = make_head(3)                       # object length, height, width
        self.depth_head = make_head(2)                     # [depth, log-variance/uncertainty]

        # Direction (observation angle "alpha") head: shared stem, then
        # a bin-classification branch and a within-bin-offset regression branch.
        self.dir_feat = nn.Sequential(
            nn.Conv2d(IN_CH, FEAT_CH, kernel_size=3, padding=1),
            AttnBatchNorm2d(FEAT_CH, num_affine_trans=10, momentum=0.03, eps=0.001),
            nn.ReLU(inplace=True))
        self.dir_cls = nn.Sequential(nn.Conv2d(FEAT_CH, NUM_ALPHA_BINS, kernel_size=1))
        self.dir_reg = nn.Sequential(nn.Conv2d(FEAT_CH, NUM_ALPHA_BINS, kernel_size=1))

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat: (B, 64, H/4, W/4) fused feature map from the neck.

        Returns:
            Dict of 9 raw dense prediction tensors, still in per-pixel
            feature-map form (not yet decoded into actual object boxes —
            that happens later, on CPU, in decode.py).
        """

        # Heatmaps: raw logits -> sigmoid confidence, clamped away from 0/1
        # to keep downstream log/loss math well-behaved.
        heat_min, heat_max = 1e-4, 1. - 1e-4
        center_heatmap = torch.clamp(torch.sigmoid(self.heatmap_head(feat)), heat_min, heat_max)
        kpt_heatmap = torch.clamp(torch.sigmoid(self.kpt_heatmap_head(feat)), heat_min, heat_max)

        wh = self.wh_head(feat)
        offset = self.offset_head(feat)
        kpt_heatmap_offset = self.kpt_heatmap_offset_head(feat)
        center2kpt_offset = self.center2kpt_offset_head(feat)

        dim = self.dim_head(feat)

        # Depth: channel 0 is the depth estimate, encoded through an
        # inverse-sigmoid so the network only ever predicts positive depth.
        # Channel 1 is left untouched — it's a log-variance / uncertainty term.
        depth_raw = self.depth_head(feat)
        depth_value = (1. / (torch.sigmoid(depth_raw[:, 0:1]) + EPS)) - 1
        depth_log_var = depth_raw[:, 1:2]
        depth = torch.cat([depth_value, depth_log_var], dim=1)

        alpha_feat = self.dir_feat(feat)
        alpha_cls = self.dir_cls(alpha_feat)
        alpha_offset = self.dir_reg(alpha_feat)

        return {
            'center_heatmap': center_heatmap,
            'kpt_heatmap': kpt_heatmap,
            'wh': wh,
            'offset': offset,
            'kpt_heatmap_offset': kpt_heatmap_offset,
            'center2kpt_offset': center2kpt_offset,
            'dim': dim,
            'depth': depth,
            'alpha_cls': alpha_cls,
            'alpha_offset': alpha_offset}