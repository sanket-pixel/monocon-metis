import torch
import torch.nn as nn

from typing import Dict, Tuple
from python.model.attentive_norm import AttnBatchNorm2d

def make_head(out_channels: int) -> nn.Sequential:
    """
    Standard MonoCon prediction head: one 3x3 conv, one attentive-BN,
    ReLU, then a 1x1 conv down to the target number of output channels.
    Used ONLY by the original MonoConHead reference implementation below,
    kept for verify_head_split.py's numerical-equivalence check.
    """
    return nn.Sequential(
        nn.Conv2d(IN_CH, FEAT_CH, kernel_size=3, padding=1),
        AttnBatchNorm2d(FEAT_CH, num_affine_trans=10, momentum=0.03, eps=0.001),
        nn.ReLU(inplace=True),
        nn.Conv2d(FEAT_CH, out_channels, kernel_size=1))
# Fixed constants for this project — MonoCon on KITTI, DLA-34 backbone.
IN_CH = 64          # Channels coming in from the neck
FEAT_CH = 64         # Internal channel width used inside every head
NUM_CLASSES = 3      # Car, Pedestrian, Cyclist
NUM_KPTS = 9         # 8 box corners + 1 projected 3D center
NUM_ALPHA_BINS = 12  # Discretized observation-angle bins

EPS = 1e-12

# Fixed order — every place that iterates heads must agree on this order,
# since HeadConv1's output gets split back into per-head slices by index.
HEAD_NAMES: Tuple[str, ...] = (
    'heatmap', 'wh', 'offset', 'center2kpt_offset',
    'kpt_heatmap', 'kpt_heatmap_offset', 'dim', 'depth', 'dir_feat')

HEAD_OUT_CHANNELS: Tuple[int, ...] = (
    NUM_CLASSES, 2, 2, NUM_KPTS * 2, NUM_KPTS, 2, 3, 2, FEAT_CH)


class HeadConv1(nn.Module):
    """
    The first, expensive 3x3 conv of every head — the part with NO
    data-dependent runtime statistics, so it's pure CNN and belongs on
    the AIPU alongside backbone+neck.

    Each head's first conv is architecturally identical (64 -> 64, 3x3),
    so they're kept as 9 separate Conv2d modules here (matching the
    original checkpoint's per-head layer names exactly) but run back-to-back
    on the same input — a compiler/exporter can still fuse these into one
    wider conv if beneficial; keeping them separate here just guarantees
    the state_dict keys match the original MonoConHead exactly.

    ~9 x (96 x 312 x 64 x 64 x 3x3) MACs ~= 20 GFLOPs total — this is
    roughly 2/3 of the entire backbone+neck's compute, NOT a trivial
    afterthought. This is why it belongs on the AIPU, not on CPU.
    """

    def __init__(self):
        super().__init__()

        for name in HEAD_NAMES:
            setattr(self, f'{name}_conv1', nn.Conv2d(IN_CH, FEAT_CH, kernel_size=3, padding=1))

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat: (B, 64, H/4, W/4) fused feature map from the neck.

        Returns:
            Dict of 9 tensors, each (B, 64, H/4, W/4) — the pre-normalization
            feature map for each head, ready for HeadTail.
        """
        return {name: getattr(self, f'{name}_conv1')(feat) for name in HEAD_NAMES}


class HeadTail(nn.Module):
    """
    Everything AFTER each head's first conv: the data-dependent
    AttnBatchNorm2d (runtime ReduceMean + MatMul — cannot be quantized,
    cannot be folded, cannot run on the AIPU), ReLU, and the cheap 1x1
    second conv. This stays FP32 on CPU via ONNXRuntime.

    Takes HeadConv1's 9 output tensors and produces the final named
    prediction dict that decode.py expects — identical output contract
    to the original single-module MonoConHead.
    """

    def __init__(self):
        super().__init__()

        self.attn_bns = nn.ModuleDict({
            name: AttnBatchNorm2d(FEAT_CH, num_affine_trans=10, momentum=0.03, eps=0.001)
            for name in HEAD_NAMES})

        self.relu = nn.ReLU(inplace=True)

        for name, out_ch in zip(HEAD_NAMES, HEAD_OUT_CHANNELS):
            if name == 'dir_feat':
                continue  # dir_feat's "conv2" is actually dir_cls/dir_reg, handled separately
            setattr(self, f'{name}_conv2', nn.Conv2d(FEAT_CH, out_ch, kernel_size=1))

        self.dir_cls = nn.Sequential(nn.Conv2d(FEAT_CH, NUM_ALPHA_BINS, kernel_size=1))
        self.dir_reg = nn.Sequential(nn.Conv2d(FEAT_CH, NUM_ALPHA_BINS, kernel_size=1))

    def forward(self, conv1_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            conv1_out: dict of 9 tensors from HeadConv1, one per HEAD_NAMES entry.

        Returns:
            Dict of 10 final named prediction tensors — same keys/shapes
            as the original MonoConHead.forward(), ready for decode.py.
        """
        normed = {name: self.relu(self.attn_bns[name](conv1_out[name])) for name in HEAD_NAMES}

        heatmap = self.heatmap_conv2(normed['heatmap'])
        wh = self.wh_conv2(normed['wh'])
        offset = self.offset_conv2(normed['offset'])
        center2kpt_offset = self.center2kpt_offset_conv2(normed['center2kpt_offset'])
        kpt_heatmap = self.kpt_heatmap_conv2(normed['kpt_heatmap'])
        kpt_heatmap_offset = self.kpt_heatmap_offset_conv2(normed['kpt_heatmap_offset'])
        dim = self.dim_conv2(normed['dim'])
        depth_raw = self.depth_conv2(normed['depth'])

        alpha_cls = self.dir_cls(normed['dir_feat'])
        alpha_offset = self.dir_reg(normed['dir_feat'])

        heat_min, heat_max = 1e-4, 1. - 1e-4
        center_heatmap = torch.clamp(torch.sigmoid(heatmap), heat_min, heat_max)
        kpt_heatmap = torch.clamp(torch.sigmoid(kpt_heatmap), heat_min, heat_max)

        depth_value = (1. / (torch.sigmoid(depth_raw[:, 0:1]) + EPS)) - 1
        depth_log_var = depth_raw[:, 1:2]
        depth = torch.cat([depth_value, depth_log_var], dim=1)

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


class MonoConHeadSplit(nn.Module):
    """
    Combines HeadConv1 + HeadTail for use in plain PyTorch (e.g. loading
    a checkpoint, running a sanity-check forward pass). For actual
    deployment, HeadConv1 gets exported as part of the AIPU graph and
    HeadTail gets exported separately for ONNXRuntime CPU — see export.py.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = HeadConv1()
        self.tail = HeadTail()

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.tail(self.conv1(feat))

class MonoConHead(nn.Module):
    """
    Original, unsplit reference implementation — kept here ONLY for
    verify_head_split.py to diff against. Not used in the actual deployed
    pipeline; HeadConv1 + HeadTail (via MonoConHeadSplit) is what gets
    exported and deployed.
    """

    def __init__(self):
        super().__init__()

        self.heatmap_head = make_head(NUM_CLASSES)
        self.wh_head = make_head(2)
        self.offset_head = make_head(2)
        self.center2kpt_offset_head = make_head(NUM_KPTS * 2)
        self.kpt_heatmap_head = make_head(NUM_KPTS)
        self.kpt_heatmap_offset_head = make_head(2)
        self.dim_head = make_head(3)
        self.depth_head = make_head(2)

        self.dir_feat = nn.Sequential(
            nn.Conv2d(IN_CH, FEAT_CH, kernel_size=3, padding=1),
            AttnBatchNorm2d(FEAT_CH, num_affine_trans=10, momentum=0.03, eps=0.001),
            nn.ReLU(inplace=True))
        self.dir_cls = nn.Sequential(nn.Conv2d(FEAT_CH, NUM_ALPHA_BINS, kernel_size=1))
        self.dir_reg = nn.Sequential(nn.Conv2d(FEAT_CH, NUM_ALPHA_BINS, kernel_size=1))

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        heat_min, heat_max = 1e-4, 1. - 1e-4
        center_heatmap = torch.clamp(torch.sigmoid(self.heatmap_head(feat)), heat_min, heat_max)
        kpt_heatmap = torch.clamp(torch.sigmoid(self.kpt_heatmap_head(feat)), heat_min, heat_max)

        wh = self.wh_head(feat)
        offset = self.offset_head(feat)
        kpt_heatmap_offset = self.kpt_heatmap_offset_head(feat)
        center2kpt_offset = self.center2kpt_offset_head(feat)

        dim = self.dim_head(feat)

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

