"""
MonoCon prediction head — split at the AIPU/CPU boundary.

AIPU boundary (compiled by Voyager, runs on Metis):
    backbone + neck + HeadConv1Fused  →  (1, 576, H, W) single tensor

CPU boundary (runs as HeadTailCPU in C++, exported to ONNX for reference):
    HeadTail  →  10 named prediction tensors

The split is forced by AttnBatchNorm2d: its ReduceMean op has no
Voyager quantization converter, so anything containing it must stay on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Tuple

from python.model.attentive_norm import AttnBatchNorm2d


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IN_CH         = 64   # neck output channels
FEAT_CH       = 64   # internal width of every head
NUM_CLASSES   = 3    # Car, Pedestrian, Cyclist
NUM_KPTS      = 9    # 8 box corners + projected 3D center
NUM_ALPHA_BINS = 12  # observation-angle discretisation bins
EPS           = 1e-12

# Canonical head order — every consumer (export, C++ remap, weight dump)
# must agree on this exact sequence.
HEAD_NAMES: Tuple[str, ...] = (
    'heatmap', 'wh', 'offset', 'center2kpt_offset',
    'kpt_heatmap', 'kpt_heatmap_offset', 'dim', 'depth', 'dir_feat')

HEAD_OUT_CHANNELS: Tuple[int, ...] = (
    NUM_CLASSES, 2, 2, NUM_KPTS * 2, NUM_KPTS, 2, 3, 2, FEAT_CH)


# ---------------------------------------------------------------------------
# AIPU side: HeadConv1Fused
# ---------------------------------------------------------------------------

class HeadConv1Fused(nn.Module):
    """
    Nine independent 64→64 3×3 convs fused into one 64→576 conv.
    Mathematically identical to running them separately (verified in
    scripts/verify_head_conv1_fusion.py — max diff 0.0).

    Output: (1, 576, H, W) — a single tensor, no dict.
    Channel block i (i*64 : (i+1)*64) corresponds to HEAD_NAMES[i].

    Single output avoids the alphabetical-reordering quirk Voyager's
    compiler applies to multi-output graphs.
    """

    def __init__(self):
        super().__init__()
        self.fused_conv1 = nn.Conv2d(IN_CH, FEAT_CH * len(HEAD_NAMES),
                                     kernel_size=3, padding=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fused_conv1(feat)

    def load_from_checkpoint(self, model_state: dict) -> None:
        """
        Loads weights from a raw monocon-pytorch checkpoint state dict.
        Stacks the 9 per-head conv1 weight tensors into the fused layout.
        """
        # Build a temporary unfused module to leverage its clean key structure
        unfused = _HeadConv1Unfused()
        conv1_state = _extract_conv1_state(model_state)
        unfused.load_state_dict(conv1_state, strict=True)

        weights = torch.cat(
            [getattr(unfused, f'{n}_conv1').weight.data for n in HEAD_NAMES], dim=0)
        biases = torch.cat(
            [getattr(unfused, f'{n}_conv1').bias.data for n in HEAD_NAMES], dim=0)
        self.fused_conv1.weight.data.copy_(weights)
        self.fused_conv1.bias.data.copy_(biases)


# ---------------------------------------------------------------------------
# CPU side: HeadTail
# ---------------------------------------------------------------------------

class HeadTail(nn.Module):
    """
    AttnBatchNorm2d + ReLU + 1×1 conv for each head.

    Input:  (1, 576, H, W) — the AIPU's fused HeadConv1 output.
    Output: dict of 10 named prediction tensors for decode.py.

    AttnBatchNorm2d contains ReduceMean (data-dependent runtime stats)
    which Voyager cannot quantize — this class must stay on CPU.
    Exported to ONNX for reference; the deployed C++ pipeline uses
    HeadTailCPU (cpp/src/postprocess/head_tail_cpu.cpp) instead of ORT.
    """

    def __init__(self):
        super().__init__()

        self.attn_bns = nn.ModuleDict({
            name: AttnBatchNorm2d(FEAT_CH, num_affine_trans=10,
                                  momentum=0.03, eps=0.001)
            for name in HEAD_NAMES})

        self.relu = nn.ReLU(inplace=True)

        for name, out_ch in zip(HEAD_NAMES, HEAD_OUT_CHANNELS):
            if name == 'dir_feat':
                continue
            setattr(self, f'{name}_conv2',
                    nn.Conv2d(FEAT_CH, out_ch, kernel_size=1))

        # dir_feat has two output branches instead of one conv2
        self.dir_cls = nn.Sequential(nn.Conv2d(FEAT_CH, NUM_ALPHA_BINS, kernel_size=1))
        self.dir_reg = nn.Sequential(nn.Conv2d(FEAT_CH, NUM_ALPHA_BINS, kernel_size=1))

    def forward(self, fused_conv1_out: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Split 576-channel AIPU output back into 9 per-head (1,64,H,W) slices
        conv1_out = {
            name: fused_conv1_out[:, i * FEAT_CH:(i + 1) * FEAT_CH]
            for i, name in enumerate(HEAD_NAMES)}

        normed = {
            name: self.relu(self.attn_bns[name](conv1_out[name]))
            for name in HEAD_NAMES}

        heatmap           = self.heatmap_conv2(normed['heatmap'])
        wh                = self.wh_conv2(normed['wh'])
        offset            = self.offset_conv2(normed['offset'])
        center2kpt_offset = self.center2kpt_offset_conv2(normed['center2kpt_offset'])
        kpt_heatmap       = self.kpt_heatmap_conv2(normed['kpt_heatmap'])
        kpt_heatmap_offset = self.kpt_heatmap_offset_conv2(normed['kpt_heatmap_offset'])
        dim               = self.dim_conv2(normed['dim'])
        depth_raw         = self.depth_conv2(normed['depth'])
        alpha_cls         = self.dir_cls(normed['dir_feat'])
        alpha_offset      = self.dir_reg(normed['dir_feat'])

        heat_min, heat_max = 1e-4, 1. - 1e-4
        center_heatmap = torch.clamp(torch.sigmoid(heatmap), heat_min, heat_max)
        kpt_heatmap    = torch.clamp(torch.sigmoid(kpt_heatmap), heat_min, heat_max)

        depth = torch.cat([
            (1. / (torch.sigmoid(depth_raw[:, 0:1]) + EPS)) - 1,
            depth_raw[:, 1:2]], dim=1)

        return {
            'center_heatmap':    center_heatmap,
            'kpt_heatmap':       kpt_heatmap,
            'wh':                wh,
            'offset':            offset,
            'kpt_heatmap_offset': kpt_heatmap_offset,
            'center2kpt_offset': center2kpt_offset,
            'dim':               dim,
            'depth':             depth,
            'alpha_cls':         alpha_cls,
            'alpha_offset':      alpha_offset}


# ---------------------------------------------------------------------------
# Combined: MonoConHead (PyTorch-only, for checkpoint loading + sanity checks)
# ---------------------------------------------------------------------------

class MonoConHead(nn.Module):
    """
    Full head for use in Python only (checkpoint loading, verification,
    ONNX export). Not used in the deployed C++ pipeline.

    forward() runs HeadConv1Fused → HeadTail in sequence.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = HeadConv1Fused()
        self.tail  = HeadTail()

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.tail(self.conv1(feat))

    def load_from_checkpoint(self, model_state: dict) -> None:
        self.conv1.load_from_checkpoint(model_state)
        tail_state = _extract_tail_state(model_state)
        self.tail.load_state_dict(tail_state, strict=True)


# ---------------------------------------------------------------------------
# Private helpers — checkpoint key remapping
# ---------------------------------------------------------------------------

class _HeadConv1Unfused(nn.Module):
    """
    Temporary unfused form used only during checkpoint loading.
    Keys match the original monocon-pytorch state dict exactly.
    Never exported, never instantiated outside HeadConv1Fused.load_from_checkpoint.
    """
    def __init__(self):
        super().__init__()
        for name in HEAD_NAMES:
            setattr(self, f'{name}_conv1',
                    nn.Conv2d(IN_CH, FEAT_CH, kernel_size=3, padding=1))


def _extract_conv1_state(model_state: dict) -> dict:
    """Maps original head.{name}_head.0.* keys → {name}_conv1.* keys."""
    state = {}
    for name in HEAD_NAMES:
        prefix = 'dir_feat' if name == 'dir_feat' else f'{name}_head'
        state[f'{name}_conv1.weight'] = model_state[f'head.{prefix}.0.weight']
        state[f'{name}_conv1.bias']   = model_state[f'head.{prefix}.0.bias']
    return state


def _extract_tail_state(model_state: dict) -> dict:
    """Maps original head.{name}_head.1.* keys → attn_bns.{name}.* keys."""
    state = {}
    for name in HEAD_NAMES:
        prefix = 'dir_feat' if name == 'dir_feat' else f'{name}_head'
        attn_prefix = f'head.{prefix}.1.'
        for k, v in model_state.items():
            if k.startswith(attn_prefix):
                state[f'attn_bns.{name}.{k[len(attn_prefix):]}'] = v
        if name != 'dir_feat':
            state[f'{name}_conv2.weight'] = model_state[f'head.{prefix}.3.weight']
            state[f'{name}_conv2.bias']   = model_state[f'head.{prefix}.3.bias']

    state['dir_cls.0.weight'] = model_state['head.dir_cls.0.weight']
    state['dir_cls.0.bias']   = model_state['head.dir_cls.0.bias']
    state['dir_reg.0.weight'] = model_state['head.dir_reg.0.weight']
    state['dir_reg.0.bias']   = model_state['head.dir_reg.0.bias']
    return state