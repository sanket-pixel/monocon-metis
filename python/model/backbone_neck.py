import torch
import torch.nn as nn

from python.model.backbone import DLA34
from python.model.neck import DLAUp


class BackboneNeck(nn.Module):
    """Wraps DLA-34 backbone + DLAUp neck into a single traceable module for ONNX export."""

    def __init__(self, num_dla_layers: int = 34, pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = DLA34()
        self.neck = DLAUp(self.backbone.get_out_channels(start_level=2), start_level=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        out = self.neck(list(feats))
        return out[0]  # single fused feature map, (B, 64, H/4, W/4)