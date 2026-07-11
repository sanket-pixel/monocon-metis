"""
MonoCon detector: DLA-34 backbone → DLAUp neck → MonoConHead.

The head is split at the AIPU/CPU boundary inside MonoConHead:
  - HeadConv1Fused  → compiled to Axelera Metis AIPU
  - HeadTail        → runs as HeadTailCPU in C++ (ONNX exported for reference)

This class handles Python-side usage only: checkpoint loading, ONNX export,
and verification scripts. The deployed C++ pipeline (monocon.hpp/cpp) owns
the actual runtime.
"""

import torch
import torch.nn as nn
from typing import Dict

from python.model.backbone import DLA34
from python.model.neck import DLAUp
from python.model.head import MonoConHead


class MonoConDetector(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone  = DLA34()
        self.neck      = DLAUp(self.backbone.get_out_channels(start_level=2), start_level=2)
        self.head      = MonoConHead()

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats      = self.backbone(img)
        fused_feat = self.neck(list(feats))[0]
        return self.head(fused_feat)

    def load_checkpoint(self, ckpt_path: str) -> None:
        """
        Loads a monocon-pytorch checkpoint. Backbone and neck keys map
        directly. Head keys are remapped from the original sequential
        layout (head.<name>_head.0/1/3.*) — see head.py's private helpers.
        """
        ckpt        = torch.load(ckpt_path, map_location="cpu")
        model_state = ckpt["state_dict"]["model"]

        for name, module in [("backbone", self.backbone), ("neck", self.neck)]:
            prefix    = f"{name}."
            sub_state = {k[len(prefix):]: v
                         for k, v in model_state.items()
                         if k.startswith(prefix)}
            missing, unexpected = module.load_state_dict(sub_state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint mismatch in '{name}': "
                    f"missing={missing}, unexpected={unexpected}")

        self.head.load_from_checkpoint(model_state)