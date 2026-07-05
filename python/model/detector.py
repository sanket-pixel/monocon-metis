import torch
import torch.nn as nn

from typing import Dict

from python.model.backbone import DLA34
from python.model.neck import DLAUp
from python.model.head import MonoConHead


class MonoConDetector(nn.Module):
    """
    Full MonoCon detector: DLA-34 backbone -> DLAUp neck -> 9 dense prediction heads.

    Inference-only. No training path, no loss computation, no target
    generation — those belong to a separate training script, not this
    deployment-focused reimplementation.

    Decoding raw predictions into actual 3D boxes happens OUTSIDE this
    module, in decode.py — kept separate because decode involves
    variable-length, per-image Python control flow that can never be
    traced to ONNX, whereas everything in this module can be.
    """

    def __init__(self):
        super().__init__()

        self.backbone = DLA34()
        self.neck = DLAUp(self.backbone.get_out_channels(start_level=2), start_level=2)
        self.head = MonoConHead()

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            img: (B, 3, H, W) preprocessed input (normalized, padded to
                 a multiple of 32 — see preprocess.py).

        Returns:
            Dict of 9 raw dense prediction tensors, at (B, C, H/4, W/4).
            Pass this straight into decode.decode_predictions().
        """
        feats = self.backbone(img)
        fused_feat = self.neck(list(feats))[0]
        return self.head(fused_feat)

    def load_checkpoint(self, ckpt_path: str) -> None:
        """
        Loads a trained checkpoint saved by the original monocon-pytorch
        training script (state_dict.model, with backbone./neck./head.
        key prefixes). Raises if any tensor fails to match — a silent
        partial load is worse than a loud failure here.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_state = ckpt["state_dict"]["model"]

        for name, module in [("backbone", self.backbone), ("neck", self.neck), ("head", self.head)]:
            prefix = f"{name}."
            sub_state = {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}

            missing, unexpected = module.load_state_dict(sub_state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint mismatch in '{name}': missing={missing}, unexpected={unexpected}")