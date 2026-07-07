import torch
import torch.nn as nn

from typing import Dict

from python.model.backbone import DLA34
from python.model.neck import DLAUp
from python.model.head import HeadConv1, HeadTail, HEAD_NAMES


class MonoConDetector(nn.Module):
    """
    Full MonoCon detector: DLA-34 backbone -> DLAUp neck -> HeadConv1 -> HeadTail.

    The head is split in two, matching the AIPU/CPU deployment boundary:
      - HeadConv1: the 9 heads' first 3x3 conv — pure CNN, ~20 GFLOPs,
        belongs on the AIPU alongside backbone+neck.
      - HeadTail: AttnBatchNorm2d (runtime ReduceMean+MatMul, cannot be
        quantized/compiled for AIPU) + cheap 1x1 convs — stays on CPU.

    Inference-only. No training path, no loss computation. Decoding raw
    predictions into actual 3D boxes happens OUTSIDE this module, in
    decode.py — it has per-image variable-length control flow that can
    never be traced to ONNX.
    """

    def __init__(self):
        super().__init__()

        self.backbone = DLA34()
        self.neck = DLAUp(self.backbone.get_out_channels(start_level=2), start_level=2)
        self.head_conv1 = HeadConv1()
        self.head_tail = HeadTail()

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            img: (B, 3, H, W) preprocessed input (normalized, padded to
                 a multiple of 32 — see preprocess.py).

        Returns:
            Dict of 10 final named prediction tensors, ready for
            decode.decode_predictions().
        """
        feats = self.backbone(img)
        fused_feat = self.neck(list(feats))[0]
        conv1_out = self.head_conv1(fused_feat)
        return self.head_tail(conv1_out)

    def load_checkpoint(self, ckpt_path: str) -> None:
        """
        Loads a trained checkpoint saved by the original monocon-pytorch
        training script. The original checkpoint's 'head.*' keys use
        nn.Sequential indices (head.<name>_head.0/1/3.*) — this remaps
        them onto HeadConv1/HeadTail's split layer names. See
        python/scripts/verify_head_split.py for the numerical-equivalence
        proof that this remap is exact.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_state = ckpt["state_dict"]["model"]

        for name, module in [("backbone", self.backbone), ("neck", self.neck)]:
            prefix = f"{name}."
            sub_state = {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}
            missing, unexpected = module.load_state_dict(sub_state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint mismatch in '{name}': missing={missing}, unexpected={unexpected}")

        conv1_state, tail_state = self._build_split_head_state_dict(model_state)

        missing, unexpected = self.head_conv1.load_state_dict(conv1_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch in 'head_conv1': missing={missing}, unexpected={unexpected}")

        missing, unexpected = self.head_tail.load_state_dict(tail_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch in 'head_tail': missing={missing}, unexpected={unexpected}")

    @staticmethod
    def _build_split_head_state_dict(model_state: dict):
        """
        Remaps original MonoConHead checkpoint keys onto HeadConv1 + HeadTail.
        See verify_head_split.py for the reference implementation and the
        numerical-equivalence check this remap was validated against.
        """
        conv1_state = {}
        tail_state = {}

        for name in HEAD_NAMES:
            prefix = 'dir_feat' if name == 'dir_feat' else f'{name}_head'

            conv1_state[f'{name}_conv1.weight'] = model_state[f'head.{prefix}.0.weight']
            conv1_state[f'{name}_conv1.bias'] = model_state[f'head.{prefix}.0.bias']

            attn_prefix = f'head.{prefix}.1.'
            for k, v in model_state.items():
                if k.startswith(attn_prefix):
                    suffix = k[len(attn_prefix):]
                    tail_state[f'attn_bns.{name}.{suffix}'] = v

            if name != 'dir_feat':
                tail_state[f'{name}_conv2.weight'] = model_state[f'head.{prefix}.3.weight']
                tail_state[f'{name}_conv2.bias'] = model_state[f'head.{prefix}.3.bias']

        tail_state['dir_cls.0.weight'] = model_state['head.dir_cls.0.weight']
        tail_state['dir_cls.0.bias'] = model_state['head.dir_cls.0.bias']
        tail_state['dir_reg.0.weight'] = model_state['head.dir_reg.0.weight']
        tail_state['dir_reg.0.bias'] = model_state['head.dir_reg.0.bias']

        return conv1_state, tail_state