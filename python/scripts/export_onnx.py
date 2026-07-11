"""
Export MonoCon's two deployment graphs to ONNX.

backbone_neck_conv1  — pure CNN, compiled for Axelera Metis AIPU
                       input:  (1, 3, H, W)
                       output: (1, 576, H/4, W/4)  — HeadConv1Fused output

head_tail            — FP32, exported for reference / CPU fallback
                       input:  (1, 576, H/4, W/4)
                       output: 10 named prediction tensors

decode.py is never exported — variable-length control flow cannot be traced.
"""

import argparse
import os
import sys

import onnx
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from python.model.detector import MonoConDetector
from python.model.head import HEAD_NAMES, FEAT_CH


# Ops expected in a pure-CNN AIPU graph — anything outside this set
# likely contains data-dependent ops that cannot be quantized.
_AIPU_OPS = {
    "Conv", "ConvTranspose", "BatchNormalization", "Relu",
    "MaxPool", "Add", "Concat", "Constant", "Cast",
    "Reshape", "Shape", "Gather", "Unsqueeze", "Slice", "Pad",
}

_HEAD_TAIL_OUTPUT_ORDER = (
    'center_heatmap', 'kpt_heatmap', 'wh', 'offset',
    'kpt_heatmap_offset', 'center2kpt_offset', 'dim', 'depth',
    'alpha_cls', 'alpha_offset',
)


def export_backbone_neck_conv1(detector: MonoConDetector,
                               input_h: int, input_w: int,
                               opset: int, out_path: str) -> None:
    """backbone + neck + HeadConv1Fused → (1, 576, H/4, W/4)."""

    class _Wrapper(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.backbone  = d.backbone
            self.neck      = d.neck
            self.head_conv1 = d.head.conv1  # HeadConv1Fused

        def forward(self, x):
            feats = self.backbone(x)
            fused = self.neck(list(feats))[0]
            return self.head_conv1(fused)   # (1, 576, H/4, W/4)

    _export(
        model       = _Wrapper(detector),
        dummy       = torch.randn(1, 3, input_h, input_w),
        out_path    = out_path,
        input_names = ["input"],
        output_names= ["feat"],
        opset       = opset,
        check_ops   = _AIPU_OPS,
        label       = "backbone_neck_conv1",
    )


def export_head_tail(detector: MonoConDetector,
                     feat_h: int, feat_w: int,
                     opset: int, out_path: str) -> None:
    """HeadTail: (1, 576, H/4, W/4) → 10 named prediction tensors."""

    class _Wrapper(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.tail = d.head.tail  # HeadTail

        def forward(self, feat):
            pred = self.tail(feat)
            return tuple(pred[k] for k in _HEAD_TAIL_OUTPUT_ORDER)

    _export(
        model       = _Wrapper(detector),
        dummy       = torch.randn(1, FEAT_CH * len(HEAD_NAMES), feat_h, feat_w),
        out_path    = out_path,
        input_names = ["feat"],
        output_names= list(_HEAD_TAIL_OUTPUT_ORDER),
        opset       = opset,
        check_ops   = None,   # ReduceMean/MatMul expected, not flagged
        label       = "head_tail",
    )


def _export(model, dummy, out_path, input_names, output_names,
            opset, check_ops, label):
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, out_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )

    graph = onnx.load(out_path)
    onnx.checker.check_model(graph)
    op_types = sorted({n.op_type for n in graph.graph.node})
    print(f"[{label}] → {out_path}")
    print(f"  ops: {op_types}")

    if check_ops is not None:
        unexpected = set(op_types) - check_ops
        if unexpected:
            print(f"  WARNING: unexpected ops (may not compile for AIPU): {unexpected}")

    print()


def main():
    parser = argparse.ArgumentParser("Export MonoCon to ONNX")
    parser.add_argument("--part",
                        choices=["backbone_neck_conv1", "head_tail", "full"],
                        default="full")
    parser.add_argument("--checkpoint", default="weights/monocon.pth")
    parser.add_argument("--input-h",   type=int, default=384)
    parser.add_argument("--input-w",   type=int, default=1248)
    parser.add_argument("--opset",     type=int, default=18)
    parser.add_argument("--out-dir",   default="weights/onnx")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    detector = MonoConDetector()
    detector.load_checkpoint(args.checkpoint)
    detector.eval()

    feat_h = args.input_h // 4
    feat_w = args.input_w // 4

    if args.part in ("backbone_neck_conv1", "full"):
        export_backbone_neck_conv1(
            detector, args.input_h, args.input_w, args.opset,
            os.path.join(args.out_dir, "monocon_backbone_neck_conv1.onnx"))

    if args.part in ("head_tail", "full"):
        export_head_tail(
            detector, feat_h, feat_w, args.opset,
            os.path.join(args.out_dir, "monocon_head_tail.onnx"))


if __name__ == "__main__":
    main()