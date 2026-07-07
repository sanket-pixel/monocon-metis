"""
Export MonoCon submodules to ONNX.

Two graphs come out of this, matching the AIPU/CPU split:
  - backbone_neck_conv1 -> INT8 quantized, compiled for the Axelera AIPU.
    Includes backbone + neck + HeadConv1 (the 9 heads' first 3x3 conv,
    ~20 GFLOPs — pure CNN, no data-dependent ops).
  - head_tail -> stays FP32, runs via ONNXRuntime on CPU. Contains
    AttnBatchNorm2d (runtime ReduceMean+MatMul, cannot be quantized —
    confirmed directly by the Voyager compiler) + cheap 1x1 convs.

decode.py is NEVER exported — it has per-image variable-length control
flow (Python loops, boolean masking) that cannot be traced to a static
ONNX graph. It always runs as plain eager PyTorch on CPU.
"""

import argparse
import os
import sys

import onnx
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from python.model.detector import MonoConDetector
from python.model.head import HEAD_NAMES


EXPECTED_AIPU_OPS = {
    "Conv", "ConvTranspose", "BatchNormalization", "Relu",
    "MaxPool", "Add", "Concat", "Constant", "Cast", "Reshape",
    "Shape", "Gather", "Unsqueeze", "Slice", "Pad",
}


def export_backbone_neck_conv1(detector: MonoConDetector, input_h: int, input_w: int,
                               opset: int, out_path: str) -> None:
    """
    Exports backbone + neck + HeadConv1 as a single graph:
    (1, 3, H, W) -> 9 named (1, 64, H/4, W/4) tensors, one per head.

    This is the graph that goes to the Axelera compiler. Bigger than the
    original backbone_neck-only export, but still pure CNN — no
    ReduceMean/MatMul, since HeadConv1 stops right before AttnBatchNorm2d.
    """

    class BackboneNeckConv1(torch.nn.Module):
        def __init__(self, detector: MonoConDetector):
            super().__init__()
            self.backbone = detector.backbone
            self.neck = detector.neck
            self.head_conv1 = detector.head_conv1

        def forward(self, x: torch.Tensor):
            feats = self.backbone(x)
            fused = self.neck(list(feats))[0]
            conv1_out = self.head_conv1(fused)
            return tuple(conv1_out[name] for name in HEAD_NAMES)

    model = BackboneNeckConv1(detector)
    model.eval()

    dummy_input = torch.randn(1, 3, input_h, input_w)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            out_path,
            input_names=["input"],
            output_names=list(HEAD_NAMES),
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )

    print(f"[backbone_neck_conv1] Exported to '{out_path}'")
    _verify_onnx(out_path, expected_ops=EXPECTED_AIPU_OPS)


def export_head_tail(detector: MonoConDetector, feat_h: int, feat_w: int,
                     opset: int, out_path: str) -> None:
    """
    Exports HeadTail as a single graph: 9 named (1, 64, feat_H, feat_W)
    inputs -> 10 named output tensors. Runs via ONNXRuntime CPU.
    """

    class HeadTailWrapper(torch.nn.Module):
        OUTPUT_ORDER = (
            'center_heatmap', 'kpt_heatmap', 'wh', 'offset',
            'kpt_heatmap_offset', 'center2kpt_offset', 'dim', 'depth',
            'alpha_cls', 'alpha_offset',
        )

        def __init__(self, detector: MonoConDetector):
            super().__init__()
            self.head_tail = detector.head_tail

        def forward(self, *conv1_outs):
            conv1_dict = dict(zip(HEAD_NAMES, conv1_outs))
            pred_dict = self.head_tail(conv1_dict)
            return tuple(pred_dict[k] for k in self.OUTPUT_ORDER)

    model = HeadTailWrapper(detector)
    model.eval()

    dummy_inputs = tuple(torch.randn(1, 64, feat_h, feat_w) for _ in HEAD_NAMES)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            out_path,
            input_names=list(HEAD_NAMES),
            output_names=list(HeadTailWrapper.OUTPUT_ORDER),
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )

    print(f"[head_tail] Exported to '{out_path}'")
    _verify_onnx(out_path, expected_ops=None)  # AttnBatchNorm2d -> MatMul/ReduceMean expected, not flagged


def _verify_onnx(onnx_path: str, expected_ops: set = None) -> None:
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    op_types = sorted(set(node.op_type for node in model.graph.node))
    print(f"  Op types: {op_types}")

    if expected_ops is not None:
        unexpected = set(op_types) - expected_ops
        if unexpected:
            print(f"  WARNING: non-standard-CNN ops found: {unexpected}")
        else:
            print(f"  All ops are standard CNN ops.")


def main():
    parser = argparse.ArgumentParser("Export MonoCon submodules to ONNX (split-head version)")
    parser.add_argument("--part", choices=["backbone_neck_conv1", "head_tail", "full"], default="full",
                        help="Which part to export. 'full' exports both.")
    parser.add_argument("--checkpoint", type=str, default="weights/monocon.pth")
    parser.add_argument("--input-h", type=int, default=384, help="Padded input image height")
    parser.add_argument("--input-w", type=int, default=1248, help="Padded input image width")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--out-dir", type=str, default="weights/onnx")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    detector = MonoConDetector()
    detector.load_checkpoint(args.checkpoint)
    detector.eval()

    feat_h, feat_w = args.input_h // 4, args.input_w // 4

    if args.part in ("backbone_neck_conv1", "full"):
        out_path = os.path.join(args.out_dir, "monocon_backbone_neck_conv1.onnx")
        export_backbone_neck_conv1(detector, args.input_h, args.input_w, args.opset, out_path)

    if args.part in ("head_tail", "full"):
        out_path = os.path.join(args.out_dir, "monocon_head_tail.onnx")
        export_head_tail(detector, feat_h, feat_w, args.opset, out_path)


if __name__ == "__main__":
    main()