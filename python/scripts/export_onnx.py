"""
Export MonoCon submodules to ONNX.

Two independent graphs come out of this, matching the AIPU/CPU split:
  - backbone_neck -> INT8 quantized, compiled for the Axelera AIPU
  - head          -> stays FP32, runs via ONNXRuntime on CPU

decode.py is NEVER exported — it has per-image variable-length control
flow (Python loops, boolean masking) that cannot be traced to a static
ONNX graph. It always runs as plain eager PyTorch on CPU.
"""

import argparse
import os
import sys

import onnx
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from python.model.detector import MonoConDetector


# Ops we expect in a pure-CNN graph. Anything outside this set is worth
# investigating before assuming AIPU-compiler compatibility.
EXPECTED_BACKBONE_NECK_OPS = {
    "Conv", "ConvTranspose", "BatchNormalization", "Relu",
    "MaxPool", "Add", "Concat", "Constant", "Cast", "Reshape",
    "Shape", "Gather", "Unsqueeze", "Slice", "Pad",
}


def export_backbone_neck(detector: MonoConDetector, input_h: int, input_w: int,
                         opset: int, out_path: str) -> None:
    """
    Exports backbone+neck as a single graph: (1, 3, H, W) -> (1, 64, H/4, W/4).
    This is the graph that goes to the Axelera compiler.
    """

    class BackboneNeck(torch.nn.Module):
        def __init__(self, detector: MonoConDetector):
            super().__init__()
            self.backbone = detector.backbone
            self.neck = detector.neck

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.backbone(x)
            return self.neck(list(feats))[0]

    model = BackboneNeck(detector)
    model.eval()

    dummy_input = torch.randn(1, 3, input_h, input_w)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            out_path,
            input_names=["input"],
            output_names=["feat"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,  # static shape — required for AIPU compilation
            dynamo=False
        )

    print(f"[backbone_neck] Exported to '{out_path}'")
    _verify_onnx(out_path, expected_ops=EXPECTED_BACKBONE_NECK_OPS)


def export_head(detector: MonoConDetector, feat_h: int, feat_w: int,
                opset: int, out_path: str) -> None:
    """
    Exports the 9 dense prediction heads as a single graph:
    (1, 64, feat_H, feat_W) -> 10 named output tensors.

    feat_h/feat_w must equal input_h/4, input_w/4 from the backbone+neck
    export — the head operates on the neck's output resolution, not the
    original image resolution.
    """

    class HeadWrapper(torch.nn.Module):
        """
        ONNX export requires a fixed, ordered set of outputs — a dict
        won't trace cleanly. This wraps MonoConHead's dict output into
        an explicit tuple with a fixed, documented order.
        """
        OUTPUT_ORDER = (
            'center_heatmap', 'kpt_heatmap', 'wh', 'offset',
            'kpt_heatmap_offset', 'center2kpt_offset', 'dim', 'depth',
            'alpha_cls', 'alpha_offset',
        )

        def __init__(self, detector: MonoConDetector):
            super().__init__()
            self.head = detector.head

        def forward(self, feat: torch.Tensor):
            pred_dict = self.head(feat)
            return tuple(pred_dict[k] for k in self.OUTPUT_ORDER)

    model = HeadWrapper(detector)
    model.eval()

    dummy_feat = torch.randn(1, 64, feat_h, feat_w)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_feat,
            out_path,
            input_names=["feat"],
            output_names=list(HeadWrapper.OUTPUT_ORDER),
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False
        )

    print(f"[head] Exported to '{out_path}'")
    _verify_onnx(out_path, expected_ops=None)  # head has AttnBatchNorm2d - MatMul/ReduceMean expected, not flagged


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
    parser = argparse.ArgumentParser("Export MonoCon submodules to ONNX")
    parser.add_argument("--part", choices=["backbone_neck", "head", "full"], default="full",
                       help="Which part to export. 'full' exports both.")
    parser.add_argument("--checkpoint", type=str, default="best.pth")
    parser.add_argument("--input-h", type=int, default=384, help="Padded input image height")
    parser.add_argument("--input-w", type=int, default=1248, help="Padded input image width")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out-dir", type=str, default="onnx_export")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    detector = MonoConDetector()
    detector.load_checkpoint(args.checkpoint)
    detector.eval()

    feat_h, feat_w = args.input_h // 4, args.input_w // 4

    if args.part in ("backbone_neck", "full"):
        out_path = os.path.join(args.out_dir, "monocon_backbone_neck.onnx")
        export_backbone_neck(detector, args.input_h, args.input_w, args.opset, out_path)

    if args.part in ("head", "full"):
        out_path = os.path.join(args.out_dir, "monocon_head.onnx")
        export_head(detector, feat_h, feat_w, args.opset, out_path)


if __name__ == "__main__":
    main()