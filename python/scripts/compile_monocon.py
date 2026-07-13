"""
Quantize and compile MonoCon's backbone+neck+HeadConv1Fused for the Axelera Metis AIPU.

Run this inside the Voyager SDK venv (activate with `v` first).

Only backbone+neck+HeadConv1Fused are compiled — HeadTail stays on CPU.
This split is required: AttnBatchNorm2d contains a ReduceMean op that has
no Voyager quantization converter, confirmed directly by the compiler.

quantization_scheme="per_tensor_min_max" is REQUIRED — not optional.
HeadConv1's raw conv output (pre-normalization) has a true dynamic range
of roughly -1350 to +1200 across the 9 heads. The default histogram scheme
clips this severely, causing near-zero heatmap activations and zero detections.
Confirmed by comparing dequantized AIPU output against FP32 PyTorch reference.
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
from axelera import compiler
from axelera.compiler import CompilerConfig


# Must match python/model/preprocess.py and cpp/src/preprocess/fused_preprocess_quantize.cpp
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)
SIZE_DIVISOR = 32


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Mirrors fused_preprocess_quantize.cpp exactly — must stay in sync."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    norm = (img - MEAN.reshape(1, 1, -1)) / STD.reshape(1, 1, -1)

    ori_h, ori_w = norm.shape[:2]
    padded_h = int(np.ceil(ori_h / SIZE_DIVISOR)) * SIZE_DIVISOR
    padded_w = int(np.ceil(ori_w / SIZE_DIVISOR)) * SIZE_DIVISOR

    canvas = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    canvas[:ori_h, :ori_w, :] = norm

    return np.expand_dims(np.transpose(canvas, (2, 0, 1)), 0)


def calibration_images(image_dir: str, limit: int = None):
    paths = sorted(glob.glob(str(Path(image_dir) / "*.png")))
    if not paths:
        raise FileNotFoundError(f"No .png images found in '{image_dir}'")
    if limit is not None:
        paths = paths[:limit]

    print(f"Using {len(paths)} calibration images from '{image_dir}'")

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  WARNING: could not read '{path}', skipping.")
            continue
        yield img


def main():
    parser = argparse.ArgumentParser(
        "Quantize + compile MonoCon backbone+neck+HeadConv1Fused for Axelera Metis")
    parser.add_argument("--onnx-model",
                        default="weights/onnx/monocon_backbone_neck_conv1.onnx")
    parser.add_argument("--image-dir",
                        default="data/KITTI_sample/images/data",
                        help="Directory of .png calibration images")
    parser.add_argument("--num-calib-images", type=int, default=100)
    parser.add_argument("--compiled-out",
                        default="weights/aipu/monocon_backbone_neck_conv1_fused")
    parser.add_argument("--quantized-out",
                        default="weights/aipu/monocon_backbone_neck_conv1_fused/quantized")
    args = parser.parse_args()

    Path(args.compiled_out).mkdir(parents=True, exist_ok=True)
    Path(args.quantized_out).mkdir(parents=True, exist_ok=True)

    config = CompilerConfig(
        remove_output_dir=True,
        save_error_artifact=True,
        model_name="monocon_backbone_neck_conv1",
        output_dir=args.compiled_out,

        aipu_cores_used=1,
        resources_used=1.0,

        pipeline_spatial_tiles=True,
        pipeline_channel_tiles=True,
        inter_operator_async=True,
        use_hw_tokens=True,
        double_buffer=True,
        enable_buffer_promotion=True,
        enable_icr=True,
        enable_swicr=True,
        dma_dual_channel=True,

        # REQUIRED — see module docstring for why histogram fails here
        quantization_scheme="per_tensor_min_max",
        dpu_allocation_algorithm="lazy",
        tiling_depth=1,
    )

    print("--- Step 1: Quantize ---")
    print(f"  ONNX model:   {args.onnx_model}")
    print(f"  Calib images: {args.image_dir} ({args.num_calib_images} images)")
    print(f"  Scheme:       {config.quantization_scheme}")
    print(f"  Cores:        {config.aipu_cores_used}")

    quantized = compiler.quantize(
        model=args.onnx_model,
        calibration_dataset=calibration_images(
            args.image_dir, limit=args.num_calib_images),
        config=config,
        transform_fn=preprocess,
    )

    print(f"  Saving quantized model to '{args.quantized_out}'")
    quantized.export(args.quantized_out)

    print(f"\n--- Step 2: Compile for Metis ---")
    compiler.compile(
        model=quantized,
        config=config,
        output_dir=Path(args.compiled_out))

    print(f"\nDone. Compiled artifact: '{args.compiled_out}'")


if __name__ == "__main__":
    main()