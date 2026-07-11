"""
Dump HeadTail weights to a flat binary for HeadTailCPU (C++).

HeadTailCPU bypasses ONNXRuntime entirely — it reads weights directly
from this binary and executes the AttnBatchNorm2d + conv2 math in plain
C++, eliminating ORT's ~150-node per-frame dispatch overhead.

Layout (in order, all float32, each tensor prefixed by 4-byte byte-length):
  For each head in HEAD_NAMES order:
    running_mean        (64,)
    running_var         (64,)
    weight_             (10, 64)
    bias_               (10, 64)
    attn_conv_w         (10, 64)   — attention[0].weight squeezed
    inner_bn_gamma      (10,)      — attention[1].weight
    inner_bn_beta       (10,)      — attention[1].bias
    inner_bn_mean       (10,)      — attention[1].running_mean
    inner_bn_var        (10,)      — attention[1].running_var
  For each head in HEAD_NAMES order (skipping dir_feat):
    conv2_weight        (out_ch, 64, 1, 1)
    conv2_bias          (out_ch,)
  dir_cls weight        (12, 64, 1, 1)
  dir_cls bias          (12,)
  dir_reg weight        (12, 64, 1, 1)
  dir_reg bias          (12,)

The C++ reader (HeadTailCPU constructor) reads tensors in this exact order.
"""

import argparse
import struct
import sys
import os

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from python.model.detector import MonoConDetector
from python.model.head import HEAD_NAMES


def _write_tensor(f, t: torch.Tensor) -> None:
    data = t.detach().contiguous().float().numpy().tobytes()
    f.write(struct.pack("<I", len(data)))
    f.write(data)


def dump(detector: MonoConDetector, out_path: str) -> None:
    tail = detector.head.tail

    with open(out_path, "wb") as f:

        # --- Per-head AttnBatchNorm2d weights ---
        for name in HEAD_NAMES:
            bn = tail.attn_bns[name]
            _write_tensor(f, bn.running_mean)
            _write_tensor(f, bn.running_var)
            _write_tensor(f, bn.weight_)
            _write_tensor(f, bn.bias_)
            _write_tensor(f, bn.attn_weights.attention[0].weight.squeeze(-1).squeeze(-1))
            _write_tensor(f, bn.attn_weights.attention[1].weight)
            _write_tensor(f, bn.attn_weights.attention[1].bias)
            _write_tensor(f, bn.attn_weights.attention[1].running_mean)
            _write_tensor(f, bn.attn_weights.attention[1].running_var)

        # --- Per-head conv2 (dir_feat has no conv2 — handled separately) ---
        for name in HEAD_NAMES:
            if name == 'dir_feat':
                continue
            conv2 = getattr(tail, f'{name}_conv2')
            _write_tensor(f, conv2.weight)
            _write_tensor(f, conv2.bias)

        # --- dir_feat branches ---
        _write_tensor(f, tail.dir_cls[0].weight)
        _write_tensor(f, tail.dir_cls[0].bias)
        _write_tensor(f, tail.dir_reg[0].weight)
        _write_tensor(f, tail.dir_reg[0].bias)

    print(f"Dumped HeadTail weights → '{out_path}'")
    print(f"  File size: {os.path.getsize(out_path) / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser("Dump HeadTail weights for HeadTailCPU")
    parser.add_argument("--checkpoint", default="weights/monocon.pth")
    parser.add_argument("--out",        default="weights/head_tail_weights.bin")
    args = parser.parse_args()

    detector = MonoConDetector()
    detector.load_checkpoint(args.checkpoint)
    detector.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dump(detector, args.out)


if __name__ == "__main__":
    main()