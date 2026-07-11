# Monocular 3D Object Detection on Axelera Metis

> End-to-end 3D object detection from a single camera, running entirely on an
> Axelera Metis M.2 AIPU single core + CPU host, no GPU required at inference time
> Running at ~25 FPS end to end with single core, no latency , no batching.

![Demo](data/kitti.gif)

---

## 1. Results

**Hardware:** AMD Ryzen desktop + Axelera Metis M.2 PCIe · 1 AIPU core · CPU host

| Stage | Time |
|---|---|
| Preprocess + Quantize | 1.4 ms |
| AIPU (backbone + neck + HeadConv1) | 31 ms |
| Dequantize | 6.5 ms |
| Head (C++ AttnBatchNorm2d) | 22 ms |
| Decode | 0.9 ms |
| **Sequential total** | **~62 ms** |
| **Pipelined throughput** | **41 ms · 24.5 FPS** |

Pipelined throughput measured over 433 real KITTI frames using a two-thread
producer-consumer pipeline that overlaps AIPU inference with CPU-side head
computation, while the AIPU processes frame N+1, the CPU decodes frame N.

**Progression from baseline to final:**

| Version | FPS | What changed |
|---|---|---|
| Baseline (ORT head, sequential) | 3.5 | Starting point |
| Head → custom C++ | 9 | Replaced 150-node ORT graph |
| Fused preprocess + quantize | 11 | One pass, no float intermediate |
| Parallelized dequantize | 16 | OpenMP over spatial rows |
| Producer-consumer pipeline | **24.5** | AIPU/CPU overlap across frames |


## 2. Architecture

The pipeline splits cleanly at the AIPU/CPU boundary, forced by a hard
compiler constraint: `AttnBatchNorm2d` contains a `ReduceMean` op that
Voyager's quantization toolchain cannot convert. Everything before it runs
on the AIPU; everything after runs on the CPU host.

<pre>
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                              │
│                    (1 × 3 × 384 × 1248)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │  fused normalize + quantize (C++)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AXELERA METIS AIPU                          │
│                                                                 │
│   DLA-34 Backbone  →  DLAUp Neck  →  HeadConv1Fused             │
│                                    (64 → 576 channels, 3×3)     │
│                                                                 │
│                 Output: (1 × 576 × 96 × 312) INT8               │
└────────────────────────────┬────────────────────────────────────┘
                             │  dequantize (C++, OpenMP)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CPU HOST                                 │
│                                                                 │
│   HeadTailCPU                                                   │
│   ├─ Split 576ch → 9 × 64ch slices                              │
│   ├─ AttnBatchNorm2d × 9  (ReduceMean + matmul)                 │
│   ├─ ReLU × 9                                                   │
│   └─ Conv1×1 × 9  →  10 named prediction tensors                │
│                                                                 │
│   Decode  →  NMS  →  3D Detections                              │
└─────────────────────────────────────────────────────────────────┘
</pre>

### AIPU Graph — backbone + neck + HeadConv1Fused

The DLA-34 backbone and DLAUp neck are standard pure-CNN — no
data-dependent ops, fully quantizable, natural fit for the AIPU.

The nine prediction head first-convs (`64 → 64`, `3×3`, one per output
head) were fused into a single `64 → 576` conv (`HeadConv1Fused`) before
export. This gives the AIPU compiler one scheduling decision instead of
nine, produces a single clean output tensor, and avoids the alphabetical
output-reordering quirk that affects multi-output compiled graphs
(see [Challenges](#challenges)).

### CPU Head — HeadTailCPU

`AttnBatchNorm2d` computes spatial mean/variance over the full `96 × 312`
feature map at runtime — inherently data-dependent, cannot be folded into
static weights, cannot be quantized. It stays on CPU.

The naive approach (ONNXRuntime session over the exported HeadTail graph)
measured at **55–60 ms** despite the model having only **24,121 parameters**.
The bottleneck was not arithmetic — it was ORT dispatching ~150 small ops
through its general-purpose graph executor, each carrying fixed per-node
overhead. At this parameter count, dispatch dominates completely.

The fix: a hand-written `HeadTailCPU` class in C++ that eliminates the
runtime graph entirely. All weights loaded once at construction, all scratch
buffers preallocated — zero heap allocation per frame. The nine heads run
in parallel via OpenMP. Result: **22 ms**, a **2.7× improvement** over ORT.

### Pipelined Video Inference

Sequential processing leaves both the AIPU and CPU idle roughly half the
time — the AIPU waits while the CPU runs the head, and vice versa. A
two-thread producer-consumer pipeline eliminates this:

<pre>
Thread A (AIPU):  ──[preprocess+AIPU]──[preprocess+AIPU]──[preprocess+AIPU]──▶
                                    ╲                   ╲
                              queue  ╲             queue  ╲
                                      ╲                     ╲
Thread B (CPU):              ──[dequant+head+decode]──[dequant+head+decode]──▶

                  ◀────────── frame N ─────────────────── frame N+1 ──────────▶
</pre>

Thread A produces raw INT8 AIPU output into a bounded queue (depth 3).
Thread B consumes from the queue, running dequantize + head + decode.
Since AIPU time (~31 ms) and CPU time (~29 ms) are nearly matched, the
pipeline runs at `max(31, 29) ≈ 31 ms/frame` sustained — **24.5 FPS**
over 433 frames.

## 3. Setup

### Prerequisites

- Axelera Metis M.2 or PCIe board with Voyager SDK v1.5.x installed
- Python 3.10+ with [uv](https://github.com/astral-sh/uv)
- CMake 3.18+, GCC with OpenMP, OpenCV 4.x, Eigen 3.x
- KITTI raw dataset or the sample sequences used here [unrecified raw sequence](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0093/2011_09_26_drive_0093_sync.zip)
and get calibration data from [calib](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip).
---

### 1. Get the model weights

Download the pretrained checkpoint from the
[monocon-pytorch releases](https://drive.google.com/drive/folders/1yVgt8cU-aHtoteATha_7_2U4TxseSrBX)
and place it at:

```
weights/monocon.pth
```

---

### 2. Get KITTI data

This project uses one KITTI raw sequence for both calibration and inference.

**Download the sequence** (synced + rectified):
```
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0093/2011_09_26_drive_0093_sync.zip
```

**Download the calibration files:**
```
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip
```

> For other sequences, the full KITTI Raw Dataset can be downloaded by scene
> from [here](https://www.cvlibs.net/datasets/kitti/raw_data.php?type=city)
> (login required). Download only **"synced+rectified data"** and
> **"calibration"** for each scene.

Extract and arrange the files as follows:

```
data/
└── KITTI_sample/
├── calib/
│   └── calib_cam_to_cam.txt          ← from calibration zip
└── images/
└── data/
└── *.png                     ← from sequence zip
```


The same sequence serves as both the **calibration dataset** for AIPU
quantization and the **test sequence** for inference.


### 3. Install Python dependencies

```bash
uv sync
```

---

### 4. Export ONNX graphs and head weights

Exports `backbone_neck_conv1.onnx`, `monocon_head_tail.onnx`, and
`head_tail_weights.bin` in one command:

```bash
make export
```

This runs:
- `python/scripts/export_onnx.py` — two ONNX graphs
- `python/scripts/export_head_tail_weights.py` — binary weights for `HeadTailCPU`

---

### 5. Compile for the AIPU

Requires the Voyager SDK venv. Activate it first, then compile:

```bash
 # activate Voyager SDK venv (~/voyager-sdk/.venv/bin/activate)
make compile
```

This quantizes with `per_tensor_min_max` (required — see [Challenges](#challenges))
and compiles to `weights/aipu/monocon_backbone_neck_conv1_fused/`.
Expect ~10–20 minutes depending on your machine.

---

### 6. Build the C++ inference binary

```bash
make build
```

Builds with `-O3 -march=native` and links against OpenMP, OpenCV, and Eigen.
The binary lands at `cpp/cmake-build-release/monocon_infer`.

---

### 7. Run

**Single image — annotated result saved to `output/sample_result.png`:**

```bash
make run
```

**Per-stage timing breakdown:**

```bash
make profile
```

```
preprocess+quantize:  1.4 ms
aipu execution:      31.1 ms
dequantize:           6.5 ms
head:                22.4 ms
decode:               0.9 ms
total:               62.0 ms   (16.1 FPS sequential)
```

**Pipelined video over full sequence — saved to `output/monocon_demo.mp4`:**

```bash
make video
```

```
Frames:          433
Total wall time: 17652 ms
Throughput FPS:  24.5
```


## Challenges

Six non-obvious problems encountered during deployment, none of them
documented in the Voyager SDK. Recorded here so the next person doesn't
spend weeks on the same issues.

---

### 1. `AttnBatchNorm2d` cannot be quantized

**Symptom:** Compiler error when trying to compile the full model in one graph:
```
axelera.compiler.exceptions.QtoolsError: Converter is not implemented
(OperationDescription(domain='', operation_type='ReduceMean', version=18))
```

**Cause:** `AttnBatchNorm2d` computes spatial mean and variance at runtime
(`torch.var_mean` over the full `H × W` feature map). This is a
data-dependent operation — it cannot be folded into static weights and has
no Voyager quantization converter.

**Fix:** Split the model at this boundary. Everything before
`AttnBatchNorm2d` (backbone + neck + HeadConv1) runs on the AIPU.
Everything from `AttnBatchNorm2d` onward runs on the CPU host.


### 2. `per_tensor_histogram` clips HeadConv1 activations

**Symptom:** Detections compiled with the default quantization scheme
produced a `center_heatmap` with `max ≈ 0.04` instead of the expected
`~0.95`, and zero detections at any threshold.

**Cause:** HeadConv1's raw conv output (before `AttnBatchNorm2d`
normalizes it) has a true dynamic range of roughly **−1350 to +1200**
across the nine heads. The default `per_tensor_histogram` scheme
optimises for the high-density region of the activation distribution,
choosing a scale that covers only **±250** — silently clipping over
60% of the real range.

Confirmed by comparing dequantized AIPU output against the FP32 PyTorch
reference on the same image, per tensor, per channel.

**Fix:** `quantization_scheme="per_tensor_min_max"` in `CompilerConfig`.
This guarantees the full observed range is representable, at the cost of
coarser resolution for typical values — the right tradeoff for a
pre-normalization layer with heavy-tailed activations.


### 3. Voyager compiler reorders multi-output graph outputs alphabetically

**Symptom:** After switching to `per_tensor_min_max` and confirming
correct heatmap values, detections were still wrong. Per-tensor statistics
showed the nine AIPU outputs did not match their expected PyTorch
counterparts by name — the channel blocks were scrambled.

**Cause:** Voyager's compiler sorts multi-output graph output nodes
**alphabetically by tensor name** during its graph canonicalization pass.
The ONNX export order is not preserved. This is not documented anywhere
and produces no warning.

Confirmed empirically by comparing min/max/mean statistics of each
dequantized AIPU output against the FP32 PyTorch reference for the same
image, then matching them by value rather than by index. The resulting
permutation was exactly alphabetical:
`center2kpt_offset, depth, dim, dir_feat, heatmap, kpt_heatmap, ...`

**Fix:** Fuse the nine HeadConv1 convs into a single `64 → 576` conv
(`HeadConv1Fused`). A single-output graph has nothing to reorder — the
compiler's alphabetical sorting is a non-issue. Verified numerically
identical to the original nine-conv version (max diff `0.00000000`).



### 4. ONNXRuntime dispatch overhead dominated head latency

**Symptom:** HeadTail (24,121 parameters, trivial arithmetic) measured
at **55–60 ms** under ONNXRuntime — far slower than the entire AIPU
backbone+neck+HeadConv1 graph.

**Cause:** The exported HeadTail ONNX graph contains approximately
**150 nodes** — the result of `AttnBatchNorm2d`'s `var_mean`, matmul,
inner BatchNorm, and HSigmoid each tracing to multiple small ops.
ORT dispatches each node individually through its general-purpose graph
executor, paying a fixed per-node overhead (~350 µs) regardless of how
little arithmetic that node performs.
`150 nodes × ~350 µs ≈ 52 ms` — matching the measurement exactly.
This is dispatch overhead, not compute.

**Fix:** Replace the ORT session with a hand-written `HeadTailCPU` class
in C++. All weights loaded once at construction. All scratch buffers
preallocated. Nine heads processed in parallel via OpenMP.
Zero heap allocation per frame. Result: **22 ms** — a **2.7× improvement**.

### 7. Closing the gap between sequential and pipelined throughput

**Observation:** After all per-stage optimisations, the sequential
pipeline measured **62 ms/frame (16 FPS)**. The AIPU and CPU host were
each idle roughly half the time — the AIPU waited while the CPU ran the
head, and the CPU waited while the AIPU ran inference.

**The opportunity:** The two bottleneck stages use completely independent
hardware:
- AIPU: `31 ms` (backbone + neck + HeadConv1)
- CPU: `29 ms` (dequantize + head + decode)

Since they don't share resources, they can run concurrently on different
frames.

**Fix:** A two-thread producer-consumer pipeline (`monocon.cpp`,
`process_video()`):

- **Thread A (AIPU thread):** preprocess → quantize → AIPU inference →
  copy raw INT8 output into a bounded queue (depth 3)
- **Thread B (CPU thread):** pop from queue → dequantize → HeadTailCPU
  → decode → store detections

The queue carries raw INT8 buffers (~1.9 MB each) rather than dequantized
floats (~7.6 MB), keeping memory pressure low and minimising copy cost.

With the two stages nearly matched in time, steady-state throughput
approaches `max(31 ms, 29 ms) ≈ 31 ms/frame`.

**Result:** **24.5 FPS** over 433 real KITTI frames — a **1.5×
improvement** over the already-optimised sequential baseline, from the
same hardware with no changes to any individual stage.

