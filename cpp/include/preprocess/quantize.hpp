#pragma once

#include "inference/aipu_inference.hpp"

#include <cstdint>
#include <vector>

namespace monocon {

    // Converts a float32 NCHW buffer into the int8, hardware-padded, NHWC
    // layout the AIPU expects — using the REAL scale/zero_point/padding
    // read from the compiled model's axrTensorInfo (via AipuTensorInfo),
    // never hardcoded constants. This is what keeps calibration and
    // inference numerically consistent with what Voyager actually compiled.
    //
    // int8_value = round(float_value / scale) + zero_point, clamped to [-128, 127]
    //
    // Writes directly into `out_buffer`, which must already be sized to
    // `tensor_info.size_bytes` (this is exactly what
    // AipuInference::input_buffer() returns space for).
    void quantize_nchw_to_padded_nhwc(
        const float* src_nchw,   // (C, H, W) float32, logical (unpadded) size
        int channels, int height, int width,
        const AipuTensorInfo& tensor_info,
        int8_t* out_buffer);

} // namespace monocon