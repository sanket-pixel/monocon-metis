#pragma once

#include "inference/aipu_inference.hpp"

#include <cstdint>
#include <vector>

namespace monocon {

    // Converts the AIPU's raw int8, hardware-padded, NHWC output back into
    // a float32, logical (unpadded), NCHW buffer — matching exactly what
    // python/model/neck.py's forward() produces, so it can be fed straight
    // into the ONNXRuntime head (monocon_head.onnx expects NCHW).
    //
    // float_value = (int8_value - zero_point) * scale
    std::vector<float> dequantize_padded_nhwc_to_nchw(
        const int8_t* raw_output,
        const AipuTensorInfo& tensor_info);

} // namespace monocon