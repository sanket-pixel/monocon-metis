#pragma once

#include "inference/aipu_inference.hpp"
#include <cstdint>
#include <vector>

namespace monocon {

    // Returns a freshly-allocated buffer — convenient, but allocates every call.
    std::vector<float> dequantize_padded_nhwc_to_nchw(
        const int8_t* raw_output, const AipuTensorInfo& tensor_info);

    // Writes into a pre-sized, caller-owned buffer — no allocation per call.
    // `out` must already be sized to tensor_info's logical element count.
    void dequantize_padded_nhwc_to_nchw_into(
        const int8_t* raw_output, const AipuTensorInfo& tensor_info, std::vector<float>& out);

} // namespace monocon