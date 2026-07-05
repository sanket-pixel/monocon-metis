#include "preprocess/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace monocon {

    void quantize_nchw_to_padded_nhwc(
        const float* src_nchw,
        int channels, int height, int width,
        const AipuTensorInfo& tensor_info,
        int8_t* out_buffer) {

        // Expect tensor_info.shape / padding to describe NHWC: (N, H, W, C)
        if (tensor_info.shape.size() != 4) {
            throw std::runtime_error("quantize_nchw_to_padded_nhwc: expected 4D NHWC tensor info");
        }

        const int64_t logical_h = tensor_info.shape[1];
        const int64_t logical_w = tensor_info.shape[2];
        const int64_t logical_c = tensor_info.shape[3];

        if (logical_h != height || logical_w != width || logical_c != channels) {
            throw std::runtime_error(
                "quantize_nchw_to_padded_nhwc: input size (" +
                std::to_string(channels) + "," + std::to_string(height) + "," + std::to_string(width) +
                ") does not match model's expected logical shape (" +
                std::to_string(logical_c) + "," + std::to_string(logical_h) + "," + std::to_string(logical_w) + ")");
        }

        const int64_t pad_h_before = tensor_info.padding[1].first;
        const int64_t pad_w_before = tensor_info.padding[2].first;
        const int64_t pad_c_before = tensor_info.padding[3].first;

        const int64_t padded_h = tensor_info.padded_shape[1];
        const int64_t padded_w = tensor_info.padded_shape[2];
        const int64_t padded_c = tensor_info.padded_shape[3];

        const double scale = tensor_info.scale;
        const int zero_point = tensor_info.zero_point;

        // Zero the full padded buffer first — padded regions (both from our
        // own size-divisor-32 padding AND the hardware's own channel/spatial
        // alignment padding) should quantize to "zero_point" at zero input,
        // not garbage. Quantized zero is zero_point, not necessarily 0 —
        // fill accordingly rather than assuming memset(0) is correct.
        const int8_t quantized_zero = static_cast<int8_t>(
            std::clamp(zero_point, -128, 127));
        std::fill(out_buffer, out_buffer + (padded_h * padded_w * padded_c), quantized_zero);

        // Source is NCHW: src_nchw[c][y][x] = src_nchw[c * height * width + y * width + x]
        // Destination is NHWC + padding offset:
        //   dst[(y + pad_h_before) * padded_w * padded_c + (x + pad_w_before) * padded_c + (c + pad_c_before)]
        for (int c = 0; c < channels; ++c) {
            const float* channel_plane = src_nchw + static_cast<size_t>(c) * height * width;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const float value = channel_plane[y * width + x];

                    int quantized = static_cast<int>(std::lround(value / scale)) + zero_point;
                    quantized = std::clamp(quantized, -128, 127);

                    const int64_t dst_y = y + pad_h_before;
                    const int64_t dst_x = x + pad_w_before;
                    const int64_t dst_c = c + pad_c_before;

                    const int64_t dst_index =
                        dst_y * padded_w * padded_c + dst_x * padded_c + dst_c;

                    out_buffer[dst_index] = static_cast<int8_t>(quantized);
                }
            }
        }
    }

} // namespace monocon