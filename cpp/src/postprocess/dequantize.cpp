#include "postprocess/dequantize.hpp"

#include <stdexcept>

namespace monocon {

    std::vector<float> dequantize_padded_nhwc_to_nchw(
        const int8_t* raw_output,
        const AipuTensorInfo& tensor_info) {

        if (tensor_info.shape.size() != 4) {
            throw std::runtime_error(
                "dequantize_padded_nhwc_to_nchw: expected 4D NHWC tensor, got " +
                std::to_string(tensor_info.shape.size()) + " dims");
        }

        const int64_t N = tensor_info.shape[0];
        const int64_t H = tensor_info.shape[1];
        const int64_t W = tensor_info.shape[2];
        const int64_t C = tensor_info.shape[3];

        if (N <= 0 || H <= 0 || W <= 0 || C <= 0) {
            throw std::runtime_error("dequantize_padded_nhwc_to_nchw: invalid logical shape");
        }

        const int64_t padded_H = tensor_info.padded_shape[1];
        const int64_t padded_W = tensor_info.padded_shape[2];
        const int64_t padded_C = tensor_info.padded_shape[3];

        const int64_t pad_h_before = tensor_info.padding[1].first;
        const int64_t pad_w_before = tensor_info.padding[2].first;
        const int64_t pad_c_before = tensor_info.padding[3].first;

        const double scale = tensor_info.scale;
        const int zero_point = tensor_info.zero_point;

        std::vector<float> out(static_cast<size_t>(N * C * H * W));

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    for (int64_t c = 0; c < C; ++c) {
                        const int64_t src_h = h + pad_h_before;
                        const int64_t src_w = w + pad_w_before;
                        const int64_t src_c = c + pad_c_before;

                        const int64_t src_idx =
                            ((n * padded_H + src_h) * padded_W + src_w) * padded_C + src_c;

                        const int64_t dst_idx = ((n * C + c) * H + h) * W + w;

                        out[dst_idx] = static_cast<float>(
                            (static_cast<double>(raw_output[src_idx]) - zero_point) * scale);
                    }
                }
            }
        }

        return out;
    }

} // namespace monocon