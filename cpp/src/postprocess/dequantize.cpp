#include "postprocess/dequantize.hpp"
#include <stdexcept>

namespace monocon {

    void dequantize_padded_nhwc_to_nchw_into(
        const int8_t* raw_output, const AipuTensorInfo& tensor_info, std::vector<float>& out) {

        if (tensor_info.shape.size() != 4) {
            throw std::runtime_error("dequantize: expected 4D NHWC tensor");
        }

        const int64_t N = tensor_info.shape[0];
        const int64_t H = tensor_info.shape[1];
        const int64_t W = tensor_info.shape[2];
        const int64_t C = tensor_info.shape[3];

        const int64_t padded_H = tensor_info.padded_shape[1];
        const int64_t padded_W = tensor_info.padded_shape[2];
        const int64_t padded_C = tensor_info.padded_shape[3];

        const int64_t pad_h_before = tensor_info.padding[1].first;
        const int64_t pad_w_before = tensor_info.padding[2].first;
        const int64_t pad_c_before = tensor_info.padding[3].first;

        const double scale = tensor_info.scale;
        const int zero_point = tensor_info.zero_point;

        // Only resize if needed — after the first call, this is a no-op,
        // so no repeated allocation once shapes stabilize.
        const size_t needed = static_cast<size_t>(N * C * H * W);
        if (out.size() != needed) out.resize(needed);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    for (int64_t c = 0; c < C; ++c) {
                        const int64_t src_h = h + pad_h_before;
                        const int64_t src_w = w + pad_w_before;
                        const int64_t src_c = c + pad_c_before;
                        const int64_t src_idx = ((n * padded_H + src_h) * padded_W + src_w) * padded_C + src_c;
                        const int64_t dst_idx = ((n * C + c) * H + h) * W + w;
                        out[dst_idx] = static_cast<float>((static_cast<double>(raw_output[src_idx]) - zero_point) * scale);
                    }
                }
            }
        }
    }

    std::vector<float> dequantize_padded_nhwc_to_nchw(
        const int8_t* raw_output, const AipuTensorInfo& tensor_info) {
        std::vector<float> out;
        dequantize_padded_nhwc_to_nchw_into(raw_output, tensor_info, out);
        return out;
    }

} // namespace monocon