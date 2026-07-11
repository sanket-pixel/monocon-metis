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

        const float scale = static_cast<float>(tensor_info.scale);
        const float neg_zp_scale = -static_cast<float>(tensor_info.zero_point) * scale;

        const size_t needed = static_cast<size_t>(N * C * H * W);
        if (out.size() != needed) out.resize(needed);

        float* const out_ptr = out.data();
        const int64_t HW = H * W;

        // Parallelize over rows — same pattern that worked for the 9
        // separate-tensor case, just applied within this one bigger call.
        // C is now 576, not 64, so each row does 9x more work than before —
        // still splits cleanly across cores.
        const int64_t BLOCK = 32;

        // Parallelize over rows
#pragma omp parallel for schedule(static)
        for (int64_t h = 0; h < H; ++h) {
            const int64_t src_h = h + pad_h_before;

            // Tile over Channels
            for (int64_t c_blk = 0; c_blk < C; c_blk += BLOCK) {
                const int64_t c_end = std::min(c_blk + BLOCK, C);

                // Tile over Width
                for (int64_t w_blk = 0; w_blk < W; w_blk += BLOCK) {
                    const int64_t w_end = std::min(w_blk + BLOCK, W);

                    // Inside this block, everything is cached!
                    for (int64_t c = c_blk; c < c_end; ++c) {
                        float* const channel_out_row = out_ptr + c * HW + h * W;
                        const int64_t src_c = c + pad_c_before;

                        #pragma omp simd
                        for (int64_t w = w_blk; w < w_end; ++w) {
                            const int64_t src_w = w + pad_w_before;
                            const int8_t src_val = raw_output[(src_h * padded_W + src_w) * padded_C + src_c];

                            // Contiguous write. The strided read is now safe because
                            // the surrounding data is still living in the L1 cache!
                            channel_out_row[w] = static_cast<float>(src_val) * scale + neg_zp_scale;
                        }
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