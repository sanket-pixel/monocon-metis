#include "preprocess/fused_preprocess_quantize.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace monocon {

    namespace {
        // ImageNet normalisation constants — must match compile_monocon.py exactly.
        constexpr float MEAN_R = 123.675f, MEAN_G = 116.28f,  MEAN_B = 103.53f;
        constexpr float STD_R  = 58.395f,  STD_G  = 57.12f,   STD_B  = 57.375f;

        // Input dimensions are padded to this stride before AIPU inference.
        constexpr int SIZE_DIVISOR = 32;
    }

    PreprocessedDims preprocess_and_quantize(
        const cv::Mat&        image_rgb,
        const AipuTensorInfo& aipu_input_info,
        int8_t*               aipu_input_buffer) {

        if (image_rgb.empty()) {
            throw std::runtime_error("preprocess_and_quantize: input image is empty");
        }
        if (aipu_input_info.shape.size() != 4) {
            throw std::runtime_error("preprocess_and_quantize: expected 4D NHWC tensor info");
        }

        const int image_h = image_rgb.rows;
        const int image_w = image_rgb.cols;

        PreprocessedDims dims;
        dims.padded_height = static_cast<int>(
            std::ceil(static_cast<double>(image_h) / SIZE_DIVISOR)) * SIZE_DIVISOR;
        dims.padded_width  = static_cast<int>(
            std::ceil(static_cast<double>(image_w) / SIZE_DIVISOR)) * SIZE_DIVISOR;

        const int64_t model_h = aipu_input_info.shape[1];
        const int64_t model_w = aipu_input_info.shape[2];
        const int64_t model_c = aipu_input_info.shape[3];

        if (model_h != dims.padded_height || model_w != dims.padded_width || model_c != 3) {
            throw std::runtime_error(
                "preprocess_and_quantize: padded image size does not match model input shape");
        }

        // Hardware buffer dimensions (include AIPU alignment padding).
        const int64_t hw_padded_h = aipu_input_info.padded_shape[1];
        const int64_t hw_padded_w = aipu_input_info.padded_shape[2];
        const int64_t hw_padded_c = aipu_input_info.padded_shape[3];

        // Offset into the padded buffer where real data starts.
        const int64_t pad_h = aipu_input_info.padding[1].first;
        const int64_t pad_w = aipu_input_info.padding[2].first;
        const int64_t pad_c = aipu_input_info.padding[3].first;

        // Quantisation parameters from the compiled model manifest.
        const float   inv_scale  = 1.0f / static_cast<float>(aipu_input_info.scale);
        const int     zero_point = aipu_input_info.zero_point;
        const int8_t  pad_value  = static_cast<int8_t>(std::clamp(zero_point, -128, 127));

        // Fill entire buffer with the quantised zero value — covers all
        // hardware padding regions without a separate loop.
        std::fill(aipu_input_buffer,
                  aipu_input_buffer + hw_padded_h * hw_padded_w * hw_padded_c,
                  pad_value);

        const float channel_mean[3]    = {MEAN_R,        MEAN_G,        MEAN_B};
        const float channel_inv_std[3] = {1.0f / STD_R,  1.0f / STD_G,  1.0f / STD_B};

        // One parallel pass: normalize → quantize → write int8 into NHWC buffer.
        // Row parallelism gives good granularity on a Ryzen-class host CPU.
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < image_h; ++row) {
            const uint8_t* src_row = image_rgb.ptr<uint8_t>(row);
            const int64_t  dst_row = row + pad_h;

            for (int col = 0; col < image_w; ++col) {
                const uint8_t* src_pixel = src_row + col * 3;
                const int64_t  dst_base  =
                    dst_row * hw_padded_w * hw_padded_c +
                    (col + pad_w) * hw_padded_c +
                    pad_c;

                for (int channel = 0; channel < 3; ++channel) {
                    const float normalized = (static_cast<float>(src_pixel[channel])
                                              - channel_mean[channel])
                                             * channel_inv_std[channel];
                    const int quantized = std::clamp(
                        static_cast<int>(std::lround(normalized * inv_scale)) + zero_point,
                        -128, 127);
                    aipu_input_buffer[dst_base + channel] = static_cast<int8_t>(quantized);
                }
            }
        }

        return dims;
    }

} // namespace monocon