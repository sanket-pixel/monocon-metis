#pragma once

#include "inference/aipu_inference.hpp"
#include "preprocess/fused_preprocess_quantize.h"

#include <string>

namespace monocon {

    // Wraps AipuInference with MonoCon-specific preprocessing.
    // Fuses BGR→RGB conversion, normalization, quantization, and NHWC
    // padding into a single pass directly into the AIPU's input buffer.
    //
    // Three forward variants serve different call sites:
    //   forward()           — single-image path, returns padded dimensions
    //   forward_timed()     — same but fills per-stage timing for profiling
    //   forward_aipu_only() — producer thread in the pipelined video path;
    //                         caller must copy output_buffer() before the
    //                         next call overwrites it
    class BackboneNeck {
    public:
        explicit BackboneNeck(const std::string& compiled_model_dir, int num_cores = 1);

        PreprocessedDims forward(const cv::Mat& image_rgb);

        PreprocessedDims forward_timed(const cv::Mat& image_rgb,
                                            double& out_preprocess_ms,
                                            double& out_aipu_ms);

        void forward_aipu_only(const cv::Mat& image_rgb,
                               double& out_preprocess_ms,
                               double& out_aipu_ms);

        int                   output_count()            const { return aipu_.output_count(); }
        const AipuTensorInfo& output_info(int index)    const { return aipu_.output_info(index); }
        const int8_t*         output_buffer(int index)  const { return aipu_.output_buffer(index); }
        const AipuTensorInfo& input_info()              const { return aipu_.input_info(0); }

    private:
        AipuInference aipu_;
    };

} // namespace monocon