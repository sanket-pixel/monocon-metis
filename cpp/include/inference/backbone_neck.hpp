#pragma once

#include "inference/aipu_inference.hpp"
#include "preprocess/opencv_preprocess.hpp"

#include <string>

namespace monocon {

    // Thin, model-specific wrapper around AipuInference: quantizes a
    // preprocessed image into the AIPU's input buffer and runs inference.
    // After forward()/forward_timed() returns, read results via
    // output_buffer(i) for i in [0, output_count()) — positional, in the
    // compiled model's declared output order.
    class BackboneNeck {
    public:
        explicit BackboneNeck(const std::string& compiled_model_dir, int num_cores = 1);

        void forward(const PreprocessedImage& image);
        void forward_timed(const PreprocessedImage& image, double& quantize_ms, double& aipu_ms);

        int output_count() const { return aipu_.output_count(); }
        const AipuTensorInfo& output_info(int index) const { return aipu_.output_info(index); }
        const int8_t* output_buffer(int index) const { return aipu_.output_buffer(index); }
        const AipuTensorInfo& input_info() const { return aipu_.input_info(0); }

    private:
        AipuInference aipu_;
    };

} // namespace monocon