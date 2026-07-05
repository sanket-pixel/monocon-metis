#pragma once

#include "inference/aipu_inference.hpp"
#include "preprocess/opencv_preprocess.hpp"

#include <memory>
#include <string>

namespace monocon {

    // Thin, model-specific wrapper around AipuInference — this is the
    // clean entry point the rest of the pipeline calls: hand it a
    // preprocessed float image, get back the raw int8 AIPU output +
    // the tensor info needed to dequantize it (done in postprocess/, not here).
    class BackboneNeck {
    public:
        explicit BackboneNeck(const std::string& compiled_model_dir, int num_cores = 4);

        // Runs quantize -> AIPU inference. Returns a pointer to the raw
        // int8 output buffer (owned internally, valid until next call)
        // plus the tensor info needed to dequantize it.
        const int8_t* forward(const PreprocessedImage& image);

        const AipuTensorInfo& output_info() const { return aipu_.output_info(0); }
        const AipuTensorInfo& input_info() const { return aipu_.input_info(0); }

    private:
        AipuInference aipu_;
    };

} // namespace monocon