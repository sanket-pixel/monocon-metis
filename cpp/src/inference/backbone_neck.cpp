#include "inference/backbone_neck.hpp"
#include "utils/timer.hpp"

namespace monocon {

    BackboneNeck::BackboneNeck(const std::string& compiled_model_dir, int num_cores)
        : aipu_(compiled_model_dir, num_cores) {}

    PreprocessedDims BackboneNeck::forward(const cv::Mat& image_rgb) {
        const AipuTensorInfo& input_info = aipu_.input_info(0);
        PreprocessedDims preprocess_result =
            preprocess_and_quantize(image_rgb, input_info, aipu_.input_buffer(0));
        aipu_.run();
        return preprocess_result;
    }

    PreprocessedDims BackboneNeck::forward_timed(const cv::Mat& image_rgb,
                                                      double& out_preprocess_ms,
                                                      double& out_aipu_ms) {
        const AipuTensorInfo& input_info = aipu_.input_info(0);

        Timer timer;
        PreprocessedDims preprocess_result =
            preprocess_and_quantize(image_rgb, input_info, aipu_.input_buffer(0));
        out_preprocess_ms = timer.elapsed_ms();

        timer.reset();
        aipu_.run();
        out_aipu_ms = timer.elapsed_ms();

        return preprocess_result;
    }

    void BackboneNeck::forward_aipu_only(const cv::Mat& image_rgb,
                                         double& out_preprocess_ms,
                                         double& out_aipu_ms) {
        const AipuTensorInfo& input_info = aipu_.input_info(0);

        Timer timer;
        preprocess_and_quantize(image_rgb, input_info, aipu_.input_buffer(0));
        out_preprocess_ms = timer.elapsed_ms();

        timer.reset();
        aipu_.run();
        out_aipu_ms = timer.elapsed_ms();

        // Raw int8 output now sits in aipu_.output_buffer(0).
        // Valid until the next call to forward_aipu_only() — caller must
        // copy it into a QueuedFrame immediately (see monocon.cpp).
    }

} // namespace monocon