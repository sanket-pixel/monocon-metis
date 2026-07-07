#include "inference/backbone_neck.hpp"
#include "preprocess/quantize.hpp"
#include "utils/timer.hpp"

namespace monocon {

    BackboneNeck::BackboneNeck(const std::string& compiled_model_dir, int num_cores)
        : aipu_(compiled_model_dir, num_cores) {}

    void BackboneNeck::forward(const PreprocessedImage& image) {
        const AipuTensorInfo& in_info = aipu_.input_info(0);

        quantize_nchw_to_padded_nhwc(
            image.data.data(),
            image.channels, image.height, image.width,
            in_info,
            aipu_.input_buffer(0));

        aipu_.run();
    }

    void BackboneNeck::forward_timed(
        const PreprocessedImage& image, double& quantize_ms, double& aipu_ms) {

        const AipuTensorInfo& in_info = aipu_.input_info(0);

        Timer t;
        quantize_nchw_to_padded_nhwc(
            image.data.data(),
            image.channels, image.height, image.width,
            in_info,
            aipu_.input_buffer(0));
        quantize_ms = t.elapsed_ms();

        t.reset();
        aipu_.run();
        aipu_ms = t.elapsed_ms();
    }

} // namespace monocon