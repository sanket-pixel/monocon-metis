#include "inference/backbone_neck.hpp"
#include "preprocess/quantize.hpp"

namespace monocon {

    BackboneNeck::BackboneNeck(const std::string& compiled_model_dir, int num_cores)
        : aipu_(compiled_model_dir, num_cores) {}

    const int8_t* BackboneNeck::forward(const PreprocessedImage& image) {
        const AipuTensorInfo& in_info = aipu_.input_info(0);

        quantize_nchw_to_padded_nhwc(
            image.data.data(),
            image.channels, image.height, image.width,
            in_info,
            aipu_.input_buffer(0));

        // In backbone_neck.cpp's forward(), or temporarily in main.cpp:
        const int8_t* input_buf = aipu_.input_buffer(0);
        std::map<int8_t, int> input_histogram;
        for (size_t i = 0; i < in_info.size_bytes; ++i) {
            input_histogram[input_buf[i]]++;
        }

        return aipu_.run();
    }

} // namespace monocon