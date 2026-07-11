#pragma once

#include "inference/aipu_inference.hpp"

#include <opencv2/opencv.hpp>
#include <cstdint>

namespace monocon {

    // Dimensions of the image after padding to a multiple of SIZE_DIVISOR.
    struct PreprocessedDims {
        int padded_height = 0;
        int padded_width  = 0;
    };

    // Fuses normalize + quantize into a single pixel pass, writing directly
    // into the AIPU's int8 input buffer (NHWC, hardware-padded layout).
    //
    // Expects image_rgb to be already loaded and in RGB order (not BGR).
    // Image I/O belongs in the caller — keeping it out of this function
    // ensures the hot path stays free of disk latency in the video pipeline.
    PreprocessedDims preprocess_and_quantize(
        const cv::Mat&        image_rgb,
        const AipuTensorInfo& aipu_input_info,
        int8_t*               aipu_input_buffer);

} // namespace monocon