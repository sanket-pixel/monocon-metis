#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace monocon {

    // Mirrors python/model/preprocess.py exactly:
    // load -> RGB -> normalize -> pad to multiple of 32 (bottom/right only).
    // No resizing anywhere in this pipeline, so no calibration rescale needed.
    struct PreprocessedImage {
        std::vector<float> data;   // NCHW, float32, normalized + padded
        int channels = 3;
        int height = 0;            // padded height
        int width = 0;             // padded width

        cv::Mat ori_img;           // original (unpadded, unnormalized) RGB image, for visualization
        int ori_height = 0;
        int ori_width = 0;
    };

    // Must match python/model/preprocess.py's MEAN/STD/SIZE_DIVISOR exactly.
    constexpr float MEAN_R = 123.675f, MEAN_G = 116.28f, MEAN_B = 103.53f;
    constexpr float STD_R = 58.395f, STD_G = 57.12f, STD_B = 57.375f;
    constexpr int SIZE_DIVISOR = 32;

    PreprocessedImage preprocess_image(const std::string& image_path);

} // namespace monocon