#include "preprocess/opencv_preprocess.hpp"

#include <stdexcept>

namespace monocon {

    PreprocessedImage preprocess_image(const std::string& image_path) {
        cv::Mat img_bgr = cv::imread(image_path, cv::IMREAD_COLOR);
        if (img_bgr.empty()) {
            throw std::runtime_error("preprocess_image: failed to read '" + image_path + "'");
        }

        cv::Mat img_rgb;
        cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

        PreprocessedImage result;
        result.ori_height = img_rgb.rows;
        result.ori_width = img_rgb.cols;
        result.ori_img = img_rgb.clone();  // kept for visualization, unnormalized

        // Padded target size — multiple of SIZE_DIVISOR, bottom/right only
        const int padded_h = static_cast<int>(std::ceil(static_cast<double>(img_rgb.rows) / SIZE_DIVISOR)) * SIZE_DIVISOR;
        const int padded_w = static_cast<int>(std::ceil(static_cast<double>(img_rgb.cols) / SIZE_DIVISOR)) * SIZE_DIVISOR;

        result.height = padded_h;
        result.width = padded_w;
        result.channels = 3;

        // NCHW float32 buffer, zero-initialized (covers the padded region for free)
        result.data.assign(static_cast<size_t>(3) * padded_h * padded_w, 0.0f);

        const float mean[3] = {MEAN_R, MEAN_G, MEAN_B};
        const float std_[3] = {STD_R, STD_G, STD_B};

        // Normalize + write into the top-left region of the padded NCHW buffer.
        // img_rgb is HWC (OpenCV convention); we scatter into CHW manually.
        for (int c = 0; c < 3; ++c) {
            float* channel_plane = result.data.data() + static_cast<size_t>(c) * padded_h * padded_w;
            for (int y = 0; y < img_rgb.rows; ++y) {
                const uint8_t* row_ptr = img_rgb.ptr<uint8_t>(y);
                for (int x = 0; x < img_rgb.cols; ++x) {
                    const float pixel = static_cast<float>(row_ptr[x * 3 + c]);
                    channel_plane[y * padded_w + x] = (pixel - mean[c]) / std_[c];
                }
            }
        }

        return result;
    }

} // namespace monocon