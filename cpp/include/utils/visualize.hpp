#pragma once

#include "postprocess/decode.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace monocon {

    cv::Mat draw_2d_boxes(const cv::Mat& rgb_image, const std::vector<Detection>& detections);
    cv::Mat draw_3d_boxes(const cv::Mat& rgb_image, const std::vector<Detection>& detections, const ProjMatrix& P2);

} // namespace monocon