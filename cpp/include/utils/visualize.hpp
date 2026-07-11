#pragma once

#include "postprocess/decode.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace monocon {

    cv::Mat draw_2d_boxes(const cv::Mat& rgb_image, const std::vector<Detection>& detections);
    cv::Mat draw_3d_boxes(const cv::Mat& rgb_image, const std::vector<Detection>& detections, const ProjMatrix& P2);
    cv::Mat draw_bev(const std::vector<Detection>& detections,
                 int canvas_w = 400, int canvas_h = 600,
                 float meters_per_pixel = 0.1f);

} // namespace monocon