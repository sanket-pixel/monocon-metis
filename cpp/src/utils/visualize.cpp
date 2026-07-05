#include "utils/visualize.hpp"

namespace monocon {

    namespace {
        cv::Scalar class_color(int class_id) {
            switch (class_id) {
                case 0: return cv::Scalar(255, 0, 0);   // Pedestrian - RGB blue-ish placeholder
                case 1: return cv::Scalar(0, 255, 0);   // Cyclist
                default: return cv::Scalar(0, 0, 255);  // Car
            }
        }

        constexpr std::array<std::pair<int, int>, 12> BOX_EDGES = {{
            {0, 1}, {0, 3}, {0, 4}, {1, 2}, {1, 5}, {3, 2},
            {3, 7}, {4, 5}, {4, 7}, {2, 6}, {5, 6}, {6, 7}}};
    }

    cv::Mat draw_2d_boxes(const cv::Mat& rgb_image, const std::vector<Detection>& detections) {
        cv::Mat out = rgb_image.clone();
        for (const auto& det : detections) {
            cv::Point p1(static_cast<int>(det.box2d[0]), static_cast<int>(det.box2d[1]));
            cv::Point p2(static_cast<int>(det.box2d[2]), static_cast<int>(det.box2d[3]));
            cv::rectangle(out, p1, p2, class_color(det.class_id), 2, cv::LINE_AA);
        }
        return out;
    }

    cv::Mat draw_3d_boxes(const cv::Mat& rgb_image, const std::vector<Detection>& detections, const ProjMatrix& P2) {
        cv::Mat out = rgb_image.clone();
        for (const auto& det : detections) {
            auto corners3d = extract_corners_from_box3d(det.box3d);
            auto corners2d = project_corners_to_image(corners3d, P2);

            const cv::Scalar color = class_color(det.class_id);
            for (const auto& [a, b] : BOX_EDGES) {
                cv::Point pa(static_cast<int>(corners2d[a][0]), static_cast<int>(corners2d[a][1]));
                cv::Point pb(static_cast<int>(corners2d[b][0]), static_cast<int>(corners2d[b][1]));
                cv::line(out, pa, pb, color, 2, cv::LINE_AA);
            }
        }
        return out;
    }

} // namespace monocon