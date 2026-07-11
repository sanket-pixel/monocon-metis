#include "utils/visualize.hpp"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>

namespace monocon {

    namespace {

        // Class colors in BGR
        const cv::Scalar CLASS_COLORS_BGR[3] = {
            cv::Scalar(255, 100, 100),  // Pedestrian — soft blue
            cv::Scalar(100, 255, 100),  // Cyclist    — soft green
            cv::Scalar(100, 180, 255),  // Car        — amber/gold
        };

        const char* CLASS_NAMES[3] = {"Pedestrian", "Cyclist", "Car"};

        cv::Scalar class_color_bgr(int class_id) {
            if (class_id < 0 || class_id > 2) return cv::Scalar(200, 200, 200);
            return CLASS_COLORS_BGR[class_id];
        }

        cv::Scalar class_color_rgb(int class_id) {
            const cv::Scalar bgr = class_color_bgr(class_id);
            return cv::Scalar(bgr[2], bgr[1], bgr[0]);
        }

        constexpr std::array<std::pair<int, int>, 12> BOX_EDGES = {{
            {0, 1}, {0, 3}, {0, 4}, {1, 2}, {1, 5}, {3, 2},
            {3, 7}, {4, 5}, {4, 7}, {2, 6}, {5, 6}, {6, 7}}};

        // Bottom face edges — drawn thicker to ground the box visually
        constexpr std::array<std::pair<int, int>, 4> BOTTOM_EDGES = {{
            {0, 1}, {1, 2}, {2, 3}, {3, 0}}};

    } // namespace

    // -----------------------------------------------------------------------
    // Camera view — 3D boxes with filled front face + score label
    // -----------------------------------------------------------------------

    cv::Mat draw_3d_boxes(const cv::Mat& rgb_image,
                          const std::vector<Detection>& detections,
                          const ProjMatrix& P2) {
        cv::Mat out = rgb_image.clone();

        for (const auto& det : detections) {
            const cv::Scalar color = class_color_rgb(det.class_id);
            const cv::Scalar color_dim(color[0] * 0.35, color[1] * 0.35, color[2] * 0.35);

            auto corners3d = extract_corners_from_box3d(det.box3d);
            auto corners2d = project_corners_to_image(corners3d, P2);

            // Convert to cv::Point
            std::array<cv::Point, 8> pts;
            for (int i = 0; i < 8; ++i) {
                pts[i] = cv::Point(static_cast<int>(corners2d[i][0]),
                                   static_cast<int>(corners2d[i][1]));
            }

            // Filled front face (corners 0,1,2,3 — front of box)
            // with transparency overlay
            cv::Mat overlay = out.clone();
            const std::array<cv::Point, 4> front_face = {pts[0], pts[1], pts[2], pts[3]};
            cv::fillConvexPoly(overlay,
                               front_face.data(), 4,
                               color_dim, cv::LINE_AA);
            cv::addWeighted(overlay, 0.35, out, 0.65, 0, out);

            // All edges — back edges slightly dimmer
            for (const auto& [a, b] : BOX_EDGES) {
                // Back edges: indices 4-7 are rear
                const bool is_back = (a >= 4 && b >= 4);
                cv::Scalar edge_color = is_back
                    ? cv::Scalar(color[0]*0.5, color[1]*0.5, color[2]*0.5)
                    : color;
                const int thickness = is_back ? 1 : 2;
                cv::line(out, pts[a], pts[b], edge_color, thickness, cv::LINE_AA);
            }

            // Score + class label above the top-left corner
            const cv::Point label_pt(
                pts[0].x, std::max(pts[0].y - 8, 14));

            const std::string label = std::string(CLASS_NAMES[det.class_id]);

            // Label background pill
            int baseline = 0;
            const cv::Size text_size = cv::getTextSize(
                label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
            cv::rectangle(out,
                          cv::Point(label_pt.x - 2, label_pt.y - text_size.height - 3),
                          cv::Point(label_pt.x + text_size.width + 2, label_pt.y + 2),
                          color_dim * 3, -1, cv::LINE_AA);
            cv::putText(out, label, label_pt,
                        cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }

        return out;
    }

    // -----------------------------------------------------------------------
    // BEV view
    // -----------------------------------------------------------------------

    cv::Mat draw_bev(const std::vector<Detection>& detections,
                     int canvas_w, int canvas_h,
                     float meters_per_pixel) {

        // Dark background with subtle gradient
        cv::Mat bev(canvas_h, canvas_w, CV_8UC3, cv::Scalar(20, 20, 25));

        const float origin_x = canvas_w * 0.5f;
        const float origin_z = canvas_h * 0.92f;

        // --- Grid ---
        for (int depth_m = 10; depth_m <= 80; depth_m += 10) {
            const int grid_y = static_cast<int>(
                origin_z - depth_m / meters_per_pixel);
            if (grid_y < 0 || grid_y >= canvas_h) continue;

            // Subtle dashed grid line
            for (int x = 0; x < canvas_w; x += 12) {
                cv::line(bev,
                         cv::Point(x, grid_y),
                         cv::Point(std::min(x + 7, canvas_w - 1), grid_y),
                         cv::Scalar(50, 50, 60), 1);
            }
            cv::putText(bev,
                        std::to_string(depth_m) + "m",
                        cv::Point(4, grid_y - 4),
                        cv::FONT_HERSHEY_PLAIN, 0.75,
                        cv::Scalar(80, 80, 90), 1, cv::LINE_AA);
        }

        // Vertical center line (ego lane)
        cv::line(bev,
                 cv::Point(static_cast<int>(origin_x), 0),
                 cv::Point(static_cast<int>(origin_x), canvas_h),
                 cv::Scalar(40, 40, 50), 1, cv::LINE_AA);

        // --- Ego vehicle ---
        const int ego_cx = static_cast<int>(origin_x);
        const int ego_cy = static_cast<int>(origin_z);

        // Simple car silhouette as rounded rectangle
        cv::rectangle(bev,
                      cv::Point(ego_cx - 8, ego_cy - 14),
                      cv::Point(ego_cx + 8, ego_cy + 6),
                      cv::Scalar(220, 220, 220), -1, cv::LINE_AA);
        cv::rectangle(bev,
                      cv::Point(ego_cx - 5, ego_cy - 20),
                      cv::Point(ego_cx + 5, ego_cy - 13),
                      cv::Scalar(180, 180, 180), -1, cv::LINE_AA);

        // --- Detections ---
        for (const auto& det : detections) {
            const Box3D& box = det.box3d;

            const float bev_cx = origin_x + box.x / meters_per_pixel;
            const float bev_cy = origin_z - box.z / meters_per_pixel;

            if (bev_cx < 0 || bev_cx >= canvas_w ||
                bev_cy < 0 || bev_cy >= canvas_h) continue;

            const cv::Scalar color = class_color_bgr(det.class_id);
            const cv::Scalar color_fill(
                color[0] * 0.25, color[1] * 0.25, color[2] * 0.25);

            const float half_l = box.length / (2.0f * meters_per_pixel);
            const float half_w = box.width  / (2.0f * meters_per_pixel);
            const float cos_r  = std::cos(-box.rotation_y);
            const float sin_r  = std::sin(-box.rotation_y);

            const std::array<std::pair<float,float>, 4> local = {{
                { half_l,  half_w},
                {-half_l,  half_w},
                {-half_l, -half_w},
                { half_l, -half_w}}};

            std::array<cv::Point, 4> corners;
            for (int c = 0; c < 4; ++c) {
                corners[c] = cv::Point(
                    static_cast<int>(bev_cx + cos_r * local[c].first
                                            - sin_r * local[c].second),
                    static_cast<int>(bev_cy + sin_r * local[c].first
                                            + cos_r * local[c].second));
            }

            // Filled box
            cv::fillConvexPoly(bev, corners.data(), 4, color_fill, cv::LINE_AA);

            // Outline
            for (int c = 0; c < 4; ++c) {
                cv::line(bev, corners[c], corners[(c + 1) % 4],
                         color, 2, cv::LINE_AA);
            }

            // Heading arrow from center toward front
            const cv::Point center_pt(static_cast<int>(bev_cx),
                                      static_cast<int>(bev_cy));
            const cv::Point front_pt(
                static_cast<int>(bev_cx + cos_r * half_l * 1.4f),
                static_cast<int>(bev_cy + sin_r * half_l * 1.4f));
            cv::arrowedLine(bev, center_pt, front_pt,
                            cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0, 0.4);

            // Depth confidence dot — brighter = more confident
            const int dot_r = std::clamp(static_cast<int>(det.score * 5), 2, 5);
            cv::circle(bev, center_pt, dot_r, color, -1, cv::LINE_AA);
        }

        // --- Title overlay ---
        cv::putText(bev, "Bird's Eye View",
                    cv::Point(8, canvas_h - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(100, 100, 110), 1, cv::LINE_AA);

        return bev;
    }

    // -----------------------------------------------------------------------
    // 2D box fallback
    // -----------------------------------------------------------------------

    cv::Mat draw_2d_boxes(const cv::Mat& rgb_image,
                          const std::vector<Detection>& detections) {
        cv::Mat out = rgb_image.clone();
        for (const auto& det : detections) {
            cv::rectangle(out,
                          cv::Point(static_cast<int>(det.box2d[0]),
                                    static_cast<int>(det.box2d[1])),
                          cv::Point(static_cast<int>(det.box2d[2]),
                                    static_cast<int>(det.box2d[3])),
                          class_color_rgb(det.class_id), 2, cv::LINE_AA);
        }
        return out;
    }

} // namespace monocon