#include "postprocess/decode.hpp"

#include <algorithm>
#include <cmath>

namespace monocon {

    namespace {
        constexpr int TOPK = 30;
        constexpr int NUM_ALPHA_BINS = 12;
        constexpr int NUM_KPTS = 9;
        constexpr float PI = 3.14159265358979323846f;

        struct TopKResult {
            std::vector<float> scores;
            std::vector<int> pixel_idx;
            std::vector<int> classes;
            std::vector<int> ys, xs;
        };

        // 3x3 local-max NMS on a single-channel plane within a (C,H,W) buffer.
        std::vector<float> local_maximum(const std::vector<float>& heat, int C, int H, int W) {
            std::vector<float> out(heat.size(), 0.0f);
            for (int c = 0; c < C; ++c) {
                const float* plane = heat.data() + c * H * W;
                float* out_plane = out.data() + c * H * W;
                for (int y = 0; y < H; ++y) {
                    for (int x = 0; x < W; ++x) {
                        const float v = plane[y * W + x];
                        bool is_max = true;
                        for (int dy = -1; dy <= 1 && is_max; ++dy) {
                            for (int dx = -1; dx <= 1; ++dx) {
                                const int ny = y + dy, nx = x + dx;
                                if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
                                if (plane[ny * W + nx] > v) { is_max = false; break; }
                            }
                        }
                        out_plane[y * W + x] = is_max ? v : 0.0f;
                    }
                }
            }
            return out;
        }

        TopKResult get_topk(const std::vector<float>& heat, int C, int H, int W, int k) {
            std::vector<std::pair<float, int>> scored;
            scored.reserve(C * H * W);
            for (int i = 0; i < C * H * W; ++i) scored.emplace_back(heat[i], i);

            std::partial_sort(scored.begin(), scored.begin() + std::min<size_t>(k, scored.size()),
                              scored.end(), std::greater<>());

            TopKResult result;
            for (int i = 0; i < k && i < static_cast<int>(scored.size()); ++i) {
                const auto [score, flat_idx] = scored[i];
                const int cls = flat_idx / (H * W);
                const int pixel_idx = flat_idx % (H * W);
                const int y = pixel_idx / W;
                const int x = pixel_idx % W;

                result.scores.push_back(score);
                result.pixel_idx.push_back(pixel_idx);
                result.classes.push_back(cls);
                result.ys.push_back(y);
                result.xs.push_back(x);
            }
            return result;
        }

        // Gathers a C-channel vector at a given (H*W) pixel index from an (C,H,W) buffer.
        std::vector<float> gather_channels(const std::vector<float>& feat, int C, int HW, int pixel_idx) {
            std::vector<float> out(C);
            for (int c = 0; c < C; ++c) out[c] = feat[c * HW + pixel_idx];
            return out;
        }

        float decode_alpha(const std::vector<float>& alpha_cls, const std::vector<float>& alpha_offset) {
            int best_bin = 0;
            float best_val = alpha_cls[0];
            for (int i = 1; i < NUM_ALPHA_BINS; ++i) {
                if (alpha_cls[i] > best_val) { best_val = alpha_cls[i]; best_bin = i; }
            }

            const float angle_per_bin = (2.0f * PI) / NUM_ALPHA_BINS;
            float alpha = best_bin * angle_per_bin + alpha_offset[best_bin];

            if (alpha > PI) alpha -= 2 * PI;
            if (alpha < -PI) alpha += 2 * PI;
            return alpha;
        }

        float alpha_to_rotation_y(float image_x, float alpha, const ProjMatrix& P2) {
            const float focal_x = P2[0];
            const float principal_x = P2[2];

            float rot_y = alpha + std::atan2(image_x - principal_x, focal_x);
            if (rot_y > PI) rot_y -= 2 * PI;
            if (rot_y < -PI) rot_y += 2 * PI;
            return rot_y;
        }

        // Un-projects (image_x, image_y, depth) into camera-space 3D via inverse(P2 padded to 4x4).
        std::array<float, 3> image_to_camera_3d(float img_x, float img_y, float depth, const ProjMatrix& P2) {
            float M[4][4] = {
                {P2[0], P2[1], P2[2], P2[3]},
                {P2[4], P2[5], P2[6], P2[7]},
                {P2[8], P2[9], P2[10], P2[11]},
                {0, 0, 0, 1}};

            float aug[4][8];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) aug[i][j] = M[i][j];
                for (int j = 0; j < 4; ++j) aug[i][4 + j] = (i == j) ? 1.0f : 0.0f;
            }

            for (int col = 0; col < 4; ++col) {
                int pivot = col;
                for (int row = col + 1; row < 4; ++row) {
                    if (std::fabs(aug[row][col]) > std::fabs(aug[pivot][col])) pivot = row;
                }
                std::swap(aug[col], aug[pivot]);

                const float pivot_val = aug[col][col];
                for (int j = 0; j < 8; ++j) aug[col][j] /= pivot_val;

                for (int row = 0; row < 4; ++row) {
                    if (row == col) continue;
                    const float factor = aug[row][col];
                    for (int j = 0; j < 8; ++j) aug[row][j] -= factor * aug[col][j];
                }
            }

            float inv[4][4];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    inv[i][j] = aug[i][4 + j];

            // point_2d_scaled = (img_x*depth, img_y*depth, depth, 1)
            const float p[4] = {img_x * depth, img_y * depth, depth, 1.0f};

            // Standard matrix-vector multiply: out = inv * p  (column vector convention)
            float out3[3];
            for (int i = 0; i < 3; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < 4; ++j) sum += inv[i][j] * p[j];
                out3[i] = sum;
            }

            return {out3[0], out3[1], out3[2]};
        }
    } // anonymous

    std::vector<Detection> decode_predictions(
        const std::map<std::string, HeadOutput>& pred,
        const ProjMatrix& P2,
        int img_h, int img_w,
        float score_thres) {

        const auto& heatmap_out = pred.at("center_heatmap");
        const int num_classes = static_cast<int>(heatmap_out.shape[1]);
        const int feat_h = static_cast<int>(heatmap_out.shape[2]);
        const int feat_w = static_cast<int>(heatmap_out.shape[3]);
        const int HW = feat_h * feat_w;

        auto heat_nms = local_maximum(heatmap_out.data, num_classes, feat_h, feat_w);
        auto topk = get_topk(heat_nms, num_classes, feat_h, feat_w, TOPK);

        const auto& wh = pred.at("wh").data;
        const auto& offset = pred.at("offset").data;
        const auto& alpha_cls_all = pred.at("alpha_cls").data;
        const auto& alpha_offset_all = pred.at("alpha_offset").data;
        const auto& depth_all = pred.at("depth").data;
        const auto& kpt_offset_all = pred.at("center2kpt_offset").data;
        const auto& dim_all = pred.at("dim").data;

        const float x_scale = static_cast<float>(img_w) / feat_w;
        const float y_scale = static_cast<float>(img_h) / feat_h;

        std::vector<Detection> detections;

        for (int k = 0; k < static_cast<int>(topk.scores.size()); ++k) {
            const int pixel_idx = topk.pixel_idx[k];
            const int y = topk.ys[k], x = topk.xs[k];

            auto wh_k = gather_channels(wh, 2, HW, pixel_idx);
            auto offset_k = gather_channels(offset, 2, HW, pixel_idx);

            const float cx = x + offset_k[0];
            const float cy = y + offset_k[1];

            const float x1 = (cx - wh_k[0] / 2.f) * x_scale;
            const float y1 = (cy - wh_k[1] / 2.f) * y_scale;
            const float x2 = (cx + wh_k[0] / 2.f) * x_scale;
            const float y2 = (cy + wh_k[1] / 2.f) * y_scale;

            auto alpha_cls_k = gather_channels(alpha_cls_all, NUM_ALPHA_BINS, HW, pixel_idx);
            auto alpha_offset_k = gather_channels(alpha_offset_all, NUM_ALPHA_BINS, HW, pixel_idx);
            const float alpha = decode_alpha(alpha_cls_k, alpha_offset_k);

            auto depth_k = gather_channels(depth_all, 2, HW, pixel_idx);
            const float depth_val = depth_k[0];
            const float depth_confidence = std::exp(-depth_k[1]);

            float score = topk.scores[k] * depth_confidence;
            if (score <= score_thres) continue;

            auto kpt_offset_k = gather_channels(kpt_offset_all, NUM_KPTS * 2, HW, pixel_idx);
            // last keypoint pair = projected 3D center, per decode.py
            const float center_x = (kpt_offset_k[(NUM_KPTS - 1) * 2] + x) * x_scale;
            const float center_y = (kpt_offset_k[(NUM_KPTS - 1) * 2 + 1] + y) * y_scale;

            const float rot_y = alpha_to_rotation_y(center_x, alpha, P2);
            const auto center3d = image_to_camera_3d(center_x, center_y, depth_val, P2);

            auto dim_k = gather_channels(dim_all, 3, HW, pixel_idx);

            Box3D box3d{
                center3d[0], center3d[1], center3d[2],
                dim_k[0], dim_k[1], dim_k[2],
                rot_y};

            // KITTI bottom-center origin fix: shift up by half height
            box3d.y += box3d.height * 0.5f;

            Detection det;
            det.class_id = topk.classes[k];
            det.score = score;
            det.box2d = {x1, y1, x2, y2};
            det.box3d = box3d;

            detections.push_back(det);
        }

        return detections;
    }

} // namespace monocon