#include "utils/geometry_ops.hpp"

#include <cmath>

namespace monocon {

    Corners3D extract_corners_from_box3d(const Box3D& box) {
        // Same corner ordering + rotation convention as python/utils/geometry_ops.py
        // extract_corners_from_bboxes_3d: origin (0.5, 1.0, 0.5) camera-coords box.
        static constexpr int order[8] = {0, 1, 3, 2, 4, 5, 7, 6};

        std::array<std::array<float, 3>, 8> corners_norm;
        for (int i = 0; i < 8; ++i) {
            corners_norm[i][0] = ((i & 1) ? 0.5f : -0.5f);
            corners_norm[i][1] = ((i & 2) ? 0.0f : -1.0f);
            corners_norm[i][2] = ((i & 4) ? 0.5f : -0.5f);
        }

        std::array<std::array<float, 3>, 8> reordered;
        for (int i = 0; i < 8; ++i) reordered[i] = corners_norm[order[i]];

        const float cos_r = std::cos(box.rotation_y);
        const float sin_r = std::sin(box.rotation_y);

        Corners3D out;
        for (int i = 0; i < 8; ++i) {
            const float cx = reordered[i][0] * box.length;
            const float cy = reordered[i][1] * box.height;
            const float cz = reordered[i][2] * box.width;

            // Rotation about the Y axis (camera up-axis), matching
            // rotation_3d_in_axis(..., axis=1) in geometry_ops.py
            const float rx = cx * cos_r - cz * sin_r;
            const float rz = cx * sin_r + cz * cos_r;

            out[i][0] = rx + box.x;
            out[i][1] = cy + box.y;
            out[i][2] = rz + box.z;
        }

        return out;
    }

    Corners2D project_corners_to_image(const Corners3D& corners, const ProjMatrix& P2) {
        Corners2D out;
        for (int i = 0; i < 8; ++i) {
            const float x = corners[i][0], y = corners[i][1], z = corners[i][2];

            const float px = P2[0] * x + P2[1] * y + P2[2] * z + P2[3];
            const float py = P2[4] * x + P2[5] * y + P2[6] * z + P2[7];
            const float pz = P2[8] * x + P2[9] * y + P2[10] * z + P2[11];

            out[i][0] = px / pz;
            out[i][1] = py / pz;
        }
        return out;
    }

} // namespace monocon