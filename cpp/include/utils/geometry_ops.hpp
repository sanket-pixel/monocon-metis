#pragma once

#include "utils/calib.hpp"

#include <array>
#include <vector>

namespace monocon {

    // One 3D box: x, y, z, length, height, width, rotation_y (camera coords).
    struct Box3D {
        float x, y, z, length, height, width, rotation_y;
    };

    // 8 corners of a 3D box, in camera coordinates.
    using Corners3D = std::array<std::array<float, 3>, 8>;

    // 8 corners projected to image pixel coordinates.
    using Corners2D = std::array<std::array<float, 2>, 8>;

    Corners3D extract_corners_from_box3d(const Box3D& box);
    Corners2D project_corners_to_image(const Corners3D& corners, const ProjMatrix& P2);

} // namespace monocon