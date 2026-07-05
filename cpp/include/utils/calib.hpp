#pragma once

#include <array>
#include <string>

namespace monocon {

    // 3x4 camera projection matrix, row-major.
    using ProjMatrix = std::array<float, 12>;

    // Parses P_rect_02 out of a KITTI calib_cam_to_cam.txt file.
    ProjMatrix parse_calib_p2(const std::string& calib_path);

} // namespace monocon