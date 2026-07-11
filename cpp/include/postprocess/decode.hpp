#pragma once

#include "utils/calib.hpp"
#include "utils/geometry_ops.hpp"

#include <array>
#include <map>
#include <string>
#include <vector>

namespace monocon {

    struct Detection {
        int class_id;
        float score;
        std::array<float, 4> box2d;
        Box3D box3d;
    };

    std::vector<Detection> decode_predictions(
     const std::map<std::string, std::vector<float>>& pred,
     const ProjMatrix& P2,
     int img_h, int img_w,
     int feat_h, int feat_w,
     float score_thres = 0.3f);

} // namespace monocon