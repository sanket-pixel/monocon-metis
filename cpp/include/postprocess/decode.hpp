#pragma once

#include "inference/head.hpp"
#include "utils/calib.hpp"
#include "utils/geometry_ops.hpp"

#include <map>
#include <string>
#include <vector>

namespace monocon {

    struct Detection {
        int class_id;          // 0=Pedestrian, 1=Cyclist, 2=Car
        float score;
        std::array<float, 4> box2d;  // x1, y1, x2, y2
        Box3D box3d;
    };

    // Direct C++ port of python/model/decode.py's decode_predictions().
    // topk=30, num_alpha_bins=12, num_kpts=9 hardcoded, matching head.py.
    std::vector<Detection> decode_predictions(
        const std::map<std::string, HeadOutput>& pred,
        const ProjMatrix& P2,
        int img_h, int img_w,
        float score_thres = 0.3f);

} // namespace monocon