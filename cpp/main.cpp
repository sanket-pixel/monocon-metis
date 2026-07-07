#include "preprocess/opencv_preprocess.hpp"
#include "inference/backbone_neck.hpp"
#include "inference/head.hpp"
#include "postprocess/dequantize.hpp"
#include "postprocess/decode.hpp"
#include "utils/calib.hpp"

#include <iostream>
#include <map>

namespace {

    // Order Voyager's compiler ACTUALLY produces for the AIPU's 9 outputs —
    // confirmed empirically (alphabetical by name), NOT the ONNX export order.
    const char* AIPU_OUTPUT_ORDER[9] = {
        "center2kpt_offset", "depth", "dim", "dir_feat", "heatmap",
        "kpt_heatmap", "kpt_heatmap_offset", "offset", "wh"};

    // Order HeadTail's ONNX graph was EXPORTED with — this is what
    // head.forward()'s positional vector<vector<float>> must match.
    const char* HEAD_TAIL_INPUT_ORDER[9] = {
        "heatmap", "wh", "offset", "center2kpt_offset",
        "kpt_heatmap", "kpt_heatmap_offset", "dim", "depth", "dir_feat"};

}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <image> <aipu_model_dir> <head_onnx> <calib>\n";
        return 1;
    }

    const std::string image_path = argv[1];
    const std::string aipu_model_dir = argv[2];
    const std::string head_onnx_path = argv[3];
    const std::string calib_path = argv[4];

    monocon::BackboneNeck backbone_neck(aipu_model_dir);
    monocon::HeadRuntime head(head_onnx_path, 4);
    auto P2 = monocon::parse_calib_p2(calib_path);

    auto image = monocon::preprocess_image(image_path);

    // Warm-up calls: first 1-2 run()s may return stale/zero output on this
    // AIPU. Third call's output is what we trust.
    backbone_neck.forward(image);
    backbone_neck.forward(image);
    backbone_neck.forward(image);

    const auto& sample_info = backbone_neck.output_info(0);
    const int64_t feat_h = sample_info.shape[1];
    const int64_t feat_w = sample_info.shape[2];

    std::map<std::string, int> aipu_index_by_name;
    for (int i = 0; i < 9; ++i) {
        aipu_index_by_name[AIPU_OUTPUT_ORDER[i]] = i;
    }

    std::vector<std::vector<float>> feats(9);
    for (int head_pos = 0; head_pos < 9; ++head_pos) {
        const int aipu_idx = aipu_index_by_name.at(HEAD_TAIL_INPUT_ORDER[head_pos]);
        monocon::dequantize_padded_nhwc_to_nchw_into(
            backbone_neck.output_buffer(aipu_idx),
            backbone_neck.output_info(aipu_idx),
            feats[head_pos]);
    }

    head.forward(feats, feat_h, feat_w);

    auto detections = monocon::decode_predictions(head, P2, image.height, image.width, 0.3f);

    std::cout << "Detections: " << detections.size() << "\n";

    return 0;
}