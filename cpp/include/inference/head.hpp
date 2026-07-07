#pragma once

#include "inference/onnx_inference.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace monocon {

    // Fixed order used by python/model/export.py's HeadTailWrapper and
    // BackboneNeckConv1's export — both AIPU output order and HeadTail
    // ONNX input order follow this exactly. Real ONNX input tensor NAMES
    // may be exporter-mangled (e.g. "wh" -> "wh.1") and must NOT be relied
    // on directly — only their POSITION in this fixed order is trustworthy.
    // Order HeadTail's ONNX graph was EXPORTED with — this is what
    // onnx_.input_names() (positionally) corresponds to. Untouched by the
    // AIPU compiler, since HeadTail was never compiled for the AIPU.
      constexpr std::array<const char*, 9> HEAD_NAMES = {
        "heatmap", "wh", "offset", "center2kpt_offset",
        "kpt_heatmap", "kpt_heatmap_offset", "dim", "depth", "dir_feat"};
    // MonoCon-specific wrapper around OnnxInference for the HeadTail model.
    // Takes HeadConv1's 9 dequantized feature maps (each 64 x feat_h x feat_w,
    // in HEAD_NAMES order) and produces the 10 final named prediction tensors
    // decode.cpp expects.
    class HeadRuntime {
    public:
        explicit HeadRuntime(const std::string& onnx_path, int intra_op_threads = 4);

        // conv1_outputs: 9 tensors in HEAD_NAMES order, each a flat
        // (64 * feat_h * feat_w) float buffer.
        void forward(const std::vector<std::vector<float>>& conv1_outputs,
                    int64_t feat_h, int64_t feat_w);

        // Final output access — these ARE the real, unmangled ONNX output
        // names (confirmed clean in the exported graph: center_heatmap,
        // wh, offset, ... no suffixing), safe to use directly.
        const BoundTensor& output(const std::string& name) const { return onnx_.output(name); }

    private:
        OnnxInference onnx_;
        bool inputs_bound_ = false;

        void bind_inputs_once(int64_t feat_h, int64_t feat_w);
    };

} // namespace monocon