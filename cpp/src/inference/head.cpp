#include "inference/head.hpp"

#include <stdexcept>

namespace monocon {

    HeadRuntime::HeadRuntime(const std::string& onnx_path, int intra_op_threads)
        : onnx_(onnx_path, intra_op_threads) {

        if (onnx_.input_count() != static_cast<int>(HEAD_NAMES.size())) {
            throw std::runtime_error(
                "HeadRuntime: expected " + std::to_string(HEAD_NAMES.size()) +
                " inputs, ONNX graph has " + std::to_string(onnx_.input_count()));
        }
    }

    void HeadRuntime::bind_inputs_once(int64_t feat_h, int64_t feat_w) {
        // Bind each ONNX input by its ACTUAL name (exporter-mangled, e.g.
        // "wh.1"), using POSITION in onnx_.input_names() to match up with
        // HEAD_NAMES — the literal strings differ, but export order is
        // preserved, which is what matters here.
        const auto& real_names = onnx_.input_names();

        if (real_names.size() != HEAD_NAMES.size()) {
            throw std::runtime_error(
                "HeadRuntime::bind_inputs_once: ONNX graph has " +
                std::to_string(real_names.size()) + " inputs, expected " +
                std::to_string(HEAD_NAMES.size()));
        }

        for (size_t i = 0; i < HEAD_NAMES.size(); ++i) {
            onnx_.bind_input(real_names[i], {1, 64, feat_h, feat_w});
        }

        inputs_bound_ = true;
    }

    void HeadRuntime::forward(const std::vector<std::vector<float>>& conv1_outputs,
                             int64_t feat_h, int64_t feat_w) {

        if (conv1_outputs.size() != HEAD_NAMES.size()) {
            throw std::runtime_error(
                "HeadRuntime::forward: expected " + std::to_string(HEAD_NAMES.size()) +
                " conv1 outputs, got " + std::to_string(conv1_outputs.size()));
        }

        if (!inputs_bound_) {
            bind_inputs_once(feat_h, feat_w);
        }

        const auto& real_names = onnx_.input_names();

        for (size_t i = 0; i < HEAD_NAMES.size(); ++i) {
            auto& dst = onnx_.input_data(real_names[i]);
            std::copy(conv1_outputs[i].begin(), conv1_outputs[i].end(), dst.begin());
        }

        onnx_.run();
    }

} // namespace monocon