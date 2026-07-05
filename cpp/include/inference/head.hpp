#pragma once

#include <onnxruntime_cxx_api.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace monocon {

    // Lightweight named tensor — owns its own float buffer, bound directly
    // into an Ort::Value at construction so ORT reads/writes this memory
    // in place, no per-frame Ort::Value allocation.
    struct BoundTensor {
        std::string name;
        std::vector<int64_t> shape;
        std::vector<float> data;

        size_t numel() const {
            size_t n = 1;
            for (auto d : shape) n *= static_cast<size_t>(d);
            return n;
        }
    };
    using HeadOutput = BoundTensor;
    // Runs monocon_head.onnx via ONNXRuntime CPU, using IoBinding so all
    // input/output tensors are allocated ONCE at construction, not per-frame.
    // forward() just writes into the bound input buffer and calls Run().
    class HeadRuntime {
    public:
        explicit HeadRuntime(const std::string& onnx_path, int intra_op_threads = 4);

        // Writes `feat` into the bound input tensor and runs inference.
        // Returns a reference to internal output tensors — valid until the
        // next forward() call, no copying needed by the caller.
        const std::map<std::string, BoundTensor>& forward(
            const std::vector<float>& feat, int64_t feat_h, int64_t feat_w);

        const BoundTensor& output(const std::string& name) const { return outputs_.at(name); }

    private:
        Ort::Env env_;
        std::unique_ptr<Ort::Session> session_;
        std::unique_ptr<Ort::IoBinding> io_binding_;
        Ort::MemoryInfo memory_info_;

        std::vector<Ort::AllocatedStringPtr> input_name_holders_;
        std::vector<Ort::AllocatedStringPtr> output_name_holders_;

        BoundTensor input_;
        std::vector<Ort::Value> ort_input_values_;   // kept alive, bound once

        std::map<std::string, BoundTensor> outputs_;
        std::vector<Ort::Value> ort_output_values_;  // kept alive, bound once

        int64_t bound_feat_h_ = 0;
        int64_t bound_feat_w_ = 0;
    };

} // namespace monocon