#include "inference/head.hpp"

#include <stdexcept>

namespace monocon {

    HeadRuntime::HeadRuntime(const std::string& onnx_path, int intra_op_threads)
        : env_(ORT_LOGGING_LEVEL_WARNING, "monocon_head"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(intra_op_threads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), options);
        io_binding_ = std::make_unique<Ort::IoBinding>(*session_);

        Ort::AllocatorWithDefaultOptions allocator;

        // --- Input: shape is only fully known once we see the real feat_h/feat_w,
        // but we can defer actual buffer allocation to the first forward() call
        // and rebind only if the shape ever changes (it won't, in this pipeline —
        // one fixed input resolution throughout).
        auto input_name_alloc = session_->GetInputNameAllocated(0, allocator);
        input_.name = input_name_alloc.get();
        input_name_holders_.push_back(std::move(input_name_alloc));

        // --- Outputs: shapes ARE fully known now (all fixed, static-shaped
        // per model/export.py's dynamic_axes=None), so bind these once, here.
        const size_t num_outputs = session_->GetOutputCount();
        ort_output_values_.reserve(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            auto name_alloc = session_->GetOutputNameAllocated(i, allocator);
            std::string name = name_alloc.get();
            output_name_holders_.push_back(std::move(name_alloc));

            Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
            auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();

            BoundTensor tensor;
            tensor.name = name;
            tensor.shape = shape;
            tensor.data.resize(tensor.numel(), 0.0f);

            ort_output_values_.push_back(Ort::Value::CreateTensor<float>(
                memory_info_,
                tensor.data.data(), tensor.data.size(),
                tensor.shape.data(), tensor.shape.size()));

            io_binding_->BindOutput(name.c_str(), ort_output_values_.back());

            outputs_[name] = std::move(tensor);
        }
    }

    const std::map<std::string, BoundTensor>& HeadRuntime::forward(
        const std::vector<float>& feat, int64_t feat_h, int64_t feat_w) {

        // Bind the input tensor once, first call only (or if shape changes —
        // it never does in this pipeline, but the check is cheap insurance).
        if (bound_feat_h_ != feat_h || bound_feat_w_ != feat_w) {
            input_.shape = {1, 64, feat_h, feat_w};
            input_.data.resize(input_.numel());

            ort_input_values_.clear();
            ort_input_values_.push_back(Ort::Value::CreateTensor<float>(
                memory_info_,
                input_.data.data(), input_.data.size(),
                input_.shape.data(), input_.shape.size()));

            io_binding_->BindInput(input_.name.c_str(), ort_input_values_.back());

            bound_feat_h_ = feat_h;
            bound_feat_w_ = feat_w;
        }

        // Copy caller's data into the already-bound buffer — this is the
        // only per-frame "allocation-like" cost, and it's just a memcpy.
        std::copy(feat.begin(), feat.end(), input_.data.begin());

        session_->Run(Ort::RunOptions{nullptr}, *io_binding_);

        // outputs_ tensors' .data vectors were written in-place by Run(),
        // since they're bound directly to the same memory — no copy needed.
        return outputs_;
    }

} // namespace monocon