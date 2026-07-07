#include "inference/onnx_inference.hpp"

#include <stdexcept>

namespace monocon {

    OnnxInference::OnnxInference(const std::string& onnx_path, int intra_op_threads)
        : env_(ORT_LOGGING_LEVEL_WARNING, "monocon_onnx"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(intra_op_threads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), options);
        io_binding_ = std::make_unique<Ort::IoBinding>(*session_);

        Ort::AllocatorWithDefaultOptions allocator;

        const size_t num_inputs = session_->GetInputCount();
        input_names_.reserve(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name_alloc = session_->GetInputNameAllocated(i, allocator);
            input_names_.emplace_back(name_alloc.get());
            input_name_holders_.push_back(std::move(name_alloc));
        }

        const size_t num_outputs = session_->GetOutputCount();
        output_names_.reserve(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name_alloc = session_->GetOutputNameAllocated(i, allocator);
            output_names_.emplace_back(name_alloc.get());
            output_name_holders_.push_back(std::move(name_alloc));
        }

        // Output shapes are fully static (graph exported with
        // dynamic_axes=None), so we can bind them immediately — no need
        // to wait for a real input to know their shape.
        bind_outputs();
    }

    void OnnxInference::bind_outputs() {
        if (outputs_bound_) {
            throw std::logic_error("OnnxInference: bind_outputs() called more than once");
        }

        ort_output_values_.reserve(output_names_.size());

        for (const auto& name : output_names_) {
            const size_t index = ort_output_values_.size();

            Ort::TypeInfo type_info = session_->GetOutputTypeInfo(index);
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

        outputs_bound_ = true;
    }

    void OnnxInference::bind_input(const std::string& name, const std::vector<int64_t>& shape) {
        if (inputs_.count(name)) {
            throw std::logic_error(
                "OnnxInference: bind_input('" + name + "') called more than once — "
                "shapes are assumed fixed for the lifetime of this object");
        }

        BoundTensor tensor;
        tensor.name = name;
        tensor.shape = shape;
        tensor.data.resize(tensor.numel(), 0.0f);

        ort_input_values_.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            tensor.data.data(), tensor.data.size(),
            tensor.shape.data(), tensor.shape.size()));

        io_binding_->BindInput(name.c_str(), ort_input_values_.back());

        inputs_[name] = std::move(tensor);
    }

    std::vector<float>& OnnxInference::input_data(const std::string& name) {
        auto it = inputs_.find(name);
        if (it == inputs_.end()) {
            throw std::runtime_error("OnnxInference: input '" + name + "' was never bound via bind_input()");
        }
        return it->second.data;
    }

    void OnnxInference::run() {
        session_->Run(Ort::RunOptions{nullptr}, *io_binding_);
    }

} // namespace monocon