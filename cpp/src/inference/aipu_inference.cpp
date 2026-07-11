#include "inference/aipu_inference.hpp"
#include <iostream>
#include <filesystem>
#include <stdexcept>

namespace monocon {

     AipuInference::AipuInference(const std::string& model_dir, int num_cores) {
        context_ = axr_create_context();
        connection_ = axr_device_connect(context_, nullptr, num_cores, nullptr);
        const auto model_path = std::filesystem::path(model_dir) / "model.json";
        model_ = axr_load_model(context_, model_path.string().c_str());
        const std::string property_string =
            "double_buffer=0;input_dmabuf=0;num_sub_devices=1;aipu_cores=" + std::to_string(num_cores);
        auto* properties = axr_create_properties(context_, property_string.c_str());
        instance_ = axr_load_model_instance(connection_, model_, properties);
        axr_destroy(reinterpret_cast<const axrObject*>(properties));
        // --- Inputs ---
        const int input_count = axr_num_model_inputs(model_);
        raw_input_infos_.resize(input_count);
        input_infos_.resize(input_count);
        input_memory_.resize(input_count);
        input_args_.resize(input_count);
        for (int i = 0; i < input_count; ++i) {
            raw_input_infos_[i] = axr_get_model_input(model_, i);
            input_infos_[i] = to_tensor_info(raw_input_infos_[i]);
            input_memory_[i].resize(input_infos_[i].size_bytes);
            input_args_[i] = axrArgument{ input_memory_[i].data(),-1, 0, input_memory_[i].size() };
        }
        // --- Outputs ---
        const int output_count = axr_num_model_outputs(model_);
        raw_output_infos_.resize(output_count);
        output_infos_.resize(output_count);
        output_memory_.resize(output_count);
        output_args_.resize(output_count);
        for (int i = 0; i < output_count; ++i) {
            raw_output_infos_[i] = axr_get_model_output(model_, i);
            output_infos_[i] = to_tensor_info(raw_output_infos_[i]);
            output_memory_[i].resize(output_infos_[i].size_bytes);
            output_args_[i] = axrArgument{output_memory_[i].data(), -1, 0, output_memory_[i].size()};
        }
    }


    AipuInference::~AipuInference() {
        if (instance_) axr_destroy(reinterpret_cast<const axrObject*>(instance_));
        if (model_) axr_destroy(reinterpret_cast<const axrObject*>(model_));
        // connection_ / context_ intentionally not destroyed here if AxRuntime
        // expects a shared context across multiple models — revisit if we
        // ever load backbone_neck + anything else AIPU-side in the same process.
    }

    AipuTensorInfo AipuInference::to_tensor_info(const axrTensorInfo& raw) const {
        AipuTensorInfo info;
        info.name = std::string(raw.name);
        info.scale = raw.scale;
        info.zero_point = raw.zero_point;
        info.size_bytes = axr_tensor_size(&raw);
        for (size_t d = 0; d < raw.ndims; ++d) {
            const int64_t pad_before = static_cast<int64_t>(raw.padding[d][0]);
            const int64_t pad_after = static_cast<int64_t>(raw.padding[d][1]);

            info.padded_shape.push_back(static_cast<int64_t>(raw.dims[d]));
            info.shape.push_back(static_cast<int64_t>(raw.dims[d]) - pad_before - pad_after);
            info.padding.emplace_back(pad_before, pad_after);
        }
        return info;
    }

    void AipuInference::run() {
    auto result = axr_run_model_instance(
        instance_,
        input_args_.data(), input_args_.size(),
        output_args_.data(), output_args_.size());
        if (result != AXR_SUCCESS) {
            throw std::runtime_error("AipuInference: axr_run_model_instance failed");
        }
    }

    int AipuInference::find_input_index(const std::string& name) const {
        for (size_t i = 0; i < input_infos_.size(); ++i) {
            if (input_infos_[i].name == name) return static_cast<int>(i);
        }
        throw std::runtime_error("AipuInference: no input named '" + name + "'");
    }

    int AipuInference::find_output_index(const std::string& name) const {
        for (size_t i = 0; i < output_infos_.size(); ++i) {
            if (output_infos_[i].name == name) return static_cast<int>(i);
        }
        throw std::runtime_error("AipuInference: no output named '" + name + "'");
    }

    const AipuTensorInfo& AipuInference::input_info(const std::string& name) const {
        return input_infos_.at(find_input_index(name));
    }

    const AipuTensorInfo& AipuInference::output_info(const std::string& name) const {
        return output_infos_.at(find_output_index(name));
    }

    int8_t* AipuInference::input_buffer(int index) {
        return input_memory_.at(index).data();
    }

    const int8_t* AipuInference::output_buffer(int index) const {
        return output_memory_.at(index).data();
    }


} // namespace monocon