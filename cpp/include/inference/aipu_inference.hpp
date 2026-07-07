#pragma once

#include "axruntime/axruntime.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace monocon {

    // Metadata for one AIPU tensor (input or output), pulled directly from
    // axrTensorInfo. Exposed so preprocess/quantize.cpp and
    // postprocess/dequantize.cpp can read the real scale/zero_point/padding
    // for THIS compiled model, instead of hardcoding values that could
    // silently drift out of sync with the actual compiled artifact.
    struct AipuTensorInfo {
        std::string name;
        std::vector<int64_t> shape;         // logical (unpadded) shape
        std::vector<int64_t> padded_shape;  // hardware-padded shape
        std::vector<std::pair<int64_t, int64_t>> padding;  // (before, after) per dim
        double scale = 1.0;
        int32_t zero_point = 0;
        size_t size_bytes = 0;
    };

    // Owns the full AxRuntime resource chain for one compiled model:
    //   Context -> Connection -> Model -> ModelInstance
    //
    // Runs raw int8 in, raw int8 out. Fully generic over input/output count —
    // does not assume a single input or single output. Quantization
    // (preprocess) and dequantization (postprocess) deliberately live
    // OUTSIDE this class, in preprocess/quantize.* and postprocess/dequantize.*
    // respectively — this class's only job is "talk to the AIPU."
    //
    // Currently wraps a single ModelInstance on `num_cores` cores. The
    // constructor and resource layout are written so that running one
    // ModelInstance PER core (round-robin pipelined across 4 instances)
    // is a straightforward extension later — not built yet, since a
    // single instance is enough to validate correctness first.
    class AipuInference {
    public:
        explicit AipuInference(const std::string& model_dir, int num_cores = 4);
        ~AipuInference();

        AipuInference(const AipuInference&) = delete;
        AipuInference& operator=(const AipuInference&) = delete;
        AipuInference(AipuInference&&) noexcept = default;
        AipuInference& operator=(AipuInference&&) noexcept = default;

        int input_count() const { return static_cast<int>(input_infos_.size()); }
        int output_count() const { return static_cast<int>(output_infos_.size()); }

        const AipuTensorInfo& input_info(int index) const { return input_infos_.at(index); }
        const AipuTensorInfo& input_info(const std::string& name) const;

        const AipuTensorInfo& output_info(int index) const { return output_infos_.at(index); }
        const AipuTensorInfo& output_info(const std::string& name) const;

        // Caller must have already quantized+padded the input into the
        // buffer(s) returned by input_buffer(i), matching input_info(i).size_bytes.
        int8_t* input_buffer(int index = 0);

        // Runs inference. After this returns, read results via output_buffer(i)
        // for i in [0, output_count()) — this does NOT return a pointer itself,
        // since there is no longer a single canonical "the output" once a
        // model can have N outputs. Throws on failure.
        void run();

        const int8_t* output_buffer(int index) const;

    private:
        axrContext* context_ = nullptr;
        axrConnection* connection_ = nullptr;
        axrModel* model_ = nullptr;
        axrModelInstance* instance_ = nullptr;

        std::vector<axrTensorInfo> raw_input_infos_;
        std::vector<axrTensorInfo> raw_output_infos_;

        std::vector<AipuTensorInfo> input_infos_;
        std::vector<AipuTensorInfo> output_infos_;

        std::vector<std::vector<int8_t>> input_memory_;
        std::vector<std::vector<int8_t>> output_memory_;

        std::vector<axrArgument> input_args_;
        std::vector<axrArgument> output_args_;

        int find_input_index(const std::string& name) const;
        int find_output_index(const std::string& name) const;

        AipuTensorInfo to_tensor_info(const axrTensorInfo& raw) const;
    };

} // namespace monocon