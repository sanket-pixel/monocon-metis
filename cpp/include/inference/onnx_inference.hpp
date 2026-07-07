#pragma once

#include <onnxruntime_cxx_api.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace monocon {

    // A named tensor with its own owned float buffer, bound directly into
    // an Ort::Value so ONNXRuntime reads/writes this memory in place —
    // no per-frame Ort::Value allocation once binding has happened once.
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

    // Generic ONNXRuntime session wrapper, with NO knowledge of MonoCon,
    // HeadTail, or any specific model's input/output semantics. Owns the
    // session, binds every declared input/output tensor by name exactly
    // once (via IoBinding), and exposes a name-keyed read/write interface.
    //
    // Model-specific concerns (which names to write, how to interpret
    // outputs, fixed shapes derived from image resolution, etc.) belong
    // in a wrapper class built on top of this — see HeadRuntime.
    class OnnxInference {
    public:
        explicit OnnxInference(const std::string& onnx_path, int intra_op_threads = 4);

        int input_count() const { return static_cast<int>(input_names_.size()); }
        int output_count() const { return static_cast<int>(output_names_.size()); }

        const std::vector<std::string>& input_names() const { return input_names_; }
        const std::vector<std::string>& output_names() const { return output_names_; }

        // Binds an input tensor's shape and allocates its buffer. Must be
        // called once per input, before the first run(), whenever the
        // caller knows the real shape (ONNXRuntime does not expose this
        // ahead of time for a graph with fully-dynamic input dims).
        // Safe to call again later only if the shape actually changes —
        // rebinding is not free, so callers should avoid calling this
        // every frame once shapes have stabilized.
        void bind_input(const std::string& name, const std::vector<int64_t>& shape);

        // Direct, mutable access to an already-bound input's buffer, for
        // the caller to fill with data before run().
        std::vector<float>& input_data(const std::string& name);

        // Runs inference using whatever has been written into each bound
        // input's buffer. Every output tensor is allocated automatically
        // on first run() (output shapes ARE known ahead of time from the
        // graph, unlike inputs) and reused on every subsequent call.
        void run();

        const BoundTensor& output(const std::string& name) const { return outputs_.at(name); }
        const BoundTensor& output(int index) const { return outputs_.at(output_names_.at(index)); }

    private:
        Ort::Env env_;
        std::unique_ptr<Ort::Session> session_;
        std::unique_ptr<Ort::IoBinding> io_binding_;
        Ort::MemoryInfo memory_info_;

        std::vector<Ort::AllocatedStringPtr> input_name_holders_;
        std::vector<Ort::AllocatedStringPtr> output_name_holders_;

        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;

        std::map<std::string, BoundTensor> inputs_;
        std::vector<Ort::Value> ort_input_values_;   // parallel to input_names_, kept alive once bound

        std::map<std::string, BoundTensor> outputs_;
        std::vector<Ort::Value> ort_output_values_;  // bound once at construction — shapes are static

        bool outputs_bound_ = false;
        void bind_outputs();
    };

} // namespace monocon