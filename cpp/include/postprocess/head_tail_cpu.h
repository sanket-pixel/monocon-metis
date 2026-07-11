#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>

namespace monocon {

    // -----------------------------------------------------------------------
    // Architecture constants — match python/model/head.py exactly.
    // -----------------------------------------------------------------------
    constexpr int NUM_HEADS      = 9;
    constexpr int FEAT_CH        = 64;
    constexpr int NUM_AFFINE     = 10;
    constexpr int NUM_ALPHA_BINS = 12;

    // Canonical head order — must match HEAD_NAMES in head.py and the
    // channel block layout of HeadConv1Fused's (1, 576, H, W) output.
    const std::array<std::string, NUM_HEADS> HEAD_TAIL_ORDER = {
        "heatmap", "wh", "offset", "center2kpt_offset",
        "kpt_heatmap", "kpt_heatmap_offset", "dim", "depth", "dir_feat"};

    // Output channel count per head. dir_feat (index 8) has no conv2 —
    // it feeds dir_cls and dir_reg instead.
    const std::array<int, NUM_HEADS> HEAD_OUT_CHANNELS = {
        3, 2, 2, 18, 9, 2, 3, 2, 0};

    // -----------------------------------------------------------------------
    // Weight structs — loaded once from the binary exported by
    // python/scripts/export_head_tail_weights.py, reused every forward().
    // -----------------------------------------------------------------------

    struct AttnBatchNormWeights {
        std::vector<float> running_mean;    // (FEAT_CH,)
        std::vector<float> running_var;     // (FEAT_CH,)
        std::vector<float> affine_weight;   // (NUM_AFFINE, FEAT_CH) — weight_ in Python
        std::vector<float> affine_bias;     // (NUM_AFFINE, FEAT_CH) — bias_ in Python
        std::vector<float> attn_conv_weight;// (NUM_AFFINE, FEAT_CH) — attention[0].weight
        std::vector<float> inner_bn_gamma;  // (NUM_AFFINE,)
        std::vector<float> inner_bn_beta;   // (NUM_AFFINE,)
        std::vector<float> inner_bn_mean;   // (NUM_AFFINE,)
        std::vector<float> inner_bn_var;    // (NUM_AFFINE,)
    };

    struct Conv1x1Weights {
        std::vector<float> weight;  // (out_channels, FEAT_CH)
        std::vector<float> bias;    // (out_channels,)
        int out_channels = 0;
    };

    // -----------------------------------------------------------------------
    // HeadTailCPU
    //
    // Direct C++ port of HeadTail (python/model/head.py).
    // Replaces an ONNXRuntime session whose ~150-node graph was measured
    // at 55–60ms per frame on a desktop Ryzen — not because the arithmetic
    // is expensive (~24K parameters), but because ORT's per-node dispatch
    // overhead dominates at this parameter count.
    //
    // This implementation eliminates dispatch overhead entirely: all weights
    // are loaded once at construction, all scratch buffers are preallocated
    // via resize_for(), and forward() performs zero heap allocation.
    // -----------------------------------------------------------------------
    class HeadTailCPU {
    public:
        // Loads weights from the binary written by export_head_tail_weights.py.
        explicit HeadTailCPU(const std::string& weights_path);

        // Allocates all scratch and output buffers for a given spatial size.
        // Called automatically by forward() on the first call or if H/W changes.
        void resize_for(int feat_height, int feat_width);

        // Runs the full HeadTail pipeline:
        //   AttnBatchNorm2d + ReLU + conv2 for each of the 9 heads,
        //   followed by sigmoid/clamp activations on heatmaps and depth.
        //
        // fused_conv1_output: flat (576 * H * W) NCHW float buffer.
        //   Channel block [i*64 : (i+1)*64] corresponds to HEAD_TAIL_ORDER[i].
        //
        // Returns a const ref to the internal output map — valid until the
        // next forward() call. Caller must not store the reference across calls.
        const std::map<std::string, std::vector<float>>& forward(
            const std::vector<float>& fused_conv1_output,
            int feat_height,
            int feat_width);

    private:
        std::array<AttnBatchNormWeights, NUM_HEADS> attn_bns_;
        std::array<Conv1x1Weights, NUM_HEADS>       conv2_;    // dir_feat slot unused
        Conv1x1Weights dir_cls_;
        Conv1x1Weights dir_reg_;

        // Preallocated scratch — zero heap allocation after resize_for().
        std::array<std::vector<float>, NUM_HEADS>   normed_outputs_;
        std::map<std::string, std::vector<float>>   outputs_;
        int current_feat_h_ = -1;
        int current_feat_w_ = -1;

        // Per-head compute primitives — called from the parallel head loop.
        void apply_attn_bn_relu(const float* __restrict__ input,
                                int head_index,
                                int feat_height, int feat_width,
                                float* __restrict__ output);

        void apply_conv1x1(const float* __restrict__ input,
                           const Conv1x1Weights& weights,
                           int feat_height, int feat_width,
                           float* __restrict__ output);
    };

} // namespace monocon