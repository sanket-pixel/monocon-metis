#include "postprocess/head_tail_cpu.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace monocon {

    namespace {

        std::vector<float> read_tensor(std::ifstream& file) {
            uint32_t byte_length = 0;
            file.read(reinterpret_cast<char*>(&byte_length), sizeof(byte_length));
            std::vector<float> tensor(byte_length / sizeof(float));
            file.read(reinterpret_cast<char*>(tensor.data()), byte_length);
            return tensor;
        }

        // HSigmoid(x) = clamp((x + 3) / 6, 0, 1) — inline to help the
        // compiler vectorize the attention weight loop.
        inline float hsigmoid(float x) {
            return std::max(0.0f, std::min(1.0f, (x + 3.0f) / 6.0f));
        }

        // Sigmoid clamped to (heat_min, heat_max) — avoids log(0) in decode.
        inline float sigmoid_clamped(float x) {
            const float sig = 1.0f / (1.0f + std::exp(-x));
            return std::max(1e-4f, std::min(1.0f - 1e-4f, sig));
        }

    } // namespace

    // -----------------------------------------------------------------------
    // Construction — load weights from binary
    // -----------------------------------------------------------------------

    HeadTailCPU::HeadTailCPU(const std::string& weights_path) {
        std::ifstream file(weights_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error(
                "HeadTailCPU: cannot open weights file '" + weights_path + "'");
        }

        for (int head = 0; head < NUM_HEADS; ++head) {
            AttnBatchNormWeights& bn = attn_bns_[head];
            bn.running_mean     = read_tensor(file);
            bn.running_var      = read_tensor(file);
            bn.affine_weight    = read_tensor(file);
            bn.affine_bias      = read_tensor(file);
            bn.attn_conv_weight = read_tensor(file);
            bn.inner_bn_gamma   = read_tensor(file);
            bn.inner_bn_beta    = read_tensor(file);
            bn.inner_bn_mean    = read_tensor(file);
            bn.inner_bn_var     = read_tensor(file);
        }

        for (int head = 0; head < NUM_HEADS; ++head) {
            if (HEAD_TAIL_ORDER[head] == "dir_feat") continue;
            conv2_[head].out_channels = HEAD_OUT_CHANNELS[head];
            conv2_[head].weight       = read_tensor(file);
            conv2_[head].bias         = read_tensor(file);
        }

        dir_cls_.out_channels = NUM_ALPHA_BINS;
        dir_cls_.weight       = read_tensor(file);
        dir_cls_.bias         = read_tensor(file);

        dir_reg_.out_channels = NUM_ALPHA_BINS;
        dir_reg_.weight       = read_tensor(file);
        dir_reg_.bias         = read_tensor(file);
    }

    // -----------------------------------------------------------------------
    // Buffer allocation
    // -----------------------------------------------------------------------

    void HeadTailCPU::resize_for(int feat_height, int feat_width) {
        if (feat_height == current_feat_h_ && feat_width == current_feat_w_) return;

        const int spatial_size = feat_height * feat_width;

        for (int head = 0; head < NUM_HEADS; ++head) {
            normed_outputs_[head].resize(FEAT_CH * spatial_size);
        }

        // Scratch buffer for raw heatmap conv2 output (before sigmoid+clamp).
        outputs_["heatmap"].resize(3 * spatial_size);

        outputs_["center_heatmap"].resize(3  * spatial_size);
        outputs_["kpt_heatmap"].resize(9     * spatial_size);
        outputs_["wh"].resize(2              * spatial_size);
        outputs_["offset"].resize(2          * spatial_size);
        outputs_["kpt_heatmap_offset"].resize(2  * spatial_size);
        outputs_["center2kpt_offset"].resize(18  * spatial_size);
        outputs_["dim"].resize(3             * spatial_size);
        outputs_["depth"].resize(2           * spatial_size);
        outputs_["alpha_cls"].resize(NUM_ALPHA_BINS  * spatial_size);
        outputs_["alpha_offset"].resize(NUM_ALPHA_BINS * spatial_size);

        current_feat_h_ = feat_height;
        current_feat_w_ = feat_width;
    }

    // -----------------------------------------------------------------------
    // AttnBatchNorm2d + ReLU
    //
    // Implements the Python AttnBatchNorm2d.forward() in three stages:
    //   1. Compute per-channel spatial mean/var over the full (H, W) map.
    //   2. Derive attention weights via a tiny 64→10 linear + BN1d + HSigmoid.
    //   3. Apply the resulting per-channel affine transform + BatchNorm +
    //      ReLU to produce the normalized output feature map.
    // -----------------------------------------------------------------------

    void HeadTailCPU::apply_attn_bn_relu(
        const float* __restrict__ input,
        int head_index,
        int feat_height, int feat_width,
        float* __restrict__ output) {

        const AttnBatchNormWeights& bn = attn_bns_[head_index];
        const int spatial_size = feat_height * feat_width;

        // --- Stage 1: spatial mean and variance per channel ---
        float channel_mean[FEAT_CH];
        float channel_var[FEAT_CH];

        for (int channel = 0; channel < FEAT_CH; ++channel) {
            const float* channel_ptr = input + channel * spatial_size;
            float sum    = 0.0f;
            float sum_sq = 0.0f;

            #pragma omp simd reduction(+:sum,sum_sq)
            for (int pixel = 0; pixel < spatial_size; ++pixel) {
                sum    += channel_ptr[pixel];
                sum_sq += channel_ptr[pixel] * channel_ptr[pixel];
            }

            const float mean   = sum / static_cast<float>(spatial_size);
            channel_mean[channel] = mean;
            channel_var[channel]  = sum_sq / static_cast<float>(spatial_size) - mean * mean;
        }

        // --- Stage 2: attention weights (64 → 10 linear → BN1d → HSigmoid) ---

        // y[c] = mean[c] / sqrt(var[c] + eps) — the RSD statistic
        float rsd[FEAT_CH];
        for (int channel = 0; channel < FEAT_CH; ++channel) {
            rsd[channel] = channel_mean[channel]
                         / std::sqrt(channel_var[channel] + 1e-3f);
        }

        // attn_conv: (NUM_AFFINE, FEAT_CH) @ (FEAT_CH,) → (NUM_AFFINE,)
        float attn_conv_out[NUM_AFFINE];
        Eigen::Map<const Eigen::Matrix<float, NUM_AFFINE, FEAT_CH, Eigen::RowMajor>>
            attn_weight_mat(bn.attn_conv_weight.data());
        Eigen::Map<const Eigen::Vector<float, FEAT_CH>>
            rsd_vec(rsd);
        Eigen::Map<Eigen::Vector<float, NUM_AFFINE>>
            attn_conv_vec(attn_conv_out);
        attn_conv_vec.noalias() = attn_weight_mat * rsd_vec;

        // Inner BatchNorm1d
        float attn_weights[NUM_AFFINE];
        for (int affine = 0; affine < NUM_AFFINE; ++affine) {
            const float normalized = (attn_conv_out[affine] - bn.inner_bn_mean[affine])
                                   / std::sqrt(bn.inner_bn_var[affine] + 1e-5f);
            const float bn1d_out   = normalized * bn.inner_bn_gamma[affine]
                                   + bn.inner_bn_beta[affine];
            attn_weights[affine] = hsigmoid(bn1d_out);
        }

        // Per-channel affine scale/shift = attn @ affine_weight/bias
        // Both are (NUM_AFFINE, FEAT_CH) — result is (FEAT_CH,)
        float channel_scale[FEAT_CH];
        float channel_shift[FEAT_CH];

        Eigen::Map<const Eigen::Matrix<float, NUM_AFFINE, FEAT_CH, Eigen::RowMajor>>
            weight_mat(bn.affine_weight.data());
        Eigen::Map<const Eigen::Matrix<float, NUM_AFFINE, FEAT_CH, Eigen::RowMajor>>
            bias_mat(bn.affine_bias.data());
        Eigen::Map<const Eigen::Vector<float, NUM_AFFINE>>
            attn_vec(attn_weights);
        Eigen::Map<Eigen::Vector<float, FEAT_CH>>
            scale_vec(channel_scale);
        Eigen::Map<Eigen::Vector<float, FEAT_CH>>
            shift_vec(channel_shift);

        scale_vec.noalias() = attn_vec.transpose() * weight_mat;
        shift_vec.noalias() = attn_vec.transpose() * bias_mat;

        // --- Stage 3: BatchNorm2d (running stats) + attentive scale/shift + ReLU ---
        const float bn_eps = 0.001f;

        for (int channel = 0; channel < FEAT_CH; ++channel) {
            const float bn_inv_std = 1.0f / std::sqrt(bn.running_var[channel] + bn_eps);
            const float bn_mean    = bn.running_mean[channel];
            const float scale      = channel_scale[channel];
            const float shift      = channel_shift[channel];

            const float* src = input  + channel * spatial_size;
            float*       dst = output + channel * spatial_size;

            #pragma omp simd
            for (int pixel = 0; pixel < spatial_size; ++pixel) {
                const float normed = (src[pixel] - bn_mean) * bn_inv_std;
                dst[pixel] = std::max(0.0f, scale * normed + shift);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 1×1 convolution
    //
    // Implements out[out_ch, H, W] = weight[out_ch, FEAT_CH] @ in[FEAT_CH, H, W]
    //                                + bias[out_ch]
    // Cache-blocked over the spatial dimension to stay within L1.
    // -----------------------------------------------------------------------

    void HeadTailCPU::apply_conv1x1(
        const float* __restrict__ input,
        const Conv1x1Weights&     weights,
        int feat_height, int feat_width,
        float* __restrict__ output) {

        const int spatial_size  = feat_height * feat_width;
        const int out_channels  = weights.out_channels;
        const int L1_BLOCK_SIZE = 1024; // 4 KB — fits in Ryzen L1 data cache

        for (int block_start = 0; block_start < spatial_size; block_start += L1_BLOCK_SIZE) {
            const int block_end = std::min(block_start + L1_BLOCK_SIZE, spatial_size);

            for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
                float* dst = output + out_ch * spatial_size;

                #pragma omp simd
                for (int pixel = block_start; pixel < block_end; ++pixel) {
                    dst[pixel] = weights.bias[out_ch];
                }

                for (int in_ch = 0; in_ch < FEAT_CH; ++in_ch) {
                    const float  w   = weights.weight[out_ch * FEAT_CH + in_ch];
                    const float* src = input + in_ch * spatial_size;

                    #pragma omp simd
                    for (int pixel = block_start; pixel < block_end; ++pixel) {
                        dst[pixel] += w * src[pixel];
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // forward()
    // -----------------------------------------------------------------------

    const std::map<std::string, std::vector<float>>& HeadTailCPU::forward(
        const std::vector<float>& fused_conv1_output,
        int feat_height,
        int feat_width) {

        resize_for(feat_height, feat_width);
        const int spatial_size = feat_height * feat_width;

        // Process all 9 heads in parallel — each thread runs the full
        // AttnBatchNorm2d + ReLU + conv2 pipeline for one head, keeping
        // normed_outputs_[i] hot in L1/L2 between the two stages.
        #pragma omp parallel for schedule(static)
        for (int head = 0; head < NUM_HEADS; ++head) {
            const float* head_input = fused_conv1_output.data()
                                    + head * FEAT_CH * spatial_size;

            apply_attn_bn_relu(head_input, head,
                               feat_height, feat_width,
                               normed_outputs_[head].data());

            const std::string& head_name = HEAD_TAIL_ORDER[head];

            if (head_name != "dir_feat") {
                apply_conv1x1(normed_outputs_[head].data(), conv2_[head],
                              feat_height, feat_width,
                              outputs_[head_name].data());
            } else {
                apply_conv1x1(normed_outputs_[head].data(), dir_cls_,
                              feat_height, feat_width,
                              outputs_["alpha_cls"].data());
                apply_conv1x1(normed_outputs_[head].data(), dir_reg_,
                              feat_height, feat_width,
                              outputs_["alpha_offset"].data());
            }
        }

        // Sigmoid + clamp activations on heatmaps
        const std::vector<float>& raw_heatmap = outputs_["heatmap"];
        std::vector<float>&       center_heatmap = outputs_["center_heatmap"];

        #pragma omp parallel for simd schedule(static)
        for (size_t pixel = 0; pixel < center_heatmap.size(); ++pixel) {
            center_heatmap[pixel] = sigmoid_clamped(raw_heatmap[pixel]);
        }

        std::vector<float>& kpt_heatmap = outputs_["kpt_heatmap"];
        #pragma omp parallel for simd schedule(static)
        for (size_t pixel = 0; pixel < kpt_heatmap.size(); ++pixel) {
            kpt_heatmap[pixel] = sigmoid_clamped(kpt_heatmap[pixel]);
        }

        // Depth: sigmoid → (1/sigmoid) - 1 for the value channel;
        // log-variance channel (depth[HW:]) is left unchanged.
        std::vector<float>& depth = outputs_["depth"];
        #pragma omp parallel for schedule(static)
        for (int pixel = 0; pixel < spatial_size; ++pixel) {
            const float sigmoid_val = 1.0f / (1.0f + std::exp(-depth[pixel]));
            depth[pixel] = (1.0f / (sigmoid_val + 1e-12f)) - 1.0f;
        }

        return outputs_;
    }

} // namespace monocon