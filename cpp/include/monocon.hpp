#pragma once

#include "inference/backbone_neck.hpp"
#include "pipeline/frame_queue.h"
#include "postprocess/decode.hpp"
#include "postprocess/head_tail_cpu.h"
#include "utils/calib.hpp"

#include <string>
#include <thread>
#include <vector>

namespace monocon {

    struct StageTimings {
        double preprocess_quantize_ms = 0;
        double aipu_ms = 0;
        double dequantize_ms = 0;
        double head_ms = 0;
        double decode_ms = 0;
        double total_ms = 0;
    };

    class Monocon {
    public:
        Monocon(const std::string& aipu_model_dir,
               const std::string& head_weights_path,
               const std::string& calib_path);

        // Single-image path — unchanged, still works as before.
        std::vector<Detection> run(const cv::Mat& img_rgb, float score_thres = 0.3f);
        std::vector<Detection> run_timed(const cv::Mat& img_rgb, StageTimings& timings,
                                         float score_thres = 0.3f);

        // Pipelined video path — processes all frames with AIPU/CPU overlap.
        // Returns one detection list per frame, in input order.
        std::vector<std::vector<Detection>> process_video(
            const std::vector<cv::Mat>& frames,
            float score_thres = 0.3f);

        const ProjMatrix& calib() const { return P2_; }

    private:
        BackboneNeck backbone_neck_;
        HeadTailCPU head_;
        ProjMatrix P2_;

        bool warmed_up_ = false;
        std::vector<float> feat_;

        void warm_up();
        std::vector<Detection> run_impl(const cv::Mat& img_rgb, float score_thres,
                                        StageTimings* timings);
    };

} // namespace monocon