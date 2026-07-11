#include "monocon.hpp"

#include "postprocess/dequantize.hpp"
#include "utils/timer.hpp"

namespace monocon {

    Monocon::Monocon(const std::string& aipu_model_dir,
                     const std::string& head_weights_path,
                     const std::string& calib_path)
        : backbone_neck_(aipu_model_dir),
          head_(head_weights_path),
          P2_(parse_calib_p2(calib_path)) {}

    void Monocon::warm_up() {
        const int input_h = static_cast<int>(backbone_neck_.input_info().shape[1]);
        const int input_w = static_cast<int>(backbone_neck_.input_info().shape[2]);

        cv::Mat dummy(input_h, input_w, CV_8UC3, cv::Scalar(0, 0, 0));

        for (int i = 0; i < 10; ++i) {
            backbone_neck_.forward(dummy);
        }
        warmed_up_ = true;
    }

    std::vector<Detection> Monocon::run_impl(const cv::Mat& img_rgb, float score_thres, StageTimings* timings) {
        Timer total_timer;
        Timer stage;

        if (!warmed_up_) warm_up();

        double quantize_ms = 0, aipu_ms = 0;
        backbone_neck_.forward_timed(img_rgb, quantize_ms, aipu_ms);
        if (timings) { timings->preprocess_quantize_ms = quantize_ms; timings->aipu_ms = aipu_ms; }

        const auto& out_info = backbone_neck_.output_info(0);
        const int feat_h = static_cast<int>(out_info.shape[1]);
        const int feat_w = static_cast<int>(out_info.shape[2]);

        stage.reset();
        dequantize_padded_nhwc_to_nchw_into(backbone_neck_.output_buffer(0), out_info, feat_);
        if (timings) timings->dequantize_ms = stage.elapsed_ms();

        stage.reset();
        auto& pred = head_.forward(feat_, feat_h, feat_w);
        if (timings) timings->head_ms = stage.elapsed_ms();

        stage.reset();
        auto detections = decode_predictions(pred, P2_, img_rgb.rows, img_rgb.cols, feat_h, feat_w, score_thres);;
        if (timings) timings->decode_ms = stage.elapsed_ms();

        if (timings) timings->total_ms = total_timer.elapsed_ms();

        return detections;
    }

    std::vector<Detection> Monocon::run(const cv::Mat& img_rgb, float score_thres) {
        return run_impl(img_rgb, score_thres, nullptr);
    }

    std::vector<Detection> Monocon::run_timed(const cv::Mat& img_rgb, StageTimings& timings, float score_thres) {
        return run_impl(img_rgb, score_thres, &timings);
    }

  std::vector<std::vector<Detection>> Monocon::process_video(
    const std::vector<cv::Mat>& frames,
    float score_thres) {

    if (!warmed_up_) warm_up();

    const int n_frames = static_cast<int>(frames.size());
    std::vector<std::vector<Detection>> results(n_frames);

    FrameQueue queue(3);

    // --- Thread B (consumer): dequantize -> head -> decode ---
    // Owns nothing from Monocon except head_ and P2_ (via capture).
    // Uses its own local_feat buffer — no sharing with Thread A.
    std::thread consumer([&]() {
        std::vector<float> local_feat;

        while (true) {
            QueuedFrame frame = queue.pop();

            if (frame.frame_index == -1) break;

            dequantize_padded_nhwc_to_nchw_into(
                frame.raw_aipu_output.data(),
                frame.output_info,
                local_feat);

            const auto& pred = head_.forward(local_feat, frame.feat_h, frame.feat_w);

            results[frame.frame_index] = decode_predictions(
                pred,
                P2_,
                frame.ori_height, frame.ori_width,
                frame.feat_h, frame.feat_w,
                score_thres);
        }
    });

    // --- Thread A (producer, main thread): preprocess -> AIPU -> queue ---
    // Owns backbone_neck_ exclusively.
    for (int i = 0; i < n_frames; ++i) {
        double pq_ms = 0, aipu_ms = 0;
        backbone_neck_.forward_aipu_only(frames[i], pq_ms, aipu_ms);

        const auto& out_info = backbone_neck_.output_info(0);

        QueuedFrame qf;
        qf.raw_aipu_output.resize(out_info.size_bytes);
        std::memcpy(
            qf.raw_aipu_output.data(),
            backbone_neck_.output_buffer(0),
            out_info.size_bytes);
        qf.output_info = out_info;
        qf.feat_h = static_cast<int>(out_info.shape[1]);
        qf.feat_w = static_cast<int>(out_info.shape[2]);
        qf.ori_height = frames[i].rows;
        qf.ori_width = frames[i].cols;
        qf.frame_index = i;

        queue.push(std::move(qf));
    }

    queue.sentinel();
    consumer.join();

    return results;
}


} // namespace monocon