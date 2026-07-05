#include "preprocess/opencv_preprocess.hpp"
#include "preprocess/quantize.hpp"
#include "inference/backbone_neck.hpp"
#include "inference/head.hpp"
#include "postprocess/dequantize.hpp"
#include "postprocess/decode.hpp"
#include "utils/calib.hpp"
#include "utils/visualize.hpp"
#include "utils/timer.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>

namespace fs = std::filesystem;

namespace {

    struct StageTimings {
        double preprocess_ms = 0;
        double quantize_aipu_ms = 0;
        double dequantize_ms = 0;
        double head_ms = 0;
        double decode_ms = 0;
        double total_ms = 0;
    };

    double mean(const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    }

    double median(std::vector<double> v) {
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        return (n % 2 == 0) ? (v[n / 2 - 1] + v[n / 2]) / 2.0 : v[n / 2];
    }

    void print_stats(const std::string& label, std::vector<double> values) {
        std::cout << "  " << label << ": mean=" << mean(values) << " ms, median=" << median(values) << " ms\n";
    }

    StageTimings run_one_frame(
        monocon::BackboneNeck& backbone_neck,
        monocon::HeadRuntime& head,
        const monocon::ProjMatrix& P2,
        const std::string& image_path,
        std::vector<monocon::Detection>* out_detections = nullptr) {

        StageTimings t;
        monocon::Timer frame_timer;
        monocon::Timer stage_timer;

        auto image = monocon::preprocess_image(image_path);
        t.preprocess_ms = stage_timer.elapsed_ms();

        stage_timer.reset();
        const int8_t* raw_output = backbone_neck.forward(image);
        t.quantize_aipu_ms = stage_timer.elapsed_ms();

        stage_timer.reset();
        const auto& out_info = backbone_neck.output_info();
        auto feat = monocon::dequantize_padded_nhwc_to_nchw(raw_output, out_info);
        t.dequantize_ms = stage_timer.elapsed_ms();

        const int64_t feat_h = out_info.shape[1];
        const int64_t feat_w = out_info.shape[2];

        stage_timer.reset();
        auto pred = head.forward(feat, feat_h, feat_w);
        t.head_ms = stage_timer.elapsed_ms();

        stage_timer.reset();
        auto detections = monocon::decode_predictions(pred, P2, image.height, image.width, 0.3f);
        t.decode_ms = stage_timer.elapsed_ms();

        t.total_ms = frame_timer.elapsed_ms();

        if (out_detections) *out_detections = std::move(detections);
        return t;
    }

    void run_benchmark(
        const std::string& image_dir,
        const std::string& aipu_model_dir,
        const std::string& head_onnx_path,
        const std::string& calib_path,
        int warmup_frames,
        int measured_frames) {

        std::vector<std::string> image_paths;
        for (const auto& entry : fs::directory_iterator(image_dir)) {
            if (entry.path().extension() == ".png") image_paths.push_back(entry.path().string());
        }
        std::sort(image_paths.begin(), image_paths.end());

        const int total_needed = warmup_frames + measured_frames;
        if (static_cast<int>(image_paths.size()) < total_needed) {
            std::cout << "WARNING: only " << image_paths.size() << " images found, need "
                      << total_needed << ". Will cycle through available images.\n";
        }

        std::cout << "Loading models...\n";
        monocon::BackboneNeck backbone_neck(aipu_model_dir);
        monocon::HeadRuntime head(head_onnx_path, 8);
        auto P2 = monocon::parse_calib_p2(calib_path);

        std::cout << "\nWarming up (" << warmup_frames << " frames, not counted)...\n";
        for (int i = 0; i < warmup_frames; ++i) {
            const auto& path = image_paths[i % image_paths.size()];
            run_one_frame(backbone_neck, head, P2, path);
        }

        std::cout << "Running benchmark (" << measured_frames << " frames)...\n";

        std::vector<double> preprocess_v, quantize_aipu_v, dequantize_v, head_v, decode_v, total_v;

        for (int i = 0; i < measured_frames; ++i) {
            const auto& path = image_paths[(warmup_frames + i) % image_paths.size()];
            auto t = run_one_frame(backbone_neck, head, P2, path);

            preprocess_v.push_back(t.preprocess_ms);
            quantize_aipu_v.push_back(t.quantize_aipu_ms);
            dequantize_v.push_back(t.dequantize_ms);
            head_v.push_back(t.head_ms);
            decode_v.push_back(t.decode_ms);
            total_v.push_back(t.total_ms);

            if ((i + 1) % 20 == 0) std::cout << "  ..." << (i + 1) << "/" << measured_frames << "\n";
        }

        std::cout << "\n=== Benchmark Results (" << measured_frames << " frames, "
                  << warmup_frames << " warm-up discarded) ===\n";
        print_stats("preprocess", preprocess_v);
        print_stats("quantize+aipu_inference", quantize_aipu_v);
        print_stats("dequantize", dequantize_v);
        print_stats("head_onnxruntime", head_v);
        print_stats("decode", decode_v);
        print_stats("TOTAL_PER_FRAME", total_v);

        const double mean_total = mean(total_v);
        std::cout << "\nSteady-state FPS: " << (1000.0 / mean_total) << "\n";
    }

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--benchmark") {
        if (argc < 6) {
            std::cerr << "Usage: " << argv[0]
                      << " --benchmark <image_dir> <aipu_model_dir> <head_onnx_path> <calib_path> "
                         "[warmup_frames=10] [measured_frames=100]\n";
            return 1;
        }
        const std::string image_dir = argv[2];
        const std::string aipu_model_dir = argv[3];
        const std::string head_onnx_path = argv[4];
        const std::string calib_path = argv[5];
        const int warmup_frames = (argc > 6) ? std::stoi(argv[6]) : 10;
        const int measured_frames = (argc > 7) ? std::stoi(argv[7]) : 100;

        run_benchmark(image_dir, aipu_model_dir, head_onnx_path, calib_path, warmup_frames, measured_frames);
        return 0;
    }

    // --- Single-image mode (unchanged from before) ---
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <image_path> <aipu_model_dir> <head_onnx_path> <calib_path>\n"
                  << "   or: " << argv[0]
                  << " --benchmark <image_dir> <aipu_model_dir> <head_onnx_path> <calib_path> [warmup] [measured]\n";
        return 1;
    }

    const std::string image_path = argv[1];
    const std::string aipu_model_dir = argv[2];
    const std::string head_onnx_path = argv[3];
    const std::string calib_path = argv[4];

    std::cout << "Loading models...\n";
    monocon::BackboneNeck backbone_neck(aipu_model_dir);
    monocon::HeadRuntime head(head_onnx_path);
    auto P2 = monocon::parse_calib_p2(calib_path);

    std::vector<monocon::Detection> detections;
    auto t = run_one_frame(backbone_neck, head, P2, image_path, &detections);

    std::cout << "\n[timing] preprocess: " << t.preprocess_ms << " ms\n";
    std::cout << "[timing] quantize+aipu_inference: " << t.quantize_aipu_ms << " ms\n";
    std::cout << "[timing] dequantize: " << t.dequantize_ms << " ms\n";
    std::cout << "[timing] head_onnxruntime: " << t.head_ms << " ms\n";
    std::cout << "[timing] decode: " << t.decode_ms << " ms\n";
    std::cout << "[timing] TOTAL_PER_FRAME: " << t.total_ms << " ms (" << (1000.0 / t.total_ms) << " FPS)\n\n";

    std::cout << "Found " << detections.size() << " detections:\n";
    static const char* class_names[] = {"Pedestrian", "Cyclist", "Car"};
    for (const auto& det : detections) {
        std::cout << "  [" << class_names[det.class_id] << "] score=" << det.score
                  << " box3d=(" << det.box3d.x << "," << det.box3d.y << "," << det.box3d.z << ")\n";
    }

    auto image = monocon::preprocess_image(image_path);
    cv::Mat img_2d = monocon::draw_2d_boxes(image.ori_img, detections);
    cv::cvtColor(img_2d, img_2d, cv::COLOR_RGB2BGR);
    cv::imwrite("output/cpp_result_2d.png", img_2d);

    cv::Mat img_3d = monocon::draw_3d_boxes(image.ori_img, detections, P2);
    cv::cvtColor(img_3d, img_3d, cv::COLOR_RGB2BGR);
    cv::imwrite("output/cpp_result_3d.png", img_3d);

    std::cout << "\nSaved output/cpp_result_2d.png and output/cpp_result_3d.png\n";

    return 0;
}