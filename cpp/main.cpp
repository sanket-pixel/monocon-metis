#include "monocon.hpp"
#include "utils/visualize.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

namespace {

    constexpr const char* AIPU_MODEL_DIR  = "weights/aipu/monocon_backbone_neck_conv1_fused";
    constexpr const char* HEAD_WEIGHTS    = "weights/head_tail_weights.bin";
    constexpr const char* CALIB_PATH      = "data/KITTI_sample/calib/calib_cam_to_cam.txt";
    constexpr const char* SAMPLE_IMAGE    = "data/sample.png";
    constexpr const char* IMAGE_DIR       = "data/KITTI_sample/images/data";
    constexpr const char* OUTPUT_IMAGE    = "output/sample_result.png";
    constexpr const char* OUTPUT_VIDEO    = "output/monocon_demo.mp4";
    constexpr int         PROFILE_FRAMES  = 50;
    constexpr int         VIDEO_FPS       = 20;

    double mean(const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    }

    double median(std::vector<double> v) {
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) / 2.0 : v[n/2];
    }

    std::vector<cv::Mat> load_images(const std::string& dir) {
        std::vector<std::string> paths;
        for (const auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".png") paths.push_back(e.path().string());
        std::sort(paths.begin(), paths.end());

        std::vector<cv::Mat> images;
        images.reserve(paths.size());
        for (const auto& p : paths) {
            cv::Mat bgr = cv::imread(p, cv::IMREAD_COLOR);
            cv::Mat rgb;
            cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
            images.push_back(rgb);
        }
        return images;
    }

    // ---------------------------------------------------------------
    // run: single image, visualized output
    // ---------------------------------------------------------------
    void run_mode() {
        monocon::Monocon model(AIPU_MODEL_DIR, HEAD_WEIGHTS, CALIB_PATH);

        cv::Mat bgr = cv::imread(SAMPLE_IMAGE, cv::IMREAD_COLOR);
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        auto detections = model.run(rgb);
        std::cout << "Detections: " << detections.size() << "\n";

        cv::Mat annotated = monocon::draw_3d_boxes(rgb, detections, model.calib());
        cv::cvtColor(annotated, annotated, cv::COLOR_RGB2BGR);
        fs::create_directories(fs::path(OUTPUT_IMAGE).parent_path());
        cv::imwrite(OUTPUT_IMAGE, annotated);
        std::cout << "Saved " << OUTPUT_IMAGE << "\n";
    }

    // ---------------------------------------------------------------
    // profile: sequential per-frame stage breakdown
    // ---------------------------------------------------------------
    void profile_mode() {
        monocon::Monocon model(AIPU_MODEL_DIR, HEAD_WEIGHTS, CALIB_PATH);
        auto images = load_images(IMAGE_DIR);

        if (images.empty()) {
            std::cerr << "No images found in '" << IMAGE_DIR << "'\n";
            return;
        }

        std::vector<double> pq, aipu, deq, head, dec, total;

        const auto t_start = Clock::now();

        for (int i = 0; i < PROFILE_FRAMES; ++i) {
            monocon::StageTimings t;
            model.run_timed(images[i % images.size()], t);

            pq.push_back(t.preprocess_quantize_ms);
            aipu.push_back(t.aipu_ms);
            deq.push_back(t.dequantize_ms);
            head.push_back(t.head_ms);
            dec.push_back(t.decode_ms);
            total.push_back(t.total_ms);
        }

        const double wall_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();

        std::cout << "\n--------------------------------------------------\n";
        std::cout << "[Stage Breakdown - Internal Timers]\n";
        std::cout << "preprocess+quantize: mean=" << mean(pq)   << " median=" << median(pq)   << " ms\n"
                  << "aipu execution:      mean=" << mean(aipu) << " median=" << median(aipu) << " ms\n"
                  << "dequantize:          mean=" << mean(deq)  << " median=" << median(deq)  << " ms\n"
                  << "head:                mean=" << mean(head) << " median=" << median(head) << " ms\n"
                  << "decode:              mean=" << mean(dec)  << " median=" << median(dec)  << " ms\n"
                  << "internal sum:        mean=" << mean(total) << " ms\n";
        std::cout << "--------------------------------------------------\n";
        std::cout << "[True End-to-End Metrics]\n";
        std::cout << "Wall Clock FPS:      " << (PROFILE_FRAMES * 1000.0 / wall_ms) << " FPS\n"
                  << "Wall Clock per img:  " << (wall_ms / PROFILE_FRAMES) << " ms\n";
        std::cout << "--------------------------------------------------\n";
    }

    // ---------------------------------------------------------------
    // video: pipelined inference over full image directory
    // ---------------------------------------------------------------
    void video_mode() {
    monocon::Monocon model(AIPU_MODEL_DIR, HEAD_WEIGHTS, CALIB_PATH);
    auto images = load_images(IMAGE_DIR);

    if (images.empty()) {
        std::cerr << "No images found in '" << IMAGE_DIR << "'\n";
        return;
    }

    std::cout << "Processing " << images.size() << " frames (pipelined)...\n";

    const auto t_start = Clock::now();
    auto all_detections = model.process_video(images);
    const double wall_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();

    const int n = static_cast<int>(images.size());
    std::cout << "--------------------------------------------------\n"
              << "[Pipelined Video Results]\n"
              << "Frames:          " << n << "\n"
              << "Total wall time: " << wall_ms << " ms\n"
              << "Throughput FPS:  " << n * 1000.0 / wall_ms << "\n"
              << "Per-frame (avg): " << wall_ms / n << " ms\n"
              << "--------------------------------------------------\n";

    // BEV canvas dimensions — tuned for KITTI's typical detection range
    constexpr int BEV_W = 500;
    constexpr int BEV_H = 600;
    constexpr float METERS_PER_PIXEL = 0.07f;  // 1 pixel = 10cm → 60m depth range

    cv::VideoWriter writer;
    int written = 0;

    for (int i = 0; i < n; ++i) {
        // Camera view with 3D boxes
        cv::Mat camera_view = monocon::draw_3d_boxes(
            images[i], all_detections[i], model.calib());
        cv::cvtColor(camera_view, camera_view, cv::COLOR_RGB2BGR);

        // BEV view
        cv::Mat bev_view = monocon::draw_bev(
            all_detections[i], BEV_W, BEV_H, METERS_PER_PIXEL);

        // Resize BEV to match camera height
        cv::Mat bev_resized;
        cv::resize(bev_view, bev_resized,
                   cv::Size(BEV_W, camera_view.rows));

        // Stack horizontally: [camera | BEV]
        cv::Mat combined;
        cv::hconcat(camera_view, bev_resized, combined);

        if (!writer.isOpened()) {
            fs::create_directories(fs::path(OUTPUT_VIDEO).parent_path());
            writer.open(OUTPUT_VIDEO,
                       cv::VideoWriter::fourcc('m','p','4','v'),
                       VIDEO_FPS, combined.size());
        }
        writer.write(combined);
        ++written;
    }

    writer.release();
    std::cout << "Saved " << written << " frames to '" << OUTPUT_VIDEO << "'\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 ||
        (std::string(argv[1]) != "run" &&
         std::string(argv[1]) != "profile" &&
         std::string(argv[1]) != "video")) {
        std::cerr << "Usage: " << argv[0] << " run|profile|video\n";
        return 1;
    }

    try {
        const std::string mode = argv[1];
        if      (mode == "run")     run_mode();
        else if (mode == "profile") profile_mode();
        else                        video_mode();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}