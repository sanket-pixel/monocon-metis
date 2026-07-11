#pragma once

#include "inference/aipu_inference.hpp"

#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

namespace monocon {

    // One frame's worth of raw AIPU output, ready for Thread B to
    // dequantize + head + decode. Carries everything Thread B needs
    // so it never touches anything Thread A owns.
    struct QueuedFrame {
        std::vector<int8_t> raw_aipu_output;  // copy of int8 output (~1.9MB for 576ch model)
        AipuTensorInfo output_info;            // scale/zero_point/padding for dequantize
        int feat_h = 0;
        int feat_w = 0;
        int ori_height = 0;                    // original image size for decode
        int ori_width = 0;
        int frame_index = 0;
    };

    // Fixed-depth, thread-safe bounded queue.
    // push() blocks if full — Thread A naturally throttles if Thread B falls behind.
    // pop() blocks if empty — Thread B waits if Thread A hasn't produced yet.
    // sentinel() pushes a poison-pill (frame_index=-1) to signal Thread B to exit.
    class FrameQueue {
    public:
        explicit FrameQueue(int max_depth = 3) : max_depth_(max_depth) {}

        void push(QueuedFrame frame) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_not_full_.wait(lock, [this] { return static_cast<int>(queue_.size()) < max_depth_; });
            queue_.push(std::move(frame));
            cv_not_empty_.notify_one();
        }

        QueuedFrame pop() {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_not_empty_.wait(lock, [this] { return !queue_.empty(); });
            QueuedFrame frame = std::move(queue_.front());
            queue_.pop();
            cv_not_full_.notify_one();
            return frame;
        }

        // Push a poison pill — tells Thread B to exit its loop.
        void sentinel() {
            QueuedFrame poison;
            poison.frame_index = -1;
            push(std::move(poison));
        }

    private:
        int max_depth_;
        std::queue<QueuedFrame> queue_;
        std::mutex mutex_;
        std::condition_variable cv_not_full_;
        std::condition_variable cv_not_empty_;
    };

} // namespace monocon