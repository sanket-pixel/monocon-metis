#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace monocon {

    // Scoped timer — prints elapsed ms when it goes out of scope.
    // Usage: { ScopedTimer t("preprocess"); ...code...; }
    class ScopedTimer {
    public:
        explicit ScopedTimer(std::string label)
            : label_(std::move(label)), start_(std::chrono::high_resolution_clock::now()) {}

        ~ScopedTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start_).count();
            std::cout << "[timing] " << label_ << ": " << ms << " ms\n";
        }

    private:
        std::string label_;
        std::chrono::high_resolution_clock::time_point start_;
    };

    // Non-scoped variant when you want to capture the value instead of just printing.
    class Timer {
    public:
        Timer() : start_(std::chrono::high_resolution_clock::now()) {}

        double elapsed_ms() const {
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start_).count();
        }

        void reset() { start_ = std::chrono::high_resolution_clock::now(); }

    private:
        std::chrono::high_resolution_clock::time_point start_;
    };

} // namespace monocon