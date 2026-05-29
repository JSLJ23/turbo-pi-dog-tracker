#ifndef TURBO_PI_DOG_TRACKER_CLI_HPP
#define TURBO_PI_DOG_TRACKER_CLI_HPP
#pragma once


#include <filesystem>
#include <string>
#include <vector>


namespace fs = std::filesystem;

// This software has two live modes and one offline render mode.
// demo: live camera -> preview window + JSONL stdout
// server: live camera -> TCP JSONL
// render: video file -> annotated output video, with telemetry disabled by default
enum class RunMode {
    Demo,
    Server,
    Render
};

enum class TelemetryMode {
    Stdout,
    TCP,
    None
};

struct Configuration {
        RunMode run_mode;
        fs::path model_weights_path{"models/yolo26n_bs1.onnx"};
        fs::path camera_device_path{"/dev/video0"};
        fs::path input_video_path{};
        fs::path output_video_path{};
        std::string telemetry_host   = "127.0.0.1";
        int telemetry_port           = 8765;
        float confidence_threshold   = 0.5f;
        float nms_threshold          = 0.5f;
        int input_size               = 640;
        int batch_size               = 1;
        TelemetryMode telemetry_mode = TelemetryMode::Stdout;
        bool show_help               = false;
};


std::string show_usage(const char* argv0);
Configuration parse_args(int argc, const char* argv[]);
std::string to_string(const RunMode& run_mode);
std::string to_string(const TelemetryMode& telemetry_mode);

#endif // TURBO_PI_DOG_TRACKER_CLI_HPP
