#ifndef TURBO_PI_DOG_TRACKER_TRACKER_HPP
#define TURBO_PI_DOG_TRACKER_TRACKER_HPP
#pragma once


#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>


namespace fs = std::filesystem;

struct Detection {
        cv::Rect2f box; // the detected object rectangle in source-frame pixel coordinates.
        float confidence = 0.0f;
        int class_id     = -1;
};

struct Track {
        // For persisting object tracking across multiple frames.
        cv::Rect2f box;
        float confidence = 0.0f;
        int id           = 1;
        int missed       = 0;
        bool active      = false;
};

struct TrackingResult {
        std::vector<Detection> detections;
        Track track;
};

struct ModelConfig {
        fs::path model_weights_path{};
        float confidence_threshold = 0.5f;
        float nms_threshold        = 0.5f;
        int input_size             = 640;
        int batch_size             = 32;
        bool use_cuda              = false;
};

class DogTracker {
    public:
        explicit DogTracker(const ModelConfig& model_config);

        struct Letterbox {
                float scale = 1.0f;
                float pad_x = 0.0f;
                float pad_y = 0.0f;
        };

        // Runs preprocessing, ONNX inference, dog-only postprocessing, NMS, and single-target
        // tracking.
        std::vector<TrackingResult> process_batch(const std::vector<cv::Mat>& frames);
        TrackingResult process(const cv::Mat& frame);

    private:
        ModelConfig model_config;
        Ort::Env ort_env;
        Ort::Session ort_session{nullptr};
        std::string input_name;
        std::string output_name;
        int fixed_batch_size = 1;
        Track track;
        // Buffers used to build one ONNX input batch.
        size_t frame_values = 0;
        std::vector<float> input_buffer;
        std::vector<float> padding_sample;
        std::vector<Letterbox> letterboxes;

        void preprocess(const cv::Mat& frame, std::vector<float>& output, Letterbox* info) const;
        std::vector<Detection> parse_detections(const Ort::Value& output,
                                                size_t batch_index,
                                                const Letterbox& letterbox,
                                                int frame_width,
                                                int frame_height) const;
};

#endif // TURBO_PI_DOG_TRACKER_TRACKER_HPP
