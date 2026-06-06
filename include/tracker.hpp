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
        bool use_tensorrt          = false;
        bool tensorrt_fp16         = true;
        fs::path tensorrt_cache_path{"tensorrt_cache"};
};

class DogTracker {
    public:
        explicit DogTracker(const ModelConfig& model_config);

        struct Letterbox {
                float scale = 1.0f;
                float pad_x = 0.0f;
                float pad_y = 0.0f;
        };

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
        bool using_tensorrt_provider = false;

#if USE_CUDA && USE_TENSORRT
        // CUDAGraphIO exists to avoid allocating/copying tensors repeatedly during TensorRT-backed
        // ONNX Runtime inference. It pre-allocates GPU input/output tensors, binds them once via
        // Ort::IoBinding, then reuses them for each run.
        struct CUDAGraphIO {
                CUDAGraphIO(Ort::Session& session,
                            const std::string& input_name,
                            const std::string& output_name,
                            const std::vector<int64_t>& input_shape,
                            const std::vector<int64_t>& output_shape,
                            int fixed_batch_size,
                            int input_size);

                const float* run(Ort::Session& session,
                                 const std::vector<float>& host_input_buffer);

                std::vector<int64_t> output_shape;
                size_t input_element_count  = 0;
                size_t output_element_count = 0;
                std::vector<float> host_output_buffer;
                // These are the actual device memory addresses. They point to GPU memory containing
                // tensor data:
                float* input_data_gpu_ptr  = nullptr;
                float* output_data_gpu_ptr = nullptr;
                // RAII wrappers around internal ONNX Runtime handles
                Ort::MemoryInfo cuda_memory_info{nullptr};
                Ort::Allocator cuda_allocator{nullptr};
                // These are ONNX Runtime tensor objects.
                Ort::Value gpu_input_tensor{nullptr};
                Ort::Value gpu_output_tensor{nullptr};
                // Reusable mapping of model inputs/outputs to fixed GPU tensor buffers.
                Ort::IoBinding io_binding{nullptr};
        };
        std::unique_ptr<CUDAGraphIO> cuda_graph_io;
#endif

        void preprocess(const cv::Mat& frame, std::vector<float>& output, Letterbox* info) const;
        [[nodiscard]] std::vector<Detection> parse_detections(const Ort::Value& output,
                                                              size_t batch_index,
                                                              const Letterbox& letterbox,
                                                              int frame_width,
                                                              int frame_height) const;
        [[nodiscard]] std::vector<Detection> parse_detections(const float* output_data,
                                                              const std::vector<int64_t>& shape,
                                                              size_t batch_index,
                                                              const Letterbox& letterbox,
                                                              int frame_width,
                                                              int frame_height) const;
};

#endif // TURBO_PI_DOG_TRACKER_TRACKER_HPP
