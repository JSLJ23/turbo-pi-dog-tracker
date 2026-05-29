#include "tracker.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <thread>


namespace fs = std::filesystem;

constexpr int K_DOG_CLASS_ID    = 16; // COCO class id for dog.
constexpr int MAX_MISSED_FRAMES = 10;


// For mapping tensor float outputs to class labels
static bool looks_like_int_class(const float value)
{
    const float rounded = std::round(value);
    return std::abs(value - rounded) < 0.01F && rounded >= 0.0F && rounded <= 1000.0F;
}

// Computes intersection over union for detection boxes.
static float compute_iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    // Finds the left edge of the overlapping area.
    const float x1 = std::max(a.x, b.x);
    // Finds the top edge of the overlapping area.
    const float y1 = std::max(a.y, b.y);
    // Finds the right edge of the overlapping area.
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    // Finds the bottom edge of the overlapping area.
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    // Computes the intersection area taking into account non-overlapping boxes.
    const float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    // Computes the union area: total area covered by both boxes.
    const float union_area = a.area() + b.area() - intersection_area;
    // If the union area is invalid or zero, return 0.0, otherwise return the IOU.
    return union_area <= 0.0f ? 0.0f : intersection_area / union_area;
}

// restore_xyxy() converts a detection box from model input coordinates back into original
// camera/video frame coordinates.
// x1, y1 = top-left corner
// x2, y2 = bottom-right corner
static cv::Rect2f restore_xyxy(float x1,
                               float y1,
                               float x2,
                               float y2,
                               const DogTracker::Letterbox& letterbox,
                               const int frame_width,
                               const int frame_height)
{
    // Before inference, the frame is resized and padded into a square YOLO input.
    // That process is called letterboxing. The model predicts boxes in that square padded space,
    // not directly in the original camera frame.

    // Remove padding and rescale coordinates back to original frame.
    x1 = (x1 - letterbox.pad_x) / letterbox.scale;
    y1 = (y1 - letterbox.pad_y) / letterbox.scale;
    x2 = (x2 - letterbox.pad_x) / letterbox.scale;
    y2 = (y2 - letterbox.pad_y) / letterbox.scale;

    // Ensure values do not exceed original frame dimensions.
    x1 = std::clamp(x1, 0.0f, static_cast<float>(frame_width - 1));
    y1 = std::clamp(y1, 0.0f, static_cast<float>(frame_height - 1));
    x2 = std::clamp(x2, 0.0f, static_cast<float>(frame_width - 1));
    y2 = std::clamp(y2, 0.0f, static_cast<float>(frame_height - 1));

    // y increases downward for OpenCV frames
    return {x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1)};
}

static std::vector<Detection> nms(std::vector<Detection> detections, const float threshold)
{
    // The model may output several overlapping boxes for the same dog.
    // NMS keeps the best one and removes weaker boxes that overlap too much.
    std::ranges::sort(detections, [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    std::vector<Detection> kept;
    std::vector<bool> removed(detections.size(), false);
    for (auto [i, main_detection] : std::views::enumerate(detections)) {
        if (removed[i]) {
            continue;
        }
        kept.push_back(main_detection);
        for (auto [j, other_detection] : std::views::enumerate(detections)) {
            if (j <= i) {
                continue;
            }

            if (!removed[j] && compute_iou(main_detection.box, other_detection.box) > threshold) {
                removed[j] = true;
            }
        }
    }

    return kept;
}

static std::vector<Detection> parse_raw_model_outputs(const float* raw_data,
                                                      const int num_rows,
                                                      const DogTracker::Letterbox& letterbox,
                                                      const int frame_width,
                                                      const int frame_height,
                                                      const float confidence_threshold)
{
    // Model might produce multiple detections per frame, hence the need to parse them.
    // For a given frame frame:
    // -> model outputs many candidate rows
    // -> keep dog rows
    // -> remove duplicate overlapping boxes
    // -> choose/update one Track
    std::vector<Detection> detections;

    // Exported Ultralytics variants in this repo produce [x1, y1, x2, y2, score, class].
    auto at = [&](const int row, const int col) -> float { return raw_data[row * 6 + col]; };

    for (int i = 0; i < num_rows; i++) {
        const float maybe_class = at(i, 5);
        const float maybe_score = at(i, 4);
        if (looks_like_int_class(maybe_class) &&
            static_cast<int>(std::round(maybe_class)) == K_DOG_CLASS_ID &&
            maybe_score >= confidence_threshold) {
            const cv::Rect2f box = restore_xyxy(
                at(i, 0), at(i, 1), at(i, 2), at(i, 3), letterbox, frame_width, frame_height);
            // Ignore boxes that are basically empty or invalid.
            if (box.area() > 1.0f) {
                detections.push_back({box, maybe_score, K_DOG_CLASS_ID});
            }
        }
    }

    return detections;
}

Track update_track(Track& track, const std::vector<Detection>& detections)
{
    // The tracker tolerates 10 missed frames.
    // On the 11th missed frame, it marks the track inactive/lost.
    if (detections.empty()) {
        track.missed += 1;
        if (track.missed > MAX_MISSED_FRAMES) {
            track.active = false;
        }
        return track;
    }

    // Keep a single target by preferring overlap with the previous target, then confidence.
    size_t best       = 0;
    float best_metric = -std::numeric_limits<float>::infinity();
    for (const auto& [i, detection] : std::views::enumerate(detections)) {
        // If we already have an active track, compute IoU between the previous track box and this
        // new detection box.
        const float overlap = track.active ? compute_iou(track.box, detection.box) : 0.0f;
        // score = 2 * overlap_with_previous_track + confidence, so it prefers boxes near the
        // previous dog detection.
        const float metric =
            track.active ? (2.0f * overlap + detection.confidence) : detection.confidence;
        if (metric > best_metric) {
            best_metric = metric;
            best        = i;
        }
    }

    // Get the selected detection.
    const Detection& best_detection = detections[best];

    if (track.active) {
        // Add a simple exponential smoothing with alpha = 0.7 so that the new track box = 70% new
        // detection + 30% previous track
        constexpr float alpha = 0.7f;
        // Mixing with previous values.
        track.box.x      = alpha * best_detection.box.x + (1.0F - alpha) * track.box.x;
        track.box.y      = alpha * best_detection.box.y + (1.0F - alpha) * track.box.y;
        track.box.width  = alpha * best_detection.box.width + (1.0F - alpha) * track.box.width;
        track.box.height = alpha * best_detection.box.height + (1.0F - alpha) * track.box.height;
    }
    else
        // If the track was inactive, no smoothing. It starts fresh from the chosen detection.
        track.box = best_detection.box;

    // Reset the missed-frame count, marks the track as active.
    track.confidence = best_detection.confidence;
    track.missed     = 0;
    track.active     = true;
    return track;
}

DogTracker::DogTracker(const ModelConfig& model_config)
    : model_config(model_config), ort_env(ORT_LOGGING_LEVEL_WARNING, "DogTracker")
{
    Ort::SessionOptions session_options;
    const int hardware_threads = static_cast<int>(std::thread::hardware_concurrency());
    session_options.SetIntraOpNumThreads(hardware_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#if USE_CUDA
    if (model_config.use_cuda) {
        try {
            Ort::CUDAProviderOptions cuda_options;
            session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
            std::cerr << "onnxruntime_provider=cuda\n";
        }
        catch (const Ort::Exception& error) {
            std::cerr << "onnxruntime_provider=cuda_unavailable message=\"" << error.what()
                      << "\" falling_back=cpu\n";
        }
    }
    else {
        std::cerr << "onnxruntime_provider=cpu\n";
    }
#else
    // When USE_CUDA is off, the CUDA branch is not compiled.
    // So model_config.use_cuda is not read anywhere inside that preprocessor path.
    // The line (void)this->model_config is to silence an “unused variable/member” warning in the
    // non-CUDA build.
    (void)this->model_config;
    std::cerr << "onnxruntime_provider=cpu\n";
#endif

    ort_session = Ort::Session(ort_env, model_config.model_weights_path.c_str(), session_options);
    // ONNX Runtime’s API allocates the input/output name strings.
    const Ort::AllocatorWithDefaultOptions allocator;
    // input  -> one image batch tensor
    // output -> one detection tensor
    if (ort_session.GetInputCount() != 1 || ort_session.GetOutputCount() != 1) {
        throw std::runtime_error("Expected ONNX model to have exactly one input and one output");
    }
    input_name  = ort_session.GetInputNameAllocated(0, allocator).get();
    output_name = ort_session.GetOutputNameAllocated(0, allocator).get();

    // Get some information about the expected input tensor shapes.
    const auto input_type                  = ort_session.GetInputTypeInfo(0);
    const auto input_info                  = input_type.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> input_shape = input_info.GetShape();
    // [batch, channels, height, width]
    if (input_shape.size() == 4 && input_shape[0] > 0) {
        fixed_batch_size = static_cast<int>(input_shape[0]);
        // Ensure that the batch size denoted on the model weights filename actually
        if (fixed_batch_size != model_config.batch_size) {
            throw std::runtime_error(
                "Model has fixed batch size " + std::to_string(fixed_batch_size) +
                ", but the selected model filename batch size is " +
                std::to_string(model_config.batch_size) + ". Select the matching _bsN model path.");
        }
    }

    // Reserve space for buffers used to build one ONNX input batch.
    frame_values = 3 * static_cast<size_t>(model_config.input_size) *
                   static_cast<size_t>(model_config.input_size);
    const size_t reserved_batch_size = std::max(static_cast<size_t>(fixed_batch_size),
                                                static_cast<size_t>(model_config.batch_size));
    input_buffer.reserve(reserved_batch_size * frame_values);
    padding_sample.reserve(frame_values);
    letterboxes.reserve(static_cast<size_t>(model_config.batch_size));
}

void DogTracker::preprocess(const cv::Mat& frame, std::vector<float>& output, Letterbox* info) const
{
    const int source_width  = frame.cols;
    const int source_height = frame.rows;
    const float scale =
        std::min(static_cast<float>(model_config.input_size) / static_cast<float>(source_width),
                 static_cast<float>(model_config.input_size) / static_cast<float>(source_height));
    const int resized_width =
        static_cast<int>(std::round(static_cast<float>(source_width) * scale));
    const int resized_height =
        static_cast<int>(std::round(static_cast<float>(source_height) * scale));
    const int pad_x = (model_config.input_size - resized_width) / 2;  // 2 for right and left.
    const int pad_y = (model_config.input_size - resized_height) / 2; // 2 for top and bottom.

    // Letterbox preserves aspect ratio while matching the square YOLO tensor shape.
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resized_width, resized_height));
    // RGB (144, 144, 144) for grey pixesl.
    cv::Mat canvas(
        model_config.input_size, model_config.input_size, CV_8UC3, cv::Scalar(114, 114, 114));
    // Copies the resized camera frame into a rectangular region inside the larger square canvas.
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    // Convert HWC uint8 RGB into normalized NCHW float data for ONNX Runtime.
    const size_t sample_offset = output.size();
    // all R values first, then all G values, then all B values.
    const size_t plane_values = frame_values / 3;
    if (sample_offset > output.max_size() - frame_values) {
        throw std::runtime_error("Model input tensor is too large");
    }
    // Append one preprocessed frame into the reusable float tensor buffer.
    output.resize(sample_offset + frame_values);

    // Convert the padded image into the ONNX model’s input tensor layout.
    cv::Mat float_canvas;
    canvas.convertTo(float_canvas, CV_32FC3, 1.0 / 255.0);
    // Create three OpenCV matrix “views” into the existing output vector.
    // planes[0] -> red channel storage
    // planes[1] -> green channel storage
    // planes[2] -> blue channel storage
    // They do not own new memory. They point directly into output.
    std::array<cv::Mat, 3> planes{
        cv::Mat(model_config.input_size,
                model_config.input_size,
                CV_32F,
                output.data() + sample_offset),
        cv::Mat(model_config.input_size,
                model_config.input_size,
                CV_32F,
                output.data() + sample_offset + plane_values),
        cv::Mat(model_config.input_size,
                model_config.input_size,
                CV_32F,
                output.data() + sample_offset + 2 * plane_values),
    };
    // Takes the interleaved image layout and and writes it into the model tensor layout.
    // RGB RGB RGB RGB ... -> RRRR... GGGG... BBBB...
    cv::split(float_canvas, planes.data());

    // write the letterbox metadata back to the caller through the Letterbox* info pointer.
    // That records how much the image was resized and padded, so later detection boxes can be
    // mapped back from model coordinates to original frame coordinates.
    *info = Letterbox{scale, static_cast<float>(pad_x), static_cast<float>(pad_y)};
}

std::vector<Detection> DogTracker::parse_detections(const Ort::Value& output,
                                                    const size_t batch_index,
                                                    const Letterbox& letterbox,
                                                    const int frame_width,
                                                    const int frame_height) const
{
    const auto info                  = output.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = info.GetShape();
    const auto* data                 = output.GetTensorData<float>();

    // shape[0] = number of images in this inference batch.
    // shape[1] = number of detection rows per image.
    // shape[2] = 6 values per detection, [x1, y1, x2, y2, score, class].
    if (shape.size() != 3 || shape[2] != 6) {
        throw std::runtime_error("Expected model output shape [batch, detections, 6].");
    }

    if (shape[0] <= static_cast<int64_t>(batch_index)) {
        throw std::runtime_error("Model output batch is smaller than the input batch.");
    }

    if (shape[1] < 0) {
        throw std::runtime_error("Model output detection dimension is invalid.");
    }

    if (shape[1] > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Model output has too many detections per frame.");
    }

    const int detections_per_frame = static_cast<int>(shape[1]);
    // data points to the start of the whole ONNX output tensor
    // Move the pointer to the start of the selected frame’s detections.
    data += batch_index * static_cast<size_t>(detections_per_frame) * 6;
    std::vector<Detection> detections     = parse_raw_model_outputs(data,
                                                                detections_per_frame,
                                                                letterbox,
                                                                frame_width,
                                                                frame_height,
                                                                model_config.confidence_threshold);
    std::vector<Detection> nms_detections = nms(std::move(detections), model_config.nms_threshold);
    return nms_detections;
}

std::vector<TrackingResult> DogTracker::process_batch(const std::vector<cv::Mat>& frames)
{
    if (frames.empty())
        return {};
    if (frames.size() > static_cast<size_t>(model_config.batch_size))
        throw std::runtime_error("Frame batch size exceeds selected model batch size.");

    // input is the big contiguous tensor buffer passed to ONNX Runtime.
    input_buffer.clear();
    // padding_sample stores a copy/move of the last real preprocessed frame.
    // The model may require exactly batch 32, but the final batch from a video might only have
    // 5 real frames.
    padding_sample.clear();
    // letterboxes stores the preprocessing metadata for each real frame.
    letterboxes.clear();

    const size_t inference_batch_size =
        std::max(static_cast<size_t>(fixed_batch_size), frames.size());

    for (const auto& frame : frames) {
        if (frame.empty()) {
            throw std::runtime_error("Cannot process an empty frame");
        }

        Letterbox lb;
        preprocess(frame, input_buffer, &lb);
        letterboxes.push_back(lb);
    }
    // Pad a partially-filled batch so it still matches the model’s required batch size.
    if (frames.size() < inference_batch_size) {
        // Find the start of the last real preprocessed frame inside input_buffer.
        // last_sample_begin points to the beginning of the final frame’s tensor data.
        const auto last_sample_begin =
            input_buffer.end() - static_cast<std::ptrdiff_t>(frame_values);
        // Copy the last frame’s tensor data into padding_sample.
        // Now padding_sample contains exactly one full preprocessed frame.
        padding_sample.assign(last_sample_begin, input_buffer.end());
        // Loop once for each missing batch slot and append that copied last-frame tensor to the
        // input buffer.
        // real frame 0
        // real frame 1
        // ...
        // real frame 4
        // duplicate of frame 4
        // duplicate of frame 4
        // ...
        // The duplicate padded outputs are ignored.
        for (size_t i = frames.size(); i < inference_batch_size; i++) {
            input_buffer.insert(input_buffer.end(), padding_sample.begin(), padding_sample.end());
        }
    }

    // Some metadata for the forward pass.
    const std::array<int64_t, 4> input_shape{static_cast<int64_t>(inference_batch_size),
                                             3,
                                             model_config.input_size,
                                             model_config.input_size};
    const Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                    input_buffer.data(),
                                                                    input_buffer.size(),
                                                                    input_shape.data(),
                                                                    input_shape.size());
    const char* input_name_ptr    = input_name.c_str();
    const char* output_name_ptr   = output_name.c_str();

    // Actual forward pass.
    const auto outputs = ort_session.Run(
        Ort::RunOptions{nullptr}, &input_name_ptr, &input_tensor, 1, &output_name_ptr, 1);


    // Convert raw ONNX outputs into one TrackingResult per real input frame.
    std::vector<TrackingResult> results;
    results.reserve(frames.size());
    for (const auto& [i, frame] : std::views::enumerate(frames)) {
        std::vector<Detection> detections;
        if (!outputs.empty() && outputs[0].IsTensor()) {
            // Parse the model output for this specific batch slot.
            detections = parse_detections(outputs[0], i, letterboxes[i], frame.cols, frame.rows);
        }
        // Update the tracker’s persistent track member using this frame’s detections.
        // track is stateful. It remembers the previous frame’s target, so this runs in frame order
        // and smooths/continues the dog track over time.
        const Track current_track = update_track(track, detections);
        results.push_back(TrackingResult{std::move(detections), current_track});
    }
    // Return one TrackingResult per real input frame.
    // Any padded batch frames are ignored because this loop only iterates over frames, not over
    // inference_batch_size.
    return results;
}

TrackingResult DogTracker::process(const cv::Mat& frame)
{
    return process_batch({frame}).front();
}