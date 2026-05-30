#include "stream_inference.hpp"
#include "cli.hpp"
#include "telemetry.hpp"
#include "tracker.hpp"
#include "video.hpp"


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>


// Global flag used by the Ctrl-C signal handler.
// std::sig_atomic_t is safe to write from a signal handler.
volatile std::sig_atomic_t g_live_stop_requested{0};

// Signal handler for SIGINT. When Ctrl-C happens, it marks stop requested.
static void request_live_stop(int /*signal*/)
{
    g_live_stop_requested = 1;
}

// Convenience wrapper that turns the flag into a bool.
static bool live_stop_requested()
{
    return g_live_stop_requested != 0;
}

// The purpose of this handler is to allow the program to shut down cleanly when the user presses
// Control-C instead of being terminated immediately by the operating system.
class ScopedLiveSignalHandler {
    public:
        ScopedLiveSignalHandler()
        {
            g_live_stop_requested = 0;
            previous_handler      = std::signal(SIGINT, request_live_stop);
        }

        ~ScopedLiveSignalHandler()
        {
            if (previous_handler != SIG_ERR) {
                std::signal(SIGINT, previous_handler);
            }
        }

    private:
        using SignalHandler            = void (*)(int);
        SignalHandler previous_handler = SIG_DFL;
};

static ModelConfig make_model_config(const Configuration& configuration)
{
    return ModelConfig{
        .model_weights_path   = configuration.model_weights_path,
        .confidence_threshold = configuration.confidence_threshold,
        .nms_threshold        = configuration.nms_threshold,
        .input_size           = configuration.input_size,
        .batch_size           = configuration.batch_size,
#if USE_CUDA
        .use_cuda = true,
#else
        .use_cuda = false,
#endif
    };
}

static void publish_frame(TelemetrySink& sink,
                          const uint64_t frame_index,
                          const TrackingResult& result,
                          const cv::Mat& frame)
{
    sink.publish(
        make_telemetry_json(monotonic_time_ms(), frame_index, result, frame.cols, frame.rows));
}

static cv::Size requested_live_capture_size(const Configuration& configuration)
{
    const int width = configuration.input_size > 640 ? 1280 : 640;
    return {width, width * 9 / 16};
}


int run_live(const Configuration& configuration)
{
    // Enables Ctrl-C handling for the duration of live mode.
    ScopedLiveSignalHandler signal_handler;
    // Load the ONNX model and prepares inference.
    DogTracker tracker(make_model_config(configuration));
    // Open the configured camera index.
    cv::VideoCapture capture = open_live_capture(configuration.camera_source);
    // Ensure that the camera capture width matches the input size for the model.
    const cv::Size requested_frame_size = requested_live_capture_size(configuration);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, requested_frame_size.width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, requested_frame_size.height);
    // Creates stdout telemetry for demo mode or TCP telemetry for server mode.
    const auto telemetry = make_telemetry_sink(configuration);

    // Only demo mode opens a preview window.
    const bool show_preview = configuration.run_mode == RunMode::Demo;
    constexpr std::string_view preview_window{"Dog Tracker"};
    if (show_preview) {
        cv::namedWindow(preview_window.data(), cv::WINDOW_NORMAL);
        cv::resizeWindow(
            preview_window.data(), requested_frame_size.width, requested_frame_size.height);
    }

    // frame receives camera reads. frames accumulates a batch for inference.
    cv::Mat frame;
    std::vector<cv::Mat> frames;
    frames.reserve(configuration.batch_size);
    // Tracks output frame number and local stop state.
    uint64_t frame_index = 0;
    bool stop_requested  = false;

    // Lambda that processes whatever frames are currently batched.
    auto run_inference_pipeline_on_batch = [&]() {
        // Runs preprocessing, ONNX inference, detection parsing, and tracking.
        const auto results = tracker.process_batch(frames);
        // Handles each real frame/result in order.
        for (size_t i = 0; i < frames.size(); i++) {
            // Stops promptly if Ctrl-C was pressed.
            if (live_stop_requested()) {
                stop_requested = true;
                break;
            }
            // Publishes JSON telemetry for this frame.
            publish_frame(*telemetry, frame_index, results[i], frames[i]);
            // In demo mode, draw overlay, show frame, and read a key press.
            if (show_preview) {
                draw_tracking_overlay(frames[i], results[i], frame_index);
                cv::imshow(preview_window.data(), frames[i]);
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q' || key == 'Q' || live_stop_requested()) {
                    stop_requested = true;
                }
            }
            // Advance frame count and clear the batch after processing.
            frame_index++;
            if (stop_requested) {
                break;
            }
        }
        frames.clear();
    };

    // Main live loop:
    // Keep reading camera frames until stopped or camera read fails.
    while (!stop_requested && !live_stop_requested() && capture.read(frame)) {
        // Skip invalid/empty frames.
        if (frame.empty()) {
            continue;
        }

        // Store a copy because OpenCV may reuse frame storage on the next read.
        frames.push_back(frame.clone());
        // Run inference when the batch is full.
        if (frames.size() == static_cast<size_t>(configuration.batch_size)) {
            run_inference_pipeline_on_batch();
        }
    }

    // Process leftover frames at shutdown.
    if (!frames.empty() && !stop_requested && !live_stop_requested()) {
        run_inference_pipeline_on_batch();
    }

    // Close the preview window and report success.
    if (show_preview) {
        cv::destroyWindow(preview_window.data());
    }
    return 0;
}

int run_render(const Configuration& configuration)
{
    // Load the ONNX model and prepares inference.
    DogTracker tracker(make_model_config(configuration));
    // Open the video file based on the file path.
    cv::VideoCapture capture = open_video_file(configuration.input_video_path);
    // Render mode usually uses null telemetry.
    const auto telemetry = make_telemetry_sink(configuration);

    cv::Mat frame;
    // Read the first frame so we know the video is valid and know its size.
    if (!capture.read(frame) || frame.empty()) {
        throw std::runtime_error("Input video has no readable frames: " +
                                 configuration.input_video_path.string());
    }

    // Get input FPS and create the output video writer.
    const double fps = capture.get(cv::CAP_PROP_FPS);
    auto writer      = open_overlay_writer(configuration.output_video_path, fps, frame.size());

    // frame receives video file reads. frames accumulates a batch for inference.
    std::vector<cv::Mat> frames;
    frames.reserve(configuration.batch_size);
    // Start the first batch with the first frame already read.
    frames.push_back(frame.clone());
    uint64_t frame_index = 0;

    // Lambda that processes whatever frames are currently batched.
    auto run_inference_pipeline_on_batch = [&]() {
        const auto results = tracker.process_batch(frames);
        for (size_t i = 0; i < frames.size(); i++) {
            // Optionally publish telemetry, draw overlay, write annotated frame, increment count.
            publish_frame(*telemetry, frame_index, results[i], frames[i]);
            draw_tracking_overlay(frames[i], results[i], frame_index);
            writer.write(frames[i]);
            frame_index++;
        }
        frames.clear();
    };

    if (frames.size() == static_cast<size_t>(configuration.batch_size)) {
        run_inference_pipeline_on_batch();
    }

    // Read the rest of the video.
    while (capture.read(frame) && !frame.empty()) {
        frames.push_back(frame.clone());
        // Batch frames and process full batches.
        if (frames.size() == static_cast<size_t>(configuration.batch_size)) {
            run_inference_pipeline_on_batch();
        }
    }

    // Process final partial batch.
    if (!frames.empty()) {
        run_inference_pipeline_on_batch();
    }

    // Log output path and number of frames rendered, then return success.
    std::cout << "rendered_video=" << configuration.output_video_path.string()
              << " frames=" << frame_index << '\n';
    return 0;
}