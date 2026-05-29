#include "video.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>


namespace fs = std::filesystem;

static std::string lower_extension(const fs::path& path)
{
    std::string ext = path.extension().string();
    std::ranges::transform(
        ext, ext.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

struct CodecCandidate {
        const char* name;
        int fourcc;
        int api_preference;
};

static std::vector<CodecCandidate> preferred_codecs(const fs::path& path)
{
    // Helper function that picks video codecs based on the output file path.
    // MP4-ish file extension -> prefer H.264, fall back to mp4v
    // anything else          -> use MJPG
    const std::string ext = lower_extension(path);
    if (ext == ".mp4" || ext == ".mov" || ext == ".m4v") {
        // fourcc means “four-character code.”
        // OpenCV uses it to choose the video encoding format.
        // avc1 selects H.264/AVC for MP4-family containers when FFmpeg supports it.
        return {
            {"h264-avc1",   cv::VideoWriter::fourcc('a', 'v', 'c', '1'), cv::CAP_FFMPEG},
            {"mp4v-ffmpeg", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), cv::CAP_FFMPEG},
            {"mp4v",        cv::VideoWriter::fourcc('m', 'p', '4', 'v'), cv::CAP_ANY   },
        };
    }
    // For every other extension, fall back to MJPG, Motion JPEG.
    return {
        {"mjpg", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), cv::CAP_ANY}
    };
}

static void draw_label(cv::Mat& frame,
                       const std::string& text,
                       const cv::Point origin,
                       const double scale,
                       const cv::Scalar& bg,
                       const cv::Scalar& fg)
{
    // Helper to draw a text label with a filled background rectangle behind it.
    // Modifies the input frame in place.
    int baseline            = 0;
    constexpr int thickness = 1;
    // Update baseline, which is extra vertical space needed below the text baseline for characters
    // like g, y, or p.
    const cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    // Build a rectangle behind the text.
    cv::Rect box(origin.x - 4,
                 origin.y - text_size.height - 6,
                 text_size.width + 8,
                 text_size.height + baseline + 8);
    // Clip the label box to the image bounds so the background rectangle does not extend outside
    // the frame. &= here is OpenCV rectangle intersection assignment.
    box &= cv::Rect(0, 0, frame.cols, frame.rows);
    // Draw the filled background rectangle.
    cv::rectangle(frame, box, bg, cv::FILLED);
    // Draw the actual text on top of the rectangle.
    cv::putText(frame, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv::LINE_AA);
}

cv::VideoCapture open_live_capture(const int source_index)
{
    // Helper function to get a cv::VideoCapture from a camera source index.
    cv::VideoCapture capture(source_index, cv::CAP_V4L2);
    if (!capture.isOpened()) {
        throw std::runtime_error("Failed to open camera index: " + std::to_string(source_index));
    }
    return capture;
}

cv::VideoCapture open_video_file(const fs::path& path)
{
    // Helper function to get a cv::VideoCapture from a input file.
    cv::VideoCapture capture(path);
    if (!capture.isOpened()) {
        throw std::runtime_error("Failed to open input video: " + path.string());
    }
    return capture;
}

cv::VideoWriter open_overlay_writer(const fs::path& path, double fps, const cv::Size& size)
{
    // Helper function to get the cv::VideoWriter based on the output file extension.
    if (fps <= 0.0 || !std::isfinite(fps)) {
        fps = 30.0;
    }

    for (const auto& [name, fourcc, api_preference] : preferred_codecs(path)) {
        cv::VideoWriter writer(path.string(), api_preference, fourcc, fps, size, true);
        if (writer.isOpened()) {
            std::cout << "output_video_encoder=" << name << std::endl;
            return writer;
        }
    }

    throw std::runtime_error("Failed to open output video writer: " + path.string());
}

void draw_tracking_overlay(cv::Mat& frame, const TrackingResult& result, uint64_t frame_index)
{
    // Some predefined colours in BGR format.
    const cv::Scalar panel_bg(20, 20, 20);
    const cv::Scalar text_fg(240, 240, 240);
    const cv::Scalar dog_green(60, 220, 120);
    const cv::Scalar lost_red(70, 70, 230);

    // Creates a string builder for the header text.
    std::ostringstream header;
    header << "frame " << frame_index;
    draw_label(frame, header.str(), {12, 28}, 0.62, panel_bg, text_fg);

    if (!result.track.active) {
        draw_label(frame, "dog: lost", {12, 58}, 0.62, lost_red, text_fg);
        return;
    }

    // Take the tracked dog box and clips it to the image bounds, & means rectangle intersection.
    const cv::Rect box =
        result.track.box &
        cv::Rect2f(0.0f, 0.0f, static_cast<float>(frame.cols), static_cast<float>(frame.rows));
    // Draw the bounding box.
    cv::rectangle(frame, box, dog_green, 2, cv::LINE_AA);

    // Compute normalized center X and Y of the dog box.
    const float cx =
        (result.track.box.x + result.track.box.width / 2.0f) / static_cast<float>(frame.cols);
    const float cy =
        (result.track.box.y + result.track.box.height / 2.0f) / static_cast<float>(frame.rows);
    // Compute normalized box width and height.
    const float width  = result.track.box.width / static_cast<float>(frame.cols);
    const float height = result.track.box.height / static_cast<float>(frame.rows);

    // String builder for the label above/near the dog box.
    std::ostringstream label;
    label << std::fixed << std::setprecision(2) << "dog " << result.track.confidence
          << "  cx=" << cx << " cy=" << cy;
    // Draw the dog label near the top-left of the box.
    draw_label(frame,
               label.str(),
               {std::max(8, box.x), std::max(24, box.y - 8)},
               0.55,
               dog_green,
               cv::Scalar(0, 0, 0));

    // String builder for the coordinate readout.
    std::ostringstream coordinates;
    // Draw that coordinate readout under the frame label in the top-left corner.
    coordinates << std::fixed << std::setprecision(3) << "box x=" << result.track.box.x
                << " y=" << result.track.box.y << " width=" << result.track.box.width
                << " height=" << result.track.box.height << "  norm width=" << width
                << " height=" << height;
    draw_label(frame, coordinates.str(), {12, 58}, 0.55, panel_bg, text_fg);
}