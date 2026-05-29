#ifndef TURBO_PI_DOG_TRACKER_VIDEO_HPP
#define TURBO_PI_DOG_TRACKER_VIDEO_HPP
#pragma once

#include "tracker.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>


namespace fs = std::filesystem;

// Opens a live camera device path with OpenCV VideoCapture.
cv::VideoCapture open_live_capture(const fs::path& device_path);
cv::VideoCapture open_video_file(const fs::path& path);

// Creates a writer for the annotated README/demo video. Uses an MP4-family codec for .mp4/.mov.
cv::VideoWriter open_overlay_writer(const fs::path& path, double fps, const cv::Size& size);

// Draws the active dog track and a compact coordinate readout in the top-left corner.
void draw_tracking_overlay(cv::Mat& frame, const TrackingResult& result, uint64_t frame_index);


#endif // TURBO_PI_DOG_TRACKER_VIDEO_HPP
