#ifndef TURBO_PI_DOG_TRACKER_TELEMETRY_HPP
#define TURBO_PI_DOG_TRACKER_TELEMETRY_HPP
#pragma once

#include "cli.hpp"
#include "tracker.hpp"

#include <memory>
#include <string>


// Abstract base class for telemetry output.
// Actual TelemetrySinks decide whether the line goes to stdout, TCP, or nowhere.
class TelemetrySink {
    public:
        virtual ~TelemetrySink() = default;

        // Publish one already-serialized telemetry object.
        // Callers pass the JSON text without a trailing newline, the sink that stream JSONL add
        // its own delimiter.
        virtual void publish(const std::string& line) = 0;
};

// Demo-mode sink.
// Prints one JSON object per frame to stdout.
class StdoutTelemetry final : public TelemetrySink {
    public:
        void publish(const std::string& line) override;
};

// Server-mode sink.
// Owns a small IPv4 TCP server and broadcasts each JSONL frame to every connected client.
class TcpTelemetry final : public TelemetrySink {
    public:
        TcpTelemetry(std::string host, int port);
        ~TcpTelemetry() override;

        void publish(const std::string& line) override;

    private:
        void accept_loop();

        std::string host;
        int port      = 0;
        int server_fd = -1;

        // Implementation details are hidden so consumers of this header do not need
        // socket/thread/vector declarations beyond the stable public class shape.
        class Impl;
        std::unique_ptr<Impl> impl;
};

// Disabled-output sink.
// Useful for render mode, where the frame loop can still call publish() but no telemetry leaves
// the process.
class NullTelemetry final : public TelemetrySink {
    public:
        void publish(const std::string& line) override;
};

// Choose the sink matching Configuration::telemetry_mode.
std::unique_ptr<TelemetrySink> make_telemetry_sink(const Configuration& configuration);

// Convert the current tracking result into the public telemetry JSON schema.
// Coordinates are normalized to frame_w/frame_h, so consumers receive values in frame fractions
// rather than pixels.
std::string make_telemetry_json(int64_t ts_ms,
                                uint64_t frame_index,
                                const TrackingResult& result,
                                int frame_width,
                                int frame_height);

// Monotonic milliseconds used for telemetry timestamps. This is intentionally not wall-clock time
// so it is stable for elapsed-time calculations during a run.
int64_t monotonic_time_ms();

#endif // TURBO_PI_DOG_TRACKER_TELEMETRY_HPP
