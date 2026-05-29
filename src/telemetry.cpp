#include "telemetry.hpp"

#include <glaze/glaze.hpp>

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>


// Wire format for a frame where the tracker has an active target.
// This struct is intentionally separate from TrackingResult/Track.
// TrackingResult is the internal C++ representation used by the tracker and overlay renderer.
// TrackedTelemetry is the smaller, stable JSON schema that external consumers read. Keeping the
// wire schema here makes it obvious which fields leave the process and what their defaults are.
struct TrackedTelemetry {
        // Timestamp from steady_clock, in milliseconds. This is monotonic process time, not
        // wall-clock Unix time, so it is useful for measuring intervals without being affected by
        // system clock changes.
        int64_t ts_ms = 0;

        // Zero-based index of the real source frame after batching has been flushed. main.cpp
        // increments this once per processed frame, not once per inference batch, so consumers see
        // a simple frame-by-frame stream.
        uint64_t frame = 0;

        // "tracked" tells the consumer that the coordinate fields below are present and meaningful
        // for this frame.
        std::string_view state = "tracked";

        // JSON wants a key named "class", but C++ reserves class as a keyword. The Glaze metadata
        // below maps this member to the JSON key "class".
        std::string_view class_name = "dog";

        // Confidence of the selected/smoothed dog track for this frame.
        float confidence = 0.0f;

        // Normalized center x/y and box size. These are fractions of the source frame dimensions,
        // not pixels, so the robot bridge can use the same controller constants across different
        // camera resolutions.
        float cx     = 0.0f;
        float cy     = 0.0f;
        float width  = 0.0f;
        float height = 0.0f;

        // Normalized image-area fraction. For example, 0.18 means the tracked box occupies about
        // 18% of the frame, which the robot bridge uses as a distance proxy when deciding whether
        // to move forward/back.
        float area = 0.0f;

        // The tracker currently maintains one target, so this is usually 1. It is still sent so the
        // telemetry schema has room for future multi-track support.
        int track_id = 1;
};

// Wire format for a frame where no dog target is active.
// Consumers can treat this as a stop/hold signal because there are no coordinates to chase.
struct LostTelemetry {
        int64_t ts_ms          = 0;
        uint64_t frame         = 0;
        std::string_view state = "lost";

        // Optional serializes as null when no track exists. Keeping the key present makes the
        // lost/tracked schemas easy to handle with simple JSON consumers.
        std::optional<int> track_id = std::nullopt;
};

// Metadata for Glaze compile-time JSON serializer.
// These metadata specializations are the complete mapping between the private telemetry structs
// above and the JSON keys sent over stdout/TCP.
// The specializations live in namespace glz because C++ requires explicit specializations to be
// declared in a namespace where the template can be specialized. The telemetry structs remain
// private to this translation unit.
template <>
struct glz::meta<TrackedTelemetry> {
        static constexpr auto value = object("ts_ms",
                                             &TrackedTelemetry::ts_ms,
                                             "frame",
                                             &TrackedTelemetry::frame,
                                             "state",
                                             &TrackedTelemetry::state,
                                             "class",
                                             &TrackedTelemetry::class_name,
                                             "confidence",
                                             &TrackedTelemetry::confidence,
                                             "cx",
                                             &TrackedTelemetry::cx,
                                             "cy",
                                             &TrackedTelemetry::cy,
                                             "width",
                                             &TrackedTelemetry::width,
                                             "height",
                                             &TrackedTelemetry::height,
                                             "area",
                                             &TrackedTelemetry::area,
                                             "track_id",
                                             &TrackedTelemetry::track_id);
};

template <>
struct glz::meta<LostTelemetry> {
        static constexpr auto value = object("ts_ms",
                                             &LostTelemetry::ts_ms,
                                             "frame",
                                             &LostTelemetry::frame,
                                             "state",
                                             &LostTelemetry::state,
                                             "track_id",
                                             &LostTelemetry::track_id);
};

void StdoutTelemetry::publish(const std::string& line)
{
    // Demo mode is JSON Lines: one complete JSON object plus a newline per frame.
    // Newline framing lets shell tools and Python's for-line iteration consume the stream without a
    // length prefix or a surrounding JSON array.
    std::cout << line << '\n';
}

void TcpTelemetry::accept_loop()
{
    // Only accept incoming connections while the TcpTelemetry is running.
    while (impl->running.load()) {
        sockaddr_in client_addr{};
        socklen_t len = sizeof(client_addr);

        // accept() blocks until a bridge connects. The destructor interrupts this by shutting down
        // the listening socket after clearing running.
        const int client = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &len);

        if (client < 0) {
            // During normal shutdown, accept() fails because the server fd was closed. While still
            // running, a short sleep avoids a tight retry loop on transient socket errors.
            if (impl->running.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            continue;
        }

        // The connected fd is kept open and receives every future telemetry line until send() fails
        // or the TcpTelemetry object is destroyed.
        try {
            // Lock the clients vector because the main thread may be broadcasting telemetry at the
            // same time.
            std::scoped_lock lock(impl->mutex);
            impl->clients.push_back(client);
        }
        catch (...) {
            // If storing the client fails, usually because vector allocation throws, close the
            // socket so it does not leak.
            close(client);
        }
    }
}

// Runtime state hidden behind TcpTelemetry's PIMPL pointer.
// The public header only needs to expose that TcpTelemetry exists; the socket list, mutex, and
// background thread are Linux/POSIX implementation details.
class TcpTelemetry::Impl {
    public:
        // Shared stop flag read by the accept loop and written by the destructor.
        std::atomic<bool> running{false};

        // Accepting clients is separated from publish() so inference can continue emitting
        // telemetry while a robot bridge connects or reconnects.
        std::thread accept_thread;

        // Protects clients because accept_loop() adds fds while publish() walks and removes them.
        // The destructor also uses the same lock before closing clients.
        std::mutex mutex;

        // Connected client sockets. Each telemetry line is broadcast to every fd.
        std::vector<int> clients;
};

TcpTelemetry::TcpTelemetry(std::string host, const int port)
    : host(std::move(host)), port(port), impl(std::make_unique<Impl>())
{
    // IPv4 TCP server socket.
    // The tracker is the server because the robot bridge may start later, restart, or reconnect
    // without forcing inference to restart.
    // Address family: IPv4. Socket type: reliable byte stream (TCP).
    // Protocol 0 indicates usage of the default protocol for this address family and socket type.
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        throw std::runtime_error("Failed to create the TCP socket.");
    }

    // Enable address reuse on this server socket so restarting the telemetry server does not get
    // blocked by the old socket state.
    constexpr int enable = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

    // BCreate an IPv4 bind address using this port and host string. If the host is not a valid
    // numeric IPv4 address, clean up the socket and throw.
    // inet_pton intentionally accepts only numeric IPv4 addresses here, for example 127.0.0.1.
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    // htons converts the port number into the correct network format.
    addr.sin_port = htons(static_cast<uint16_t>(port));
    // Converts the host string into a binary IPv4 address.
    // inet_pton means “internet presentation to network.”
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        close(server_fd);
        server_fd = -1;
        throw std::runtime_error("Invalid telemetry host: " + host);
    }

    // Bind the socket to the configured host/port. Then start listening for TCP clients. If either
    // step fails, clean up and stop with a clear error.
    // bind() attaches the socket to the address built earlier.
    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(server_fd);
        server_fd = -1;
        throw std::runtime_error("Failed to bind telemetry TCP server.");
    }
    if (listen(server_fd, 4) < 0) {
        close(server_fd);
        server_fd = -1;
        throw std::runtime_error("Failed to listen on telemetry TCP server.");
    }

    // Starts the background thread that accepts TCP clients.
    // accept() blocks while waiting for a robot/client to connect so offloading this to a
    // background thread allows the main inference thread to conitnue running.
    try {
        impl->running.store(true);
        impl->accept_thread = std::thread([this] { accept_loop(); });
    }
    // Catches any exception. std::thread(...) can throw, commonly std::system_error, if the OS
    // cannot create a new thread. Without this catch, if thread creation threw after the socket was
    // opened/listening, the constructor would fail but the raw socket fd could leak because the
    // destructor is not called for an object whose constructor did not finish.
    catch (...) {
        impl->running.store(false);
        close(server_fd);
        server_fd = -1;
        throw;
    }

    std::cout << "TcpTelemetry host=" << host << " port=" << port << std::endl;
}

TcpTelemetry::~TcpTelemetry()
{
    // Tell accept_loop() to stop before shutting down the server socket.
    impl->running.store(false);
    if (server_fd >= 0) {
        // shutdown() wakes a thread blocked in accept(), and close() releases the fd.
        shutdown(server_fd, SHUT_RDWR);
        close(server_fd);
    }

    // Wait for the accept thread before closing client sockets so no thread can add another fd
    // after destructor cleanup begins.
    if (impl->accept_thread.joinable()) {
        impl->accept_thread.join();
    }
    server_fd = -1;

    // Close any connected clients. The vector itself is owned by impl and will be destroyed
    // automatically after the descriptors are released.
    std::scoped_lock lock(impl->mutex);
    for (const int& fd : impl->clients) {
        close(fd);
    }
}

void TcpTelemetry::publish(const std::string& line)
{
    // TCP has no message boundaries, so telemetry uses JSON Lines framing. Each payload gets
    // exactly one trailing newline and receivers read line-by-line.
    const std::string payload = line + "\n";

    // Hold the client lock for the full broadcast so accept_loop() cannot mutate the vector while
    // we are iterating and possibly erasing dead sockets.
    std::scoped_lock lock(impl->mutex);
    auto it = impl->clients.begin();
    while (it != impl->clients.end()) {
        // If send() succeeds, wrote contains the number of bytes actually written to the socket.
        // MSG_NOSIGNAL prevents a disconnected client from raising SIGPIPE and killing the tracker
        // process. A send failure means that client is gone.
        const ssize_t wrote = send(*it, payload.data(), payload.size(), MSG_NOSIGNAL);
        // Closed client would lead to wrote bytes < 0, so we close that fd and remove it from the
        // vector of client fds.
        if (wrote < 0) {
            close(*it);
            // erase(it) returns the next valid iterator.
            it = impl->clients.erase(it);
        }
        else {
            ++it;
        }
    }
}

void NullTelemetry::publish(const std::string&)
{
    // Render mode can use the exact same publishing path as live modes while this sink
    // intentionally discards the line. That keeps the frame loop simple and avoids sprinkling
    // telemetry conditionals through main.cpp.
}

std::unique_ptr<TelemetrySink> make_telemetry_sink(const Configuration& configuration)
{
    // Helper function for stream_inference.cpp so the frame-processing loops can depend only on the
    // TelemetrySink interface. Configuration selection stays out of those loops.
    switch (configuration.telemetry_mode) {
        case TelemetryMode::Stdout:
            return std::make_unique<StdoutTelemetry>();
        case TelemetryMode::TCP:
            return std::make_unique<TcpTelemetry>(configuration.telemetry_host,
                                                  configuration.telemetry_port);
        case TelemetryMode::None:
            return std::make_unique<NullTelemetry>();
    }
}

std::string make_telemetry_json(const int64_t ts_ms,
                                const uint64_t frame_index,
                                const TrackingResult& result,
                                const int frame_width,
                                const int frame_height)
{
    // The tracker emits one event per processed source frame. If no track is active, send the
    // compact "lost" event instead of stale coordinates from the previous frame. The robot might
    // interpret this as a stop command.
    if (!result.track.active) {
        const LostTelemetry event{.ts_ms = ts_ms, .frame = frame_index};
        std::string out;
        // glz::write_json(...) returns an error code/status.
        // If serialization succeeds, that status is “empty” / false-like.
        // If it fails, it is true-like.
        if (const auto& ec = glz::write_json(event, out); ec) {
            throw std::runtime_error("Glaze failed to serialize lost telemetry.");
        }
        return out;
    }

    // Convert the smoothed source-pixel box into normalized frame coordinates.
    // result.track.box is already in the original frame coordinate system by the time telemetry
    // sees it; these divisions make the output resolution-agnostic.
    const float cx =
        (result.track.box.x + result.track.box.width / 2.0F) / static_cast<float>(frame_width);
    const float cy =
        (result.track.box.y + result.track.box.height / 2.0F) / static_cast<float>(frame_height);
    const float width  = result.track.box.width / static_cast<float>(frame_width);
    const float height = result.track.box.height / static_cast<float>(frame_height);

    // Build the tracked event explicitly rather than aggregate-initializing every field so the
    // schema defaults above remain the single source for constants like state="tracked" and
    // class="dog".
    TrackedTelemetry event;
    event.ts_ms      = ts_ms;
    event.frame      = frame_index;
    event.confidence = result.track.confidence;
    event.cx         = cx;
    event.cy         = cy;
    event.width      = width;
    event.height     = height;
    event.area       = width * height;
    event.track_id   = result.track.id;

    std::string out;
    // Serialization errors should be impossible for this simple POD-like schema, but throwing keeps
    // the caller from publishing malformed or partial telemetry if the schema is changed
    // incorrectly in the future.
    if (const auto& ec = glz::write_json(event, out); ec) {
        throw std::runtime_error("Glaze failed to serialize tracked telemetry");
    }
    return out;
}

int64_t monotonic_time_ms()
{
    // steady_clock is monotonic, so elapsed-time calculations stay sane even if NTP or the user
    // adjusts the system wall clock while the tracker is running.
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}