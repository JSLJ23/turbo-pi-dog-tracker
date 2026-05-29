#include "cli.hpp"

#include <cxxopts.hpp>

#include <charconv>
#include <filesystem>
#include <string>
#include <string_view>


namespace fs = std::filesystem;

template <typename T>
static T parse_number(const std::string_view text, const std::string_view option)
{
    T value{};
    const char* begin = text.data();
    const char* end   = text.data() + text.size();
    // Analyzes the character sequence [first, last) for a pattern described below.
    // If no characters match the pattern or if the value obtained by parsing the matched
    // characters is not representable in the type of value, value is unmodified, otherwise the
    // characters matching the pattern are interpreted as a text representation of an arithmetic
    // value, which is stored in value.
    const auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end) {
        throw std::runtime_error("Invalid numeric value for " + std::string(option) + ": " +
                                 std::string(text));
    }
    return value;
}

static RunMode parse_run_mode(const std::string_view text)
{
    if (text == "demo" || text == "Demo")
        return RunMode::Demo;
    if (text == "server" || text == "Server")
        return RunMode::Server;
    if (text == "render" || text == "Render")
        return RunMode::Render;
    throw std::runtime_error("Unknown run mode, --mode must be demo, server, or render");
}

static cxxopts::Options make_options(const char* argv0)
{
    cxxopts::Options options(
        argv0 == nullptr ? "turbo_pi_dog_tracker" : argv0,
        "Modes:\n"
        "  demo    Live camera preview with JSONL telemetry on stdout (default)\n"
        "  server  Live camera with TCP JSONL telemetry\n"
        "  render  Read --input-video, draw overlays, write --output-video");
    options.custom_help("[options]");
    options.add_options()
        // clang-format off
    ("model", "ONNX model path (default: models/yolo26n.onnx)", cxxopts::value<fs::path>(), "PATH")
    ("camera", "Live camera index (default: 0)", cxxopts::value<int>(), "INT")
    ("input-video", "MP4/MOV/video input for render mode", cxxopts::value<fs::path>(), "PATH")
    ("output-video", "Annotated video output for render mode", cxxopts::value<fs::path>(), "PATH")
    ("confidence", "Dog confidence threshold, 0..1 (default: 0.5)", cxxopts::value<float>(), "FLOAT")
    ("nms", "NMS IoU threshold, 0..1 (default: 0.5)", cxxopts::value<float>(), "FLOAT")
    ("input-size", "Square model input size, multiple of 32 (default: 640)", cxxopts::value<int>(), "INT")
    ("h,help", "Show this help");
    // clang-format on
    return options;
}


static std::optional<int> batch_size_from_model_path(const fs::path& model_path)
{
    const std::string stem = model_path.stem().string();
    // _bs suffix denotes the fixed batch size in the ONNX exported mode.
    const size_t marker = stem.rfind("_bs");
    if (marker == std::string::npos) {
        return std::nullopt;
    }

    // model_path = "models/yolo26n_bs32.onnx"
    //  stem       = "yolo26n_bs32"
    //  marker     = 7 (position of "_bs")
    //  start      = 7 + 3 = 10
    //  length     = 12 - 7 - 3 = 2
    //  suffix     = "32"
    const std::string_view suffix(stem.data() + marker + 3, stem.size() - marker - 3);
    if (suffix.empty()) {
        throw std::runtime_error("Model filename batch suffix must end with digits after _bs: " +
                                 model_path.string());
    }
    return parse_number<int>(suffix, "model filename batch suffix");
}


static void apply_telemetry_from_environment(Configuration& config)
{
    // Sets the host IP and port from enviroment variables
    if (const char* host = std::getenv("TELEMETRY_HOST"); host != nullptr && host[0] != '\0') {
        config.telemetry_host = host;
    }
    if (const char* port = std::getenv("TELEMETRY_PORT"); port != nullptr && port[0] != '\0') {
        config.telemetry_port = parse_number<int>(port, "TELEMETRY_PORT");
    }
}

std::string show_usage(const char* argv0)
{
    return make_options(argv0).help();
}

static void validate_configuration(Configuration& config)
{
    if (config.show_help)
        return;

    switch (config.run_mode) {
        case RunMode::Demo:
            config.telemetry_mode = TelemetryMode::Stdout;
            break;
        case RunMode::Server:
            config.telemetry_mode = TelemetryMode::TCP;
            break;
        case RunMode::Render:
            config.telemetry_mode = TelemetryMode::None;
            break;
    }

    if (const auto model_batch_size = batch_size_from_model_path(config.model_weights_path)) {
        config.batch_size = *model_batch_size;
    }

    if (config.input_size <= 0 || config.input_size % 32 != 0) {
        throw std::runtime_error("--input-size must be a positive multiple of 32");
    }

    if (config.batch_size <= 0) {
        throw std::runtime_error("Model filename batch suffix must be positive");
    }

    if (config.confidence_threshold < 0.0f || config.confidence_threshold > 1.0f) {
        throw std::runtime_error("--confidence must be between 0 and 1");
    }

    if (config.nms_threshold < 0.0f || config.nms_threshold > 1.0f) {
        throw std::runtime_error("--nms must be between 0 and 1");
    }

    if (config.telemetry_port <= 0 || config.telemetry_port > 65535) {
        throw std::runtime_error("Telemetry port must be between 1 and 65535");
    }

    if (config.run_mode != RunMode::Render && config.camera_source < 0) {
        throw std::runtime_error("--camera must be a non-negative camera index");
    }

    if (config.run_mode == RunMode::Render) {
        if (config.input_video_path.empty()) {
            throw std::runtime_error("render mode requires --input-video PATH");
        }
        if (!fs::is_regular_file(config.input_video_path)) {
            throw std::runtime_error("supplied --input-video PATH is not a file");
        }
        if (config.output_video_path.empty()) {
            throw std::runtime_error("render mode requires --output-video PATH");
        }
    }
}

Configuration parse_args(const int argc, const char* argv[])
{
    if (argc < 2) {
        throw std::runtime_error("The first argument must be mode: demo, server, or render.");
    }

    const RunMode run_mode = parse_run_mode(argv[1]);
    Configuration configuration{.run_mode = run_mode};
    apply_telemetry_from_environment(configuration);

    std::vector<const char*> option_args;
    option_args.reserve(static_cast<size_t>(argc - 1));
    option_args.push_back(argv[0]);
    for (int i = 2; i < argc; ++i) {
        option_args.push_back(argv[i]);
    }

    const int option_argc = static_cast<int>(option_args.size());
    const cxxopts::ParseResult parse_result =
        make_options(argv[0]).parse(option_argc, option_args.data());

    configuration.show_help = parse_result.count("help") != 0;
    if (parse_result.count("model") != 0)
        configuration.model_weights_path = parse_result["model"].as<fs::path>();
    if (parse_result.count("camera") != 0)
        configuration.camera_source = parse_result["camera"].as<int>();
    if (parse_result.count("input-video") != 0)
        configuration.input_video_path = parse_result["input-video"].as<fs::path>();
    if (parse_result.count("output-video") != 0)
        configuration.output_video_path = parse_result["output-video"].as<fs::path>();
    if (parse_result.count("confidence") != 0)
        configuration.confidence_threshold = parse_result["confidence"].as<float>();
    if (parse_result.count("nms") != 0)
        configuration.nms_threshold = parse_result["nms"].as<float>();
    if (parse_result.count("input-size") != 0)
        configuration.input_size = parse_result["input-size"].as<int>();

    validate_configuration(configuration);

    return configuration;
}

std::string to_string(const RunMode& run_mode)
{
    switch (run_mode) {
        case RunMode::Demo:
            return "demo";
        case RunMode::Server:
            return "server";
        case RunMode::Render:
            return "render";
    }
    return "unknown";
}

std::string to_string(const TelemetryMode& telemetry_mode)
{
    switch (telemetry_mode) {
        case TelemetryMode::Stdout:
            return "stdout";
        case TelemetryMode::TCP:
            return "tcp";
        case TelemetryMode::None:
            return "none";
    }
    return "unknown";
}