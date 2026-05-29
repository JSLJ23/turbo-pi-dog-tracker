#include "cli.hpp"
#include "stream_inference.hpp"

#include <iostream>
#include <stdexcept>


int main(const int argc, const char* argv[])
{
    try {
        const Configuration configuration = parse_args(argc, argv);
        if (configuration.show_help) {
            std::cout << show_usage(argv[0]) << std::endl;
            return 0;
        }

        std::cout << "Run mode: " << to_string(configuration.run_mode) << std::endl;
        std::cout << "Telemetry: " << to_string(configuration.telemetry_mode) << std::endl;
        std::cout << "Batch size: " << configuration.batch_size << std::endl;

        switch (configuration.run_mode) {
            case RunMode::Demo:
            case RunMode::Server:
                return run_live(configuration);
            case RunMode::Render:
                return run_render(configuration);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::cerr << show_usage(argv[0]) << std::endl;
    }

    return 0;
}