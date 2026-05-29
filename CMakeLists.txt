cmake_minimum_required(VERSION 3.25.0)

option(USE_CUDA "Enable ONNX Runtime CUDA execution provider" OFF)

if (USE_CUDA)
    project(turbo_pi_dog_tracker LANGUAGES CXX CUDA)
    enable_language(CUDA)
else ()
    project(turbo_pi_dog_tracker LANGUAGES CXX)
endif ()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

unset(CMAKE_C_COMPILER)

find_package(cxxopts REQUIRED CONFIG)
find_package(OpenSSL REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc videoio)
find_package(onnxruntime REQUIRED CONFIG)
find_package(glaze REQUIRED)
find_package(Threads REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME}
        src/cli.cpp
        src/main.cpp
        src/stream_inference.cpp
        src/telemetry.cpp
        src/tracker.cpp
        src/video.cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA=$<BOOL:${USE_CUDA}>)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)

target_link_libraries(${PROJECT_NAME} PRIVATE
        opencv_core
        opencv_imgproc
        opencv_videoio
        cxxopts::cxxopts
        onnxruntime::onnxruntime
        glaze::glaze
        Threads::Threads
)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
