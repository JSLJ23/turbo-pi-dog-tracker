cmake_minimum_required(VERSION 4.1.2)

option(USE_CUDA "Enable ONNX Runtime CUDA execution provider" OFF)
option(USE_TENSORRT "Enable ONNX Runtime TensorRT execution provider" OFF)

if(USE_TENSORRT AND NOT USE_CUDA)
  message(FATAL_ERROR "USE_TENSORRT requires USE_CUDA=ON")
endif()

if(USE_CUDA)
  project(turbo_pi_dog_tracker LANGUAGES CXX CUDA)
  enable_language(CUDA)
else()
  project(turbo_pi_dog_tracker LANGUAGES CXX)
endif()

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

if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(
  ${PROJECT_NAME} src/cli.cpp src/main.cpp src/stream_inference.cpp
                  src/telemetry.cpp src/tracker.cpp src/video.cpp)

target_compile_definitions(
  ${PROJECT_NAME} PRIVATE USE_CUDA=$<BOOL:${USE_CUDA}>
                          USE_TENSORRT=$<BOOL:${USE_TENSORRT}>)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE opencv_core
          opencv_highgui
          opencv_imgproc
          opencv_videoio
          cxxopts::cxxopts
          onnxruntime::onnxruntime
          glaze::glaze
          Threads::Threads)

if(USE_CUDA)
  target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
