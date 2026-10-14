cmake_minimum_required(VERSION 3.16)
project(heatmap_encoder C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ENCODER_DEPS REQUIRED IMPORTED_TARGET
    libavcodec libavformat libavutil libswscale x264)

add_executable(encoder encoder.c)
target_link_libraries(encoder PRIVATE PkgConfig::ENCODER_DEPS)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(encoder PRIVATE -Wall -Wextra)
endif()
