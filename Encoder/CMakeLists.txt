cmake_minimum_required(VERSION 3.10)
project(heatmap_encoder C)

set(CMAKE_C_STANDARD 99)

if (WIN32 AND NOT MSVC)
    include_directories(SYSTEM C:/msys64/mingw64/include)
    link_directories(C:/msys64/mingw64/lib)
endif()

add_executable(encoder encoder.c)

if(WIN32 AND NOT MSVC)
    target_link_libraries(encoder PRIVATE
        avformat
        avcodec
        avutil
        swscale
        x264
    )
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(AVCODEC REQUIRED libavcodec)
    pkg_check_modules(AVFORMAT REQUIRED libavformat)
    pkg_check_modules(AVUTIL REQUIRED libavutil)
    pkg_check_modules(SWSCALE REQUIRED libswscale)
    pkg_check_modules(X264 REQUIRED x264)

    target_include_directories(encoder PRIVATE
        ${AVCODEC_INCLUDE_DIRS}
        ${AVFORMAT_INCLUDE_DIRS}
        ${AVUTIL_INCLUDE_DIRS}
        ${SWSCALE_INCLUDE_DIRS}
        ${X264_INCLUDE_DIRS}
    )

    target_link_libraries(encoder PRIVATE
        ${AVCODEC_LIBRARIES}
        ${AVFORMAT_LIBRARIES}
        ${AVUTIL_LIBRARIES}
        ${SWSCALE_LIBRARIES}
        ${X264_LIBRARIES}
    )
endif()