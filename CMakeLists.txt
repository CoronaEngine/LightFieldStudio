cmake_minimum_required(VERSION 3.24)

project(LightFieldStudio VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(lightfieldstudio_config INTERFACE)
target_compile_features(lightfieldstudio_config INTERFACE cxx_std_20)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/backend/Vision/CMakeLists.txt")
    set(VISION_BUILD_APPLICATIONS OFF CACHE BOOL
        "Build standalone VisionBackend applications" FORCE)
    set(VISION_BUILD_BACKEND_DENOISERS OFF CACHE BOOL
        "Build VisionBackend denoiser plugins" FORCE)
    set(VISION_BUILD_HOTFIX_TOOLING OFF CACHE BOOL
        "Build VisionBackend hotfix tooling" FORCE)
    set(VISION_EXCLUDED_PLUGINS "framebuffer:lightfield" CACHE STRING
        "VisionBackend plugins not used by LightFieldStudio" FORCE)
    add_subdirectory(backend/Vision)
else()
    message(FATAL_ERROR "LightFieldStudio requires the backend/Vision source submodule")
endif()

add_subdirectory(src)
