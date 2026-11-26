find_package(CUDAToolkit REQUIRED)
add_executable(LightFieldStudio WIN32 main.cpp)
if(MSVC)
    target_link_options(LightFieldStudio PRIVATE /ENTRY:mainCRTStartup)
endif()
set_target_properties(LightFieldStudio PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
target_link_libraries(LightFieldStudio PRIVATE lightfieldstudio_core vision-window-ext CUDA::cuda_driver opengl32 comdlg32)
target_include_directories(LightFieldStudio PRIVATE
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${PROJECT_SOURCE_DIR}/backend/Vision/src")
