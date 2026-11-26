add_library(lightfieldstudio_core STATIC
    ssat/phase2/encoded_filter_core.cpp
    ssat/resident_owner.cpp
    ssat/visible_surface.cpp
    ssat/denoise_pipeline.cpp
    ssat/denoise_presets.cpp
    ssat/encoded_carrier_composer.cpp
    ssat/native/encoded_carrier_composer_native_runtime.cpp
    ssat/phase2/encoded_atrous_spatial.cpp
    ssat/phase2/native/encoded_atrous_native_runtime.cpp
    ssat/phase3/physical_lf_reprojection_taa.cpp
    ssat/phase3/material_normal_history.cpp
    ssat/phase3/native/early_measurement_runtime.cpp
    lightfield_framebuffer.cpp
    json_document.cpp
    ssat/phase1/light_field_processor.cpp
    ssat/phase1/phase_frame_state.cpp
    ssat/phase1/phase_mapping.cpp
    render_scene_transform.cpp
    file_hash.cpp
)
target_include_directories(lightfieldstudio_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/backend/Vision/include")
target_compile_definitions(lightfieldstudio_core PUBLIC LFS_SOURCE_BACKEND=1)
target_link_libraries(lightfieldstudio_core PUBLIC lightfieldstudio_config ocarina-include ocarina ocarina-rhi bcrypt ole32 windowscodecs PRIVATE vision-source-runtime)
# Studio-owned framebuffer plugin uses the frozen renderer's plugin ABI.
vision_add_plugin(ssat_resident CATEGORY framebuffer SOURCES ssat/resident_framebuffer.cpp)
target_include_directories(vision-framebuffer-ssat_resident PRIVATE "${PROJECT_SOURCE_DIR}/backend/Vision/src")
set_target_properties(vision-framebuffer-ssat_resident PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
target_include_directories(lightfieldstudio_core PRIVATE "${PROJECT_SOURCE_DIR}/backend/Vision/src")
target_link_libraries(lightfieldstudio_core PRIVATE vision-base)
add_dependencies(lightfieldstudio_core vision-framebuffer-ssat_resident)

file(GLOB_RECURSE LFS_NATIVE_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")
function(lfs_native stem source)
    set(output "${CMAKE_CURRENT_BINARY_DIR}/generated/${stem}.fatbin")
    add_custom_command(OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --fatbin -allow-unsupported-compiler
            --std=c++17 -O3 --fmad=true
            --restrict --extra-device-vectorization
            --generate-code=arch=compute_120,code=sm_120
            --generate-code=arch=compute_75,code=compute_75
            ${ARGN} -I "${CMAKE_CURRENT_SOURCE_DIR}" -o "${output}" "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
        DEPENDS "${source}" ${LFS_NATIVE_HEADERS} VERBATIM)
    add_custom_target(${stem} ALL
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${output}" "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/${stem}.fatbin"
        DEPENDS "${output}")
    add_dependencies(lightfieldstudio_core ${stem})
endfunction()
lfs_native(lfs_encoded_atrous_native ssat/phase2/native/encoded_atrous_native.cu --use_fast_math --ftz=false --prec-div=false --prec-sqrt=false)
lfs_native(lfs_encoded_atrous_plain ssat/phase2/native/encoded_atrous_native.cu --use_fast_math --ftz=false --prec-div=false --prec-sqrt=false -DLFS_ENCODED_PLAIN_CURRENT=1)
lfs_native(lfs_encoded_atrous_glossy ssat/phase2/native/encoded_atrous_native.cu --use_fast_math --ftz=false --prec-div=false --prec-sqrt=false -DLFS_ENCODED_GLOSSY_MEASUREMENT=1)
lfs_native(lfs_encoded_carrier_composer ssat/native/encoded_carrier_composer_native.cu --ftz=false --prec-div=true --prec-sqrt=true)
lfs_native(lfs_early_measurement ssat/phase3/native/early_measurement.cu --ftz=false --prec-div=false --prec-sqrt=true -DLFS_TEMPORAL_PRODUCTION=1)

add_subdirectory(studio)
