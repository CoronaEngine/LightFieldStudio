#pragma once
#include "ssat/phase3/native/physical_lf_taa_native_params.h"
#include "ssat/native/reconstruction_coverage.h"
#include "material_vertex_frame.h"
namespace lfs::denoise::mr_native {
inline constexpr std::uint32_t kHistoryFp16RgbRms=0x4d523136u;
struct Params {
    physical_lf::native::Phase3Params geometry{};
    std::uint64_t measurement_read{};
    std::uint64_t measurement_write{};
    std::uint64_t stationary_radiance{};
    // Live between M reconstruction and commit, beyond the displayed pixels.
    std::uint64_t history_metadata_staging{};
    union { std::uint64_t held_angular_residuals{}; std::uint64_t previous_panel_samples; };
    union { std::uint64_t current_angular_residuals{}; std::uint64_t current_panel_samples; };
    std::uint64_t native_second_hit{}, committed_measurement{};
    std::uint32_t history_storage{kHistoryFp16RgbRms};
    std::int32_t rgb_b0x{}, rgb_b0y{}, rgb_b1x{}, rgb_b1y{};
    std::uint32_t rgb_basis_valid{}, deterministic_motion_environment{};
    RegularR2Coverage regular_r2_coverage{};
    std::uint64_t current_material_normals{},previous_material_normals{};
    MaterialVertexFrame vertex_frame{};
};
}
