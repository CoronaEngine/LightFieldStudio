#pragma once

#include "ssat/native/denoise_native_types.h"

#include <cstdint>
#include <type_traits>

namespace lfs::denoise::physical_lf::native {

// Keep that prefix stable; larger runtime layouts use the appended LUT views.
inline constexpr std::uint32_t kInlinePhase3ViewCount = 64u;
inline constexpr std::uint32_t kMaximumPhase3ViewCount = 256u;
inline constexpr std::uint32_t kPhase3FrameStatusNonfinite = 1u << 0u;
inline constexpr std::uint32_t kPhase3FrameStatusInvalidReconstruction = 1u << 1u;

struct alignas(16) Float4 {
    float x{};
    float y{};
    float z{};
    float w{};
};

struct alignas(16) Uint4 {
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t z{};
    std::uint32_t w{};
};

struct Matrix4x4 {
    Float4 columns[4]{};
};

struct alignas(8) Uint2 {
    std::uint32_t x{};
    std::uint32_t y{};
};

using Half2 = ::lfs::denoise::native::Half2;
using Half4 = ::lfs::denoise::native::Half4;

using CurrentTemporalInput = ::lfs::denoise::native::CurrentTemporalInput24;
using TemporalSample = ::lfs::denoise::native::TemporalSample32;

using PackedDenoiserPathGuide =
    ::lfs::denoise::native::PackedDenoiserPathGuide12;

struct alignas(16) RigidInstanceMotionRecord {
    Float4 current_world_to_previous_world_row0{};
    Float4 current_world_to_previous_world_row1{};
    Float4 current_world_to_previous_world_row2{};
    Uint4 metadata{};
};

struct Phase3Params {
    std::uint64_t history_geometry_read{},history_geometry_write{};    std::uint64_t current_phase2{};
    std::uint64_t current_albedo{};
    std::uint64_t history_read{};
    std::uint64_t history_write{};
    std::uint64_t rigid_instance_motion{};
    std::uint64_t frame_status{};
    Matrix4x4 current_local_to_world{};
    Matrix4x4 previous_world_to_local{};
    std::uint32_t output_width{};
    std::uint32_t output_height{};
    std::uint32_t view_count{};
    std::uint32_t rigid_instance_count{};
    std::uint32_t rigid_instance_stride_bytes{};
    std::uint32_t history_valid{};
    std::uint32_t static_layout{};
    float pitch{};
    float inverse_pitch{};
    float slant_tangent{};
    float offset{};
    float focal_distance{};
    float focal_width{};
    float focal_height{};
    float array_angle_radians{};
    float focal_lattice_width{};
    float focal_lattice_height{};
    float sigma_x_lattice{};
    float sigma_u_views{};
    float position_tolerance_lattice{};
    float minimum_normal_similarity{};
    float maximum_albedo_relative_error{};
    float maximum_ray_angular_error{};
    float inverse_focal_distance{};
    float inverse_focal_lattice_width{};
    float inverse_focal_lattice_height{};
    float inverse_sigma_x_lattice{};
    float inverse_sigma_u_views{};
    float phase_view_scale{};
    float pitch_per_view{};
    float inverse_array_angle_radians{};
    float inverse_maximum_ray_angular_error{};
    float inverse_normal_error_range{};
    float phase_to_angle_scale{};
    float phase_angle_bias{};
    float base_position_tolerance{};
    float view_positions[kInlinePhase3ViewCount]{};
    float inverse_view_spacings[kInlinePhase3ViewCount]{};
    std::uint64_t current_packed_observation{};
    std::uint64_t current_motion{};
    std::uint64_t current_motion_validity{};

    // path leaves both fields zero and uses only current_phase2.

    std::uint64_t current_denoiser_path_guide{};
    std::uint32_t current_path_risk_packed{};
    std::uint32_t response_direction_glossy_only{};
    std::uint64_t extended_view_positions{};
    std::uint64_t extended_inverse_view_spacings{};
    std::uint32_t extended_view_count{};
    std::uint64_t native_first_history_head{};
};

static_assert(std::is_standard_layout_v<Float4>);
static_assert(std::is_standard_layout_v<Uint2>);
static_assert(std::is_standard_layout_v<Uint4>);
static_assert(std::is_standard_layout_v<CurrentTemporalInput>);
static_assert(std::is_standard_layout_v<TemporalSample>);
static_assert(std::is_standard_layout_v<PackedDenoiserPathGuide>);
static_assert(std::is_standard_layout_v<RigidInstanceMotionRecord>);
static_assert(std::is_trivially_copyable_v<Phase3Params>);
static_assert(std::is_standard_layout_v<Phase3Params>);
static_assert(sizeof(Float4) == 16u);
static_assert(sizeof(Phase3Params) <= 4096u,
              "CUDA kernel parameter ABI exceeds the launch limit");
static_assert(sizeof(Uint2) == 8u);
static_assert(sizeof(Uint4) == 16u);
static_assert(sizeof(Half2) == 4u);
static_assert(sizeof(Half4) == 8u);
static_assert(sizeof(CurrentTemporalInput) == 24u);
static_assert(sizeof(TemporalSample) == 32u);
static_assert(sizeof(PackedDenoiserPathGuide) == 12u);
static_assert(sizeof(RigidInstanceMotionRecord) == 64u);

} // namespace lfs::denoise::physical_lf::native
