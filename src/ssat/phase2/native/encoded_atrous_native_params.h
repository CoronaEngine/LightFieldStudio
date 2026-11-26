#pragma once

#include "ssat/native/denoise_native_types.h"

#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace lfs::denoise::encoded_atrous::native {

using Float4 = ::lfs::denoise::native::Float4;
using Half4 = ::lfs::denoise::native::Half4;

enum class HeldRawStorage : std::uint32_t { Empty, MovingExposure, HeldSeries };

struct EncodedAtrousNativeParams {
    std::uint64_t radiance{};
    std::uint64_t packed_observation{};
    std::uint64_t static_view_id{};
    std::uint64_t static_inverse_ray_length{};
    std::uint64_t static_view_x{};
    std::uint64_t static_focal_x{};
    std::uint64_t static_focal_y{};
    std::uint64_t scratch_a{};
    std::uint64_t scratch_b{};
    std::uint64_t frame_status{};
    Float4 world_to_local_row0{};
    Float4 world_to_local_row1{};
    Float4 world_to_local_row2{};
    float focal_distance{};
    std::uint32_t static_view_count{};
    std::uint32_t count{};
    std::uint32_t angular_compatibility_spatial{};
    std::uint32_t responsive_angular_radiometry{};
    std::uint32_t angular_compatibility_aperture_only{};
    std::uint32_t responsive_radiometry_glossy_only{};
    std::uint32_t angular_compatibility_responsive_only{};
    std::uint32_t statistical_reconstruction{};
    std::uint32_t temporal_history_available{};
    std::uint64_t view_chart_ids{},view_chart_offsets{};
    std::uint32_t view_chart_templates{},view_chart_count{},view_chart_views{},view_chart_wire{};
    Float4 view_chart_focal_bounds{};
    std::uint64_t view_chart_centers{},view_chart_blocks{},view_chart_sources{};
    std::uint64_t view_chart_slots{};
    std::uint32_t view_chart_block_count{},view_chart_source_count{},view_chart_cache_wire{};
    float final_energy{};
    std::uint64_t canonical{};
    std::uint32_t deterministic_environment{}, register_motion{};
    std::uint64_t atlas{};
    std::uint32_t atlas_side{}, atlas_cells{};
    float atlas_inverse_dx{}, atlas_inverse_span{};
    std::uint64_t stationary_weights{};
    // Stationary filter coefficients and the ordinary moving panel have disjoint submissions.
    union { std::uint64_t weights_refresh{}; std::uint64_t panel_samples; };
    std::uint64_t rigid_records{};
    std::uint32_t rigid_count{};
    // Stationary coefficient rebuild and ordinary moving/held raw storage are
    // mutually exclusive owners. Preserve the original CUDA launch layout.
    union {
        std::uint32_t rebuild_weights{};
        HeldRawStorage previous_held_raw;
    };
    std::uint64_t atlas_moments{}, cell_coefficients{};
    std::uint64_t hold_state{}, hold_previous{};
    std::uint32_t hold_same{};
    std::uint64_t motion_cell_sites{}, motion_cell_values{}, field_measurements{};
    std::uint32_t motion_cell_count{};
    std::uint32_t template_id_bits{}, aperture_cell_bits{}, resident_storage{};
    std::uint64_t regular_repeat_state{};
    // Read-only same-exposure regular M record. Its sigma_albedo.x stores sum of squared weights.
    std::uint64_t regular_measurement_geometry{};
    union {
        std::uint64_t resident_angular_residuals{};
        // Ordinary post-M passes borrow the current world-space BSDF normal,
        // one oct16x2 word per physical ray; all-ones denotes no unique frame.
        std::uint64_t regular_material_normals;
    };
};
static_assert(offsetof(EncodedAtrousNativeParams,regular_material_normals)==
              offsetof(EncodedAtrousNativeParams,resident_angular_residuals));

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool retained_current_compatible(
    const EncodedAtrousNativeParams &p) noexcept {
    return p.temporal_history_available <= 1u;
}

// The plain module is an execution specialization of the same ES math.
// Reject incompatible requests rather than silently dropping their features.
#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool plain_current_compatible(
    const EncodedAtrousNativeParams &p) noexcept {
    // Official Quality requests glossy-only aperture admission but does not
    // produce responsive guides. build_guide_from_surface leaves path_kind=0,
    // and make_angular_center_state rejects every center before angular math.
    // This exact inactive tuple has the same filter as the all-zero profile.

    const bool inactive_aperture = p.angular_compatibility_spatial == 1u &&
        p.angular_compatibility_aperture_only == 1u &&
        p.responsive_radiometry_glossy_only == 1u &&
        p.angular_compatibility_responsive_only <= 1u;
    const bool no_aperture = p.angular_compatibility_spatial == 0u &&
        p.angular_compatibility_aperture_only == 0u &&
        p.responsive_radiometry_glossy_only == 0u &&
        p.angular_compatibility_responsive_only == 0u;
    return p.temporal_history_available == 0u && (inactive_aperture || no_aperture) &&
        p.responsive_angular_radiometry == 0u;
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool glossy_measurement_compatible(const EncodedAtrousNativeParams &p) noexcept {
    return retained_current_compatible(p) && p.angular_compatibility_spatial==1u &&
        p.responsive_angular_radiometry==1u && p.angular_compatibility_aperture_only==1u &&
        p.responsive_radiometry_glossy_only==1u && p.angular_compatibility_responsive_only<=1u &&
        p.statistical_reconstruction==1u;
}

inline constexpr std::uint32_t kFrameStatusNonfinite = 1u << 0u;
inline constexpr std::uint32_t kFrameStatusInvalidLayout = 1u << 1u;

static_assert(std::is_standard_layout_v<EncodedAtrousNativeParams>);
static_assert(std::is_trivially_copyable_v<EncodedAtrousNativeParams>);
static_assert(sizeof(Half4) == 8u);

} // namespace lfs::denoise::encoded_atrous::native
