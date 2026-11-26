#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace lfs::denoise::native {

inline constexpr std::uint32_t kMaximumLfViewCount = 64u;

struct alignas(16) Float4 {
    float x{};
    float y{};
    float z{};
    float w{};
};

struct alignas(8) Float2 {
    float x{};
    float y{};
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

struct alignas(2) Half4 {
    std::uint16_t x{};
    std::uint16_t y{};
    std::uint16_t z{};
    std::uint16_t w{};
};

struct alignas(4) Half2 {
    std::uint16_t x{};
    std::uint16_t y{};
};

// Geometry and metadata accompanying the RGB measurement history.
struct alignas(8) CurrentTemporalInput24 {
    std::uint64_t surface_key{};
    float depth{};
    std::uint32_t packed_normal{};
    std::uint32_t metadata{};
    Half2 signal_albedo{};
};

struct alignas(8) TemporalSample32 {
    std::uint64_t surface_key{};
    float depth{};
    std::uint32_t packed_normal{};
    std::uint32_t metadata{};
    Half2 sigma_albedo{};
    Half4 signal_physical_mean_rms{};
};

struct alignas(16) RigidInstanceMotionRecord {
    Float4 current_world_to_previous_world_row0{};
    Float4 current_world_to_previous_world_row1{};
    Float4 current_world_to_previous_world_row2{};
    Uint4 metadata{};
};

inline constexpr std::uint32_t kPackedSurfaceDomainMask = 0x3u;
inline constexpr std::uint32_t kPackedSurfaceFlagsShift = 2u;
inline constexpr std::uint32_t kPackedSurfaceFlagsMask = 0x7fu;
inline constexpr std::uint32_t kPackedSurfaceRoughnessShift = 9u;
inline constexpr std::uint32_t kPackedSurfaceReactivityShift = 17u;
inline constexpr std::uint32_t kPackedSurfacePathRiskShift = 25u;
inline constexpr std::uint32_t kPackedSurfacePathRiskMask = 0x3fu;
inline constexpr std::uint32_t kPackedSurfaceResponseEtaShift = 25u;
inline constexpr std::uint32_t kPackedSurfaceResponseEtaMask = 0x3fu;
inline constexpr std::uint32_t kPackedSurfaceReservedMask = 0x80000000u;
inline constexpr std::uint32_t kPackedSurfaceDomainSurface = 1u;
inline constexpr std::uint32_t kPackedSurfaceDomainEnvironment = 2u;

struct alignas(8) PackedSurfaceObservationV1 {
    std::uint64_t surface_key{};
    float hit_distance{};
    std::uint32_t packed_geometric_normal_oct16{};
    std::uint32_t primitive_id{};
    std::uint32_t metadata{};
    Half4 linear_albedo_rgb_diffuse_factor{};
};

struct alignas(4) PackedDenoiserPathGuide12 {
    std::uint32_t payload0{};
    std::uint32_t payload1{};
    std::uint32_t metadata{};
};

static_assert(sizeof(Float4) == 16u);
static_assert(sizeof(Float2) == 8u);
static_assert(sizeof(Uint4) == 16u);
static_assert(sizeof(Half2) == 4u);
static_assert(sizeof(Half4) == 8u);
static_assert(sizeof(CurrentTemporalInput24) == 24u);
static_assert(sizeof(TemporalSample32) == 32u);
static_assert(sizeof(PackedSurfaceObservationV1) == 32u);
static_assert(alignof(PackedSurfaceObservationV1) == 8u);
static_assert(sizeof(PackedDenoiserPathGuide12) == 12u);
static_assert(alignof(PackedDenoiserPathGuide12) == 4u);
static_assert(sizeof(RigidInstanceMotionRecord) == 64u);
static_assert(std::is_standard_layout_v<PackedSurfaceObservationV1>);
static_assert(std::is_standard_layout_v<PackedDenoiserPathGuide12>);
static_assert(std::is_standard_layout_v<CurrentTemporalInput24>);
static_assert(std::is_standard_layout_v<TemporalSample32>);

} // namespace lfs::denoise::native
