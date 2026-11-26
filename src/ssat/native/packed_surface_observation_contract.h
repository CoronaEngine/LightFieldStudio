#pragma once

#include "ssat/native/denoise_native_types.h"

#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define LFS_SSAT_PACKED_HOST_DEVICE __host__ __device__
#else
#define LFS_SSAT_PACKED_HOST_DEVICE
#endif

namespace lfs::ssat::native {

using PackedSurfaceObservationV1 = ::lfs::denoise::native::PackedSurfaceObservationV1;
using ::lfs::denoise::native::kPackedSurfaceDomainMask;
using ::lfs::denoise::native::kPackedSurfaceFlagsShift;
using ::lfs::denoise::native::kPackedSurfaceFlagsMask;
using ::lfs::denoise::native::kPackedSurfaceRoughnessShift;
using ::lfs::denoise::native::kPackedSurfaceReactivityShift;
using ::lfs::denoise::native::kPackedSurfaceReservedMask;
using ::lfs::denoise::native::kPackedSurfaceDomainSurface;
using ::lfs::denoise::native::kPackedSurfaceDomainEnvironment;

inline constexpr std::uint32_t kPackedObservationContractInvalidLayout =
    1u << 0u;
inline constexpr std::uint32_t kPackedObservationContractNonfinite =
    1u << 1u;

struct PackedObservationContractResultV1 {
    std::uint32_t status{};
    std::uint32_t domain{};
    std::uint32_t surface_flags{};
    std::uint32_t roughness_unorm8{};
    std::uint32_t reactivity_unorm8{};
};

LFS_SSAT_PACKED_HOST_DEVICE inline bool packed_half_is_finite(
    std::uint16_t value) noexcept {
    return (value & 0x7c00u) != 0x7c00u;
}

LFS_SSAT_PACKED_HOST_DEVICE inline bool packed_float_is_finite(
    float value) noexcept {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

LFS_SSAT_PACKED_HOST_DEVICE inline bool packed_positive_infinity(
    float value) noexcept {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(value) == 0x7f800000u;
#else
    return value == std::numeric_limits<float>::infinity();
#endif
}

LFS_SSAT_PACKED_HOST_DEVICE inline PackedObservationContractResultV1
validate_packed_surface_observation_v1(
    const PackedSurfaceObservationV1 &observation) noexcept {
    PackedObservationContractResultV1 result{};
    result.domain = observation.metadata & kPackedSurfaceDomainMask;
    result.surface_flags =
        (observation.metadata >> kPackedSurfaceFlagsShift) &
        kPackedSurfaceFlagsMask;
    result.roughness_unorm8 =
        (observation.metadata >> kPackedSurfaceRoughnessShift) & 0xffu;
    result.reactivity_unorm8 =
        (observation.metadata >> kPackedSurfaceReactivityShift) & 0xffu;
    if ((observation.metadata & kPackedSurfaceReservedMask) != 0u ||
        (result.domain != kPackedSurfaceDomainSurface &&
         result.domain != kPackedSurfaceDomainEnvironment)) {
        result.status |= kPackedObservationContractInvalidLayout;
    }
    if (!packed_half_is_finite(
            observation.linear_albedo_rgb_diffuse_factor.x) ||
        !packed_half_is_finite(
            observation.linear_albedo_rgb_diffuse_factor.y) ||
        !packed_half_is_finite(
            observation.linear_albedo_rgb_diffuse_factor.z) ||
        !packed_half_is_finite(
            observation.linear_albedo_rgb_diffuse_factor.w)) {
        result.status |= kPackedObservationContractNonfinite;
    }
    if (result.domain == kPackedSurfaceDomainSurface) {
        if (observation.surface_key == ~std::uint64_t{0u} ||
            observation.hit_distance < 0.0f) {
            result.status |= kPackedObservationContractInvalidLayout;
        }
        if (!packed_float_is_finite(observation.hit_distance)) {
            result.status |= kPackedObservationContractNonfinite;
        }
    } else if (result.domain == kPackedSurfaceDomainEnvironment) {
        if (observation.surface_key != ~std::uint64_t{0u} ||
            !packed_positive_infinity(observation.hit_distance)) {
            result.status |= kPackedObservationContractInvalidLayout;
        }
        if (!packed_float_is_finite(observation.hit_distance) &&
            !packed_positive_infinity(observation.hit_distance)) {
            result.status |= kPackedObservationContractNonfinite;
        }
    }
    return result;
}

static_assert(sizeof(PackedObservationContractResultV1) == 20u);
static_assert(alignof(PackedObservationContractResultV1) == 4u);

} // namespace lfs::ssat::native

#undef LFS_SSAT_PACKED_HOST_DEVICE
