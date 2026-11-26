#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace lfs::denoise::physical_lf::native_current {

#if defined(__CUDACC__)

__device__ __forceinline__ float encoded_atrous_modulation(
    std::uint32_t validity,
    float albedo) {
    if (validity == 0u) return 1.0f;
    const float radicand = fmaf(albedo, albedo, 0.01f);
    float result;
    asm("sqrt.rn.f32 %0, %1;" : "=f"(result) : "f"(radicand));
    return result;
}

__device__ __forceinline__ bool encoded_atrous_domain(
    std::uint32_t validity,
    bool diffuse,
    bool responsive,
    float diffuse_factor,
    float albedo) {
    return validity != 0u && diffuse && !responsive &&
        diffuse_factor >= 0.5f && albedo >= 0.05f;
}

__device__ __forceinline__ float encoded_atrous_variance_scale_squared(
    std::uint32_t validity,
    bool domain_v12,
    float albedo) {
    if (validity == 0u) return 1.0f;
    const float modulation_squared = fmaf(albedo, albedo, 0.01f);
    if (!domain_v12) return modulation_squared;
    const float denominator = fmaxf(albedo, 0.05f);
    return modulation_squared / (denominator * denominator);
}

#endif

} // namespace lfs::denoise::physical_lf::native_current
