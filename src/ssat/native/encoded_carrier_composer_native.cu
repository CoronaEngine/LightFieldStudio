#include "ssat/native/encoded_carrier_composer_native_params.h"
#include "ssat/native/canonical_transfer.cuh"

#include "ssat/native/packed_surface_observation_contract.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace denoise_native = lfs::denoise::native;
namespace ssat = lfs::ssat;
namespace ssat_native = lfs::ssat::native;

namespace {

constexpr std::uint32_t kSurfaceDiffuse = 1u << 0u;
constexpr std::uint32_t kSurfaceGlossy = 1u << 1u;
constexpr std::uint32_t kSurfaceNearSpecular = 1u << 2u;
constexpr std::uint32_t kSurfaceTransmission = 1u << 3u;
constexpr std::uint32_t kSurfaceEmissive = 1u << 4u;

template<typename T>
__device__ __forceinline__ T *pointer(std::uint64_t address) {
    return reinterpret_cast<T *>(static_cast<std::uintptr_t>(address));
}

__device__ __forceinline__ bool finite_value(float value) {
    return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ float half_to_float(std::uint16_t value) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
    return result;
}

__device__ __forceinline__ float demodulated_carrier_value(
    const denoise_native::EncodedCarrierComposeParams &params,
    std::uint32_t pixel,
    std::uint32_t carrier) {
    const std::uint32_t index = pixel * 3u + carrier;
    const auto observation =
        pointer<const denoise_native::PackedSurfaceObservationV1>(
            params.current_observation)[index];
    const auto contract =
        ssat_native::validate_packed_surface_observation_v1(observation);
    if (contract.status != 0u) return 0.0f;
    if (params.deterministic_environment != 0u &&
        contract.domain == ssat_native::kPackedSurfaceDomainEnvironment) {
        // A primary miss in fixed RGB without media is a direct environment
        // evaluation. Filtering adjacent encoded sites mixes different rays.
        // Use this target's traced value for both profiles; history is untouched.
        const auto radiance =
            pointer<const denoise_native::Float4>(params.current_radiance)[index];
        const float value = carrier == 0u ? radiance.x :
            (carrier == 1u ? radiance.y : radiance.z);
        return finite_value(value) && value >= 0.0f ? value : 0.0f;
    }
    const denoise_native::Half4 signal =
        pointer<const denoise_native::Half4>(params.source)[index];
    const float demodulated = half_to_float(
        carrier == 0u ? signal.x : (carrier == 1u ? signal.y : signal.z));
    if (!finite_value(demodulated) || demodulated < 0.0f) return 0.0f;
    if (contract.domain != ssat_native::kPackedSurfaceDomainSurface) {
        return demodulated;
    }
    const auto albedo = observation.linear_albedo_rgb_diffuse_factor;
    const float component = half_to_float(
        carrier == 0u ? albedo.x : (carrier == 1u ? albedo.y : albedo.z));
    if (!finite_value(component)) return 0.0f;
    return demodulated * sqrtf(fmaf(component, component, 0.01f));
}

__device__ __forceinline__ float present_transfer_physical(float value) {
    return denoise_native::canonical_physical_transfer(value);
}

__device__ __forceinline__ void write_failed_canonical(
    const denoise_native::EncodedCarrierComposeParams &params,
    std::uint32_t pixel) {
    const float invalid = 1.0e30f;
    pointer<denoise_native::Float4>(params.canonical)[pixel] = {
        invalid, invalid, invalid, 0.0f};
}

} // namespace

extern "C" __global__ void lfs_encoded_carrier_compose_demodulated_half4(
    denoise_native::EncodedCarrierComposeParams params) {
    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = params.output_width * params.output_height;
    if (pixel >= count) return;
    if (params.frame_status_enabled != 0u &&
        *pointer<const std::uint32_t>(params.frame_status) != 0u) {
        write_failed_canonical(params, pixel);
        return;
    }
    const float red = demodulated_carrier_value(params, pixel, 0u);
    const float green = demodulated_carrier_value(params, pixel, 1u);
    const float blue = demodulated_carrier_value(params, pixel, 2u);
    pointer<denoise_native::Float4>(params.canonical)[pixel] = {
        present_transfer_physical(red), present_transfer_physical(green),
        present_transfer_physical(blue), 1.0f};
}

// Preserve the whole-frame fail-closed contract after fusing final+composition.
// Normal frames perform no color read/write in this explicit guard dispatch.
extern "C" __global__ void lfs_encoded_carrier_guard_canonical(denoise_native::EncodedCarrierComposeParams params){
 const unsigned pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=params.output_width*params.output_height)return;
 if(params.frame_status_enabled&&*pointer<const unsigned>(params.frame_status)!=0u)write_failed_canonical(params,pixel);
}
