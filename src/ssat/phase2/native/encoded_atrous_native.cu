#include "ssat/phase2/native/encoded_atrous_native_params.h"
#include "ssat/native/packed_surface_observation_contract.h"
#include "ssat/native/reconstruction_coverage.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#if defined(LFS_ENCODED_GLOSSY_MEASUREMENT) || defined(LFS_ENCODED_PLAIN_CURRENT)
#include "ssat/native/canonical_transfer.cuh"
#endif
#include <cfloat>
#include <cstdint>

#ifndef LFS_ENCODED_ATROUS_TILE_ROWS
#define LFS_ENCODED_ATROUS_TILE_ROWS 16
#endif

namespace lfs::denoise::encoded_atrous::native {
namespace {

namespace denoise_native = ::lfs::denoise::native;
__device__ __forceinline__ unsigned chart_template_id(const EncodedAtrousNativeParams& p,unsigned i) {
    const auto* words=reinterpret_cast<const unsigned*>(p.view_chart_ids);
    if(p.template_id_bits==16u)return (words[i>>1]>>((i&1u)*16u))&65535u;
    return words[i];
}

namespace ssat_native = ::lfs::ssat::native;

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Vec4 {
    float x;
    float y;
    float z;
    float w;
};

struct Guide {
    Vec3 point;
    // albedo.w is an exact raw status code: 0=environment, 1=surface,
    // 0xffff=invalid.  The fourth albedo half was otherwise unused.
    Half4 albedo;
#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
    // Decode the original quantized normal once per shared tile site, rather
    // than once per filter tap. Values retain the original oct16 boundary.
    Vec3 normal;
#else
    std::uint32_t packed_normal;
#endif
    std::uint32_t instance;
#ifndef LFS_ENCODED_PLAIN_CURRENT
    std::uint16_t view_id;
    std::uint16_t roughness_unorm16;
#ifndef LFS_ENCODED_GLOSSY_MEASUREMENT
    std::uint32_t bank_padding{};
#endif
#endif
};
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
static_assert(sizeof(Guide) == 40u);
struct SharedGuide {
    Vec3 point;
    Half4 albedo;
    std::uint32_t instance;
    std::uint16_t view_id,roughness_unorm16;
};
static_assert(sizeof(SharedGuide)==28u);
#define LFS_SHARED_NORMALS(Name,Count) __shared__ Vec3 Name[Count];
#else
static_assert(sizeof(Guide) == 36u);
using SharedGuide=Guide;
#define LFS_SHARED_NORMALS(Name,Count) Vec3 *Name=nullptr;
#endif

__device__ __forceinline__ Guide load_tile_guide(const SharedGuide *shared,const Vec3 *normals,int index) {
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    const auto v=shared[index];Guide g{};
    g.point=v.point;g.albedo=v.albedo;g.instance=v.instance;
    g.view_id=v.view_id;g.roughness_unorm16=v.roughness_unorm16;g.normal=normals[index];return g;
#else
    return shared[index];
#endif
}
__device__ __forceinline__ void store_tile_guide(SharedGuide *shared,Vec3 *normals,int index,const Guide &g) {
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    shared[index]={g.point,g.albedo,g.instance,g.view_id,g.roughness_unorm16};normals[index]=g.normal;
#else
    shared[index]=g;
#endif
}

__device__ __forceinline__ std::uint16_t guide_view(const Guide &guide) {
#ifdef LFS_ENCODED_PLAIN_CURRENT
    return 0u; // Angular consumers are absent from the plain specialization.
#else
    return guide.view_id;
#endif
}

__device__ __forceinline__ std::uint16_t guide_roughness(const Guide &guide) {
#ifdef LFS_ENCODED_PLAIN_CURRENT
    return 0u;
#else
    return guide.roughness_unorm16;
#endif
}

constexpr std::uint32_t kInvalidGuide = 0xffffffffu;
constexpr std::uint16_t kGuideStatusMask = 0x3u;
constexpr std::uint16_t kGuidePathKindShift = 2u;
constexpr std::uint16_t kGuidePathKindMask = 0x3u;
constexpr std::uint16_t kGuidePathPayloadShift = 4u;
constexpr std::uint16_t kGuidePathPayloadMask = 0x0fffu;
constexpr std::uint16_t kGuidePathResponsive = 1u;
constexpr std::uint32_t kGuideResponseGlossyOnlyBit = 8u;
constexpr std::uint32_t kSurfaceDiffuse = 1u << 0u;
constexpr std::uint32_t kSurfaceGlossy = 1u << 1u;
constexpr std::uint32_t kSurfaceNearSpecular = 1u << 2u;
constexpr std::uint32_t kSurfaceTransmission = 1u << 3u;
__device__ __forceinline__ bool register_motion_surface(unsigned md) {
    const unsigned flags=(md>>2u)&127u;
    return (md&3u)==1u && (flags==kSurfaceDiffuse || flags==kSurfaceGlossy || flags==kSurfaceNearSpecular);
}


__device__ __forceinline__ std::uint32_t guide_validity(
    const Guide &guide) {
    return guide.albedo.w == 0xffffu
        ? kInvalidGuide
        : static_cast<std::uint32_t>(guide.albedo.w & kGuideStatusMask);
}

__device__ __forceinline__ void set_guide_validity(
    Guide &guide,
    std::uint32_t validity) {
    guide.albedo.w = validity == kInvalidGuide
        ? static_cast<std::uint16_t>(0xffffu)
        : static_cast<std::uint16_t>(
              (guide.albedo.w & ~kGuideStatusMask) |
              (validity & kGuideStatusMask));
}

__device__ __forceinline__ std::uint32_t guide_path_kind(
    const Guide &guide) {
#ifdef LFS_ENCODED_PLAIN_CURRENT
    return 0u;
#else
    return guide.albedo.w == 0xffffu ? 0u :
        (guide.albedo.w >> kGuidePathKindShift) & kGuidePathKindMask;
#endif
}

__device__ __forceinline__ std::uint32_t guide_path_payload(
    const Guide &guide) {
#ifdef LFS_ENCODED_PLAIN_CURRENT
    return 0u;
#else
    return guide.albedo.w == 0xffffu ? 0u :
        (guide.albedo.w >> kGuidePathPayloadShift) & kGuidePathPayloadMask;
#endif
}

__device__ __forceinline__ void set_guide_path(
    Guide &guide,
    std::uint32_t kind,
    std::uint32_t payload) {
    if (guide.albedo.w == 0xffffu) return;
    const std::uint16_t status = guide.albedo.w & kGuideStatusMask;
    guide.albedo.w = static_cast<std::uint16_t>(
        status |
        ((kind & kGuidePathKindMask) << kGuidePathKindShift) |
        ((payload & kGuidePathPayloadMask) << kGuidePathPayloadShift));
}

__device__ __forceinline__ bool responsive_compatible(
    const Guide &center,
    const Guide &sample) {
    if (guide_path_kind(center) != kGuidePathResponsive) return true;
    return guide_path_kind(sample) == kGuidePathResponsive &&
        guide_path_payload(center) == guide_path_payload(sample);
}

struct InitGuideValue {
    Half4 signal;
    Guide guide;
    bool grid_site;
};

__device__ __forceinline__ Vec3 make_vec3(float value) {
    return {value, value, value};
}

__device__ __forceinline__ Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ Vec3 operator-(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __forceinline__ Vec3 operator*(Vec3 a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

__device__ __forceinline__ Vec3 operator/(Vec3 a, float b) {
    return {a.x / b, a.y / b, a.z / b};
}

__device__ __forceinline__ Vec3 &operator+=(Vec3 &a, Vec3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ __forceinline__ float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float length(Vec3 value) {
    return sqrtf(dot(value, value));
}

__device__ __forceinline__ Vec3 normalize(Vec3 value) {
    const float squared = dot(value, value);
    return squared > 0.0f && isfinite(squared)
        ? value * rsqrtf(squared) : Vec3{0.0f, 0.0f, 1.0f};
}

__device__ __forceinline__ Vec3 clamp3(Vec3 value, float low, float high) {
    return {
        fminf(fmaxf(value.x, low), high),
        fminf(fmaxf(value.y, low), high),
        fminf(fmaxf(value.z, low), high)};
}

__device__ __forceinline__ bool finite3(Vec3 value) {
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

__device__ __forceinline__ float luminance(Vec3 value) {
    return value.x * 0.2126f + value.y * 0.7152f + value.z * 0.0722f;
}

__device__ __forceinline__ float safe_luminance(Vec3 value) {
    return fminf(fmaxf(luminance(value), 0.0f), 200.0f);
}

__device__ __forceinline__ float pow32(float value) {
    const float squared = value * value;
    const float fourth = squared * squared;
    const float eighth = fourth * fourth;
    const float sixteenth = eighth * eighth;
    return sixteenth * sixteenth;
}

__device__ __forceinline__ std::uint16_t to_half(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

__device__ __forceinline__ float from_half(std::uint16_t value) {
    return __half2float(__ushort_as_half(value));
}

__device__ __forceinline__ Half4 pack(Vec4 value) {
    return {to_half(value.x), to_half(value.y),
            to_half(value.z), to_half(value.w)};
}

__device__ __forceinline__ Vec4 unpack(Half4 value) {
    return {from_half(value.x), from_half(value.y),
            from_half(value.z), from_half(value.w)};
}

// The global half4 arrays have 8-byte aligned allocations and strides. Keep
// the ABI's 2-byte Half4 alignment, but express a single 64-bit memory access
// instead of four strided 16-bit loads/stores. Every stored bit is unchanged.
__device__ __forceinline__ Half4 load_signal(const Half4 *source, int index) {
    const uint2 bits = reinterpret_cast<const uint2 *>(source)[index];
    return {static_cast<std::uint16_t>(bits.x), static_cast<std::uint16_t>(bits.x >> 16u),
            static_cast<std::uint16_t>(bits.y), static_cast<std::uint16_t>(bits.y >> 16u)};
}

__device__ __forceinline__ void store_signal(Half4 *destination, int index, Half4 value) {
    reinterpret_cast<uint2 *>(destination)[index] = make_uint2(
        static_cast<unsigned>(value.x) | (static_cast<unsigned>(value.y) << 16u),
        static_cast<unsigned>(value.z) | (static_cast<unsigned>(value.w) << 16u));
}

__device__ __forceinline__ void fail(
    const EncodedAtrousNativeParams &params,
    std::uint32_t bit = kFrameStatusNonfinite) {
    atomicOr(reinterpret_cast<unsigned int *>(params.frame_status), bit);
}

__device__ __forceinline__ Vec3 safe_albedo(Vec3 value) {
    return {
        sqrtf(value.x * value.x + 0.01f),
        sqrtf(value.y * value.y + 0.01f),
        sqrtf(value.z * value.z + 0.01f)};
}

__device__ __forceinline__ Vec3 local_normal(
    const EncodedAtrousNativeParams &params,
    Vec3 world) {
    // The following octahedral pack divides by the vector's L1 norm, so a
    // positive normalization scale cancels exactly. The unpack used by every
    // filter tap normalizes again. Preserve direction while avoiding a
    // redundant rsqrt for every reconstructed guide.
    return {
        world.x * params.world_to_local_row0.x +
            world.y * params.world_to_local_row0.y +
            world.z * params.world_to_local_row0.z,
        world.x * params.world_to_local_row1.x +
            world.y * params.world_to_local_row1.y +
            world.z * params.world_to_local_row1.z,
        world.x * params.world_to_local_row2.x +
            world.y * params.world_to_local_row2.y +
            world.z * params.world_to_local_row2.z};
}

__device__ __forceinline__ Vec3 unpack_normal(std::uint32_t packed) {
    const float x = static_cast<float>(packed & 0xffffu) /
        65535.0f * 2.0f - 1.0f;
    const float y = static_cast<float>((packed >> 16u) & 0xffffu) /
        65535.0f * 2.0f - 1.0f;
    const float z = 1.0f - fabsf(x) - fabsf(y);
    const float fold = fminf(fmaxf(-z, 0.0f), 1.0f);
    return normalize({
        x + (x >= 0.0f ? -fold : fold),
        y + (y >= 0.0f ? -fold : fold),
        z});
}

__device__ __forceinline__ Vec3 guide_normal(const Guide &guide) {
#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
    return guide.normal;
#else
    return unpack_normal(guide.packed_normal);
#endif
}

__device__ __forceinline__ std::uint32_t pack_normal(Vec3 normal) {
    const float denominator = fmaxf(
        fabsf(normal.x) + fabsf(normal.y) + fabsf(normal.z), 1.0e-20f);
    const float inverse_denominator = 1.0f / denominator;
    const float projected_x = normal.x * inverse_denominator;
    const float projected_y = normal.y * inverse_denominator;
    const float sign_x = projected_x >= 0.0f ? 1.0f : -1.0f;
    const float sign_y = projected_y >= 0.0f ? 1.0f : -1.0f;
    const float encoded_x = normal.z >= 0.0f
        ? projected_x
        : (1.0f - fabsf(projected_y)) * sign_x;
    const float encoded_y = normal.z >= 0.0f
        ? projected_y
        : (1.0f - fabsf(projected_x)) * sign_y;
    const std::uint32_t x = static_cast<std::uint32_t>(fminf(fmaxf(
        roundf((encoded_x * 0.5f + 0.5f) * 65535.0f), 0.0f), 65535.0f));
    const std::uint32_t y = static_cast<std::uint32_t>(fminf(fmaxf(
        roundf((encoded_y * 0.5f + 0.5f) * 65535.0f), 0.0f), 65535.0f));
    return x | (y << 16u);
}

__device__ __forceinline__ Vec3 guide_albedo(const Guide &guide) {
    return {from_half(guide.albedo.x), from_half(guide.albedo.y),
            from_half(guide.albedo.z)};
}

__device__ __forceinline__ bool guide_response_glossy_only(
    const Guide &guide) {
    return guide_path_kind(guide) == kGuidePathResponsive &&
        ((guide_path_payload(guide) >> kGuideResponseGlossyOnlyBit) & 1u) != 0u;
}

struct AngularCenterState {
    Vec3 direction{};
    float alpha{1.0f};
    float aperture{};
    bool aperture_only{};
    bool enabled{};
};

template<int Width>
__device__ __forceinline__ Vec3 static_primary_direction(
    const EncodedAtrousNativeParams &params,
    const Guide &guide,
    int x,
    int y) {
    const auto *inverse_length =
        reinterpret_cast<const float *>(params.static_inverse_ray_length);
    const auto *view_x = reinterpret_cast<const float *>(params.static_view_x);
    const auto *focal_x = reinterpret_cast<const float *>(params.static_focal_x);
    const auto *focal_y = reinterpret_cast<const float *>(params.static_focal_y);
    const float inverse = inverse_length[y * Width + x];
    const float origin = view_x[guide_view(guide)];
    return {
        (focal_x[x] - origin) * inverse,
        focal_y[y] * inverse,
        params.focal_distance * inverse};
}

template<int Width>
__device__ __forceinline__ AngularCenterState make_angular_center_state(
    const EncodedAtrousNativeParams &params,
    const Guide &center,
    int center_x,
    int center_y) {
    AngularCenterState state{};
    if (params.angular_compatibility_spatial == 0u ||
        params.static_view_count <= 1u || guide_validity(center) != 1u ||
        guide_view(center) >= params.static_view_count ||
        (params.responsive_radiometry_glossy_only != 0u &&
         !guide_response_glossy_only(center)) ||
        (params.angular_compatibility_responsive_only != 0u &&
         guide_path_kind(center) != kGuidePathResponsive)) {
        return state;
    }
    const auto *view_x = reinterpret_cast<const float *>(params.static_view_x);
    const float center_origin = view_x[guide_view(center)];
    state.aperture = center_origin;
    state.aperture_only =
        params.angular_compatibility_aperture_only != 0u &&
        (params.responsive_radiometry_glossy_only == 0u ||
         guide_response_glossy_only(center));
    if (!state.aperture_only) {
        state.direction = static_primary_direction<Width>(
            params, center, center_x, center_y);

    }
    float spacing = FLT_MAX;
    if (guide_view(center) > 0u) {
        spacing = fminf(
            spacing, fabsf(center_origin - view_x[guide_view(center) - 1u]));
    }
    if (static_cast<std::uint32_t>(guide_view(center)) + 1u <
        params.static_view_count) {
        spacing = fminf(
            spacing, fabsf(
                view_x[guide_view(center) + 1u] - center_origin));
    }
    if (!isfinite(spacing)) spacing = 0.0f;
    const float angular_footprint = spacing /
        fmaxf(params.focal_distance, 1.0e-20f);
    const float roughness = static_cast<float>(guide_roughness(center)) *
        (1.0f / 65535.0f);
    const float lobe_width = roughness * roughness;
    state.alpha = fmaxf(
        fmaxf(lobe_width, angular_footprint), 1.0e-6f);
    state.enabled = isfinite(state.alpha) && isfinite(state.aperture) &&
        (state.aperture_only || finite3(state.direction));
    if (!state.enabled) fail(params);
    return state;
}

template<int Width>
__device__ __forceinline__ float angular_compatibility_weight(
    const EncodedAtrousNativeParams &params,
    const Guide &center,
    const Guide &sample,
    int sample_x,
    int sample_y,
    const AngularCenterState &center_state) {
    if (!center_state.enabled || guide_validity(sample) != 1u ||
        guide_view(sample) >= params.static_view_count) {
        return 1.0f;
    }
    if (params.responsive_radiometry_glossy_only != 0u &&
        !guide_response_glossy_only(sample)) {
        return 0.0f;
    }
    float theta_squared = 0.0f;
    if (center_state.aperture_only) {
        const auto *view_x =
            reinterpret_cast<const float *>(params.static_view_x);
        const float delta = (view_x[guide_view(sample)] - center_state.aperture) /
            fmaxf(params.focal_distance, 1.0e-20f);
        theta_squared = delta * delta;
    } else {
        Vec3 sample_direction = static_primary_direction<Width>(
            params, sample, sample_x, sample_y);

        theta_squared = fmaxf(
            2.0f * (1.0f - dot(center_state.direction, sample_direction)),
            0.0f);
    }
    const float result = expf(
        -0.5f * theta_squared / (center_state.alpha * center_state.alpha));
    if (!isfinite(result)) {
        fail(params);
        return 0.0f;
    }
    return result;
}

#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
// This module validates the aperture-only, pure-glossy contract before dispatch.
// Keep its exponent until the geometry/range exponential instead of evaluating
// two exponentials and multiplying their rounded values at every accepted tap.
__device__ __forceinline__ float measurement_aperture_exponent(
    const EncodedAtrousNativeParams &p,const Guide &sample,const AngularCenterState &center) {
    if(!center.enabled || guide_validity(sample)!=1u || guide_view(sample)>=p.static_view_count)return 0.f;
    if(!guide_response_glossy_only(sample))return -CUDART_INF_F;
    const float delta=(reinterpret_cast<const float*>(p.static_view_x)[guide_view(sample)]-center.aperture)/fmaxf(p.focal_distance,1.e-20f);
    return -.5f*delta*delta/(center.alpha*center.alpha);
}
#endif

struct SurfaceInput {
    Vec3 albedo{};
    float hit{};
    Vec3 world_normal{};
    std::uint32_t validity{kInvalidGuide};
    std::uint32_t instance{};
    std::uint32_t surface_flags{};
    std::uint32_t roughness_unorm8{};
};

__device__ __forceinline__ SurfaceInput load_surface_input(
    const EncodedAtrousNativeParams &params,
    std::uint32_t index) {
    SurfaceInput result{};
    const denoise_native::PackedSurfaceObservationV1 observation =
        reinterpret_cast<const denoise_native::PackedSurfaceObservationV1 *>(
            params.packed_observation)[index];
    const auto contract =
        ssat_native::validate_packed_surface_observation_v1(observation);
    result.albedo = {
        from_half(observation.linear_albedo_rgb_diffuse_factor.x),
        from_half(observation.linear_albedo_rgb_diffuse_factor.y),
        from_half(observation.linear_albedo_rgb_diffuse_factor.z)};
    if (contract.status != 0u || !finite3(result.albedo)) {
        fail(params);
        return result;
    }
    if (contract.domain == ssat_native::kPackedSurfaceDomainEnvironment) {
        result.validity = 0u;
        return result;
    }
    if (contract.domain != ssat_native::kPackedSurfaceDomainSurface) {
        fail(params);
        return result;
    }
    result.hit = observation.hit_distance;
    result.world_normal = unpack_normal(
        observation.packed_geometric_normal_oct16);
    result.instance = static_cast<std::uint32_t>(
        observation.surface_key >> 32u);
    result.surface_flags =
        (observation.metadata >> denoise_native::kPackedSurfaceFlagsShift) &
        denoise_native::kPackedSurfaceFlagsMask;
    result.roughness_unorm8 = contract.roughness_unorm8;
    result.validity = 1u;
    if (!isfinite(result.hit) || result.hit < 0.0f ||
        !finite3(result.world_normal)) {
        fail(params);
        result.validity = kInvalidGuide;
    }
    return result;
}

template<int Width,bool Exposure=false>
__device__ __forceinline__ Guide build_guide_from_surface(
    const EncodedAtrousNativeParams &params,
    int x,
    int y,
    std::uint32_t index,
    const SurfaceInput &surface) {
    const auto *view_id =
        reinterpret_cast<const std::uint8_t *>(params.static_view_id);
    const auto *inverse_length =
        reinterpret_cast<const float *>(params.static_inverse_ray_length);
    const auto *view_x =
        reinterpret_cast<const float *>(params.static_view_x);
    const auto *focal_x =
        reinterpret_cast<const float *>(params.static_focal_x);
    const auto *focal_y =
        reinterpret_cast<const float *>(params.static_focal_y);

    Guide result{};
#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
    result.normal = unpack_normal(0u); // Preserve the environment center normal.
#endif
    result.instance = surface.instance;
#ifndef LFS_ENCODED_PLAIN_CURRENT
    result.roughness_unorm16 = static_cast<std::uint16_t>(
        surface.roughness_unorm8 * 257u);
#endif
    result.albedo = pack(
        {surface.albedo.x, surface.albedo.y, surface.albedo.z, 0.0f});
    set_guide_validity(result, surface.validity);
    if (guide_validity(result) == 0u) {
        if (!finite3(surface.albedo)) {
            fail(params);
            set_guide_validity(result, kInvalidGuide);
        }
        return result;
    }
    const bool glossy_only =
        (surface.surface_flags & kSurfaceGlossy) != 0u &&
        (surface.surface_flags &
         (kSurfaceDiffuse | kSurfaceNearSpecular | kSurfaceTransmission)) ==
            0u;
    bool resolved=true;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if constexpr(Exposure && (Width==4608 || Width==11520)) {
        if(!params.resident_storage && params.register_motion && glossy_only && params.static_view_count>1u) {
            // The central adjacent nodes of the existing eight-interval Quality
            // angular reconstruction are actual calibrated ray origins.
            const unsigned v0=params.static_view_count/2u;
            const unsigned v1=min(5u*params.static_view_count/8u,params.static_view_count-1u);
            const float f=params.focal_distance;
            const float u0=view_x[v0]/f,u1=view_x[v1]/f;
            const float c=fmaf(u0,u1,1.f)*rsqrtf(fmaf(u0,u0,1.f)*fmaf(u1,u1,1.f));
            const float sin_half_squared=.5f*fmaxf(1.f-c,0.f);
            const float r=float(surface.roughness_unorm8)*(1.f/255.f);
            const float a2=(r*r)*(r*r);
            // Reflected GGX NDF remains above half peak at this measured spacing.
            resolved=a2*.41421356237f>=sin_half_squared*(1.f-a2);
        }
    }
#endif
    if (resolved && params.responsive_angular_radiometry != 0u &&
        (params.responsive_radiometry_glossy_only == 0u || glossy_only) &&
        (surface.surface_flags & kSurfaceDiffuse) == 0u &&
        (surface.surface_flags &
         (kSurfaceGlossy | kSurfaceNearSpecular | kSurfaceTransmission)) !=
            0u) {
        const std::uint32_t response_payload =
            (glossy_only ? 1u : 0u) << kGuideResponseGlossyOnlyBit;
        set_guide_path(
            result, kGuidePathResponsive, response_payload);
    }

    const std::uint32_t view = view_id[index];
#ifndef LFS_ENCODED_PLAIN_CURRENT
    result.view_id = static_cast<std::uint16_t>(view);
#endif
    const float inverse = inverse_length[index];
    const float hit = surface.hit;
    if (view >= params.static_view_count || !isfinite(inverse) ||
        !isfinite(hit) || !finite3(surface.world_normal) ||
        !finite3(surface.albedo)) {
        fail(params, view >= params.static_view_count
            ? kFrameStatusInvalidLayout : kFrameStatusNonfinite);
        set_guide_validity(result, kInvalidGuide);
        return result;
    }
    const float origin_x = view_x[view];
    const Vec3 direction{
        (focal_x[x] - origin_x) * inverse,
        focal_y[y] * inverse,
        params.focal_distance * inverse};
    result.point = Vec3{origin_x, 0.0f, 0.0f} + direction * hit;
    const Vec3 normal = local_normal(params, surface.world_normal);
#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
    result.normal = unpack_normal(pack_normal(normal));
#else
    result.packed_normal = pack_normal(normal);
#endif

    if (!isfinite(origin_x) || !finite3(direction) ||
        !finite3(result.point) || !finite3(normal)) {
        fail(params);
        set_guide_validity(result, kInvalidGuide);
    }
    return result;
}

template<int Width>
__device__ __forceinline__ Guide build_guide(
    const EncodedAtrousNativeParams &params,
    int x,
    int y) {
    const std::uint32_t index =
        static_cast<std::uint32_t>(y * Width + x);
    const SurfaceInput surface = load_surface_input(params, index);
    return build_guide_from_surface<Width>(params, x, y, index, surface);
}

template<int Width>
__device__ __forceinline__ Guide load_guide(
    const EncodedAtrousNativeParams &params,
    int x,
    int y) {

    return build_guide<Width>(params, x, y);
}

template<int Width,bool Exposure=false,bool CaptureRaw=false>
__device__ __forceinline__ InitGuideValue initialize_guide_site(
    const EncodedAtrousNativeParams &params,
    int x,
    int y,bool owner=false) {
    const std::uint32_t index =
        static_cast<std::uint32_t>(y * Width + x);
    const auto *radiance =
        reinterpret_cast<const float4 *>(params.radiance);
    const float4 r = radiance[index];
    const SurfaceInput surface = load_surface_input(params, index);
    if constexpr(CaptureRaw) {
        if(owner && params.atlas_moments) {
            const unsigned metadata=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(params.packed_observation)[index].metadata;
            if(register_motion_surface(metadata)) {
                const bool valid=finite3({r.x,r.y,r.z}) && fmaxf(fmaxf(r.x,r.y),r.z)<=65504.f;
                const auto captured=valid?pack({r.x,r.y,r.z,1.f}):Half4{};
                store_signal(reinterpret_cast<Half4*>(params.atlas_moments),index,captured);
            }
        }
    }
    Vec3 raw{fmaxf(r.x, 0.0f), fmaxf(r.y, 0.0f), fmaxf(r.z, 0.0f)};
    const Vec3 albedo_value = surface.albedo;
    if (!finite3(raw) || !finite3(albedo_value)) {
        fail(params);
        raw = make_vec3(0.0f);
    }
    Vec3 signal = raw;
    if (surface.validity != 0u && surface.validity != kInvalidGuide) {
        const Vec3 modulation = safe_albedo(albedo_value);
        signal = {signal.x / modulation.x,
                  signal.y / modulation.y,
                  signal.z / modulation.z};
    }
    signal = clamp3(signal, 0.0f, 500.0f);
    const float luma = safe_luminance(signal);
    (void)luma;
    return {
        pack({signal.x, signal.y, signal.z,
              1.0e-4f
             }),
        build_guide_from_surface<Width,Exposure>(params, x, y, index, surface),
        surface.validity!=1u || (surface.surface_flags!=kSurfaceGlossy && surface.surface_flags!=kSurfaceNearSpecular)};
}

template<int Width,bool UniformLinear=false>
__device__ __forceinline__ float prefilter_geometry(
    const EncodedAtrousNativeParams &params,
    const Guide &center,
    Vec3 center_normal,
    const Guide &sample,
    int sample_x,
    int sample_y,
    const AngularCenterState &center_angular) {
    if constexpr(!UniformLinear) {
    const std::uint32_t center_validity = guide_validity(center);
    const std::uint32_t sample_validity = guide_validity(sample);
    if (center_validity == kInvalidGuide ||
        sample_validity == kInvalidGuide) {
        return 0.0f;
    }
    if (!responsive_compatible(center, sample)) return 0.0f;
    if (center_validity == 0u && sample_validity == 0u) return 1.0f;
    if (center_validity == 0u || sample_validity == 0u ||
        center.instance != sample.instance) {
        return 0.0f;
    }
    }
    const Vec3 delta = sample.point - center.point;
    const Vec3 sample_normal = guide_normal(sample);
    const float normal_weight = pow32(
        fmaxf(dot(center_normal, sample_normal), 0.0f));
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    const float result=normal_weight*expf(-fabsf(dot(delta,center_normal))/
        fmaxf(length(delta)*.1f,1.e-3f)+measurement_aperture_exponent(params,sample,center_angular));
#else
    const float depth_weight = expf(
        -fabsf(dot(delta, center_normal)) /
        fmaxf(length(delta) * 0.1f, 1.0e-3f));
    const float result = normal_weight * depth_weight *
        angular_compatibility_weight<Width>(
            params, center, sample, sample_x, sample_y, center_angular);
#endif
    if (!isfinite(result)) fail(params);
    return isfinite(result) ? result : 0.0f;
}

template<int Width,bool UniformLinear=false>
__device__ __forceinline__ bool atrous_geometry(
    const EncodedAtrousNativeParams &params,
    const Guide &center,
    Vec3 center_normal,
    Vec3 center_albedo,
    const Guide &sample,
    int sample_x,
    int sample_y,
    const AngularCenterState &center_angular,
    float &weight,float &geometry_exponent) {
    weight = 0.0f;
    if constexpr(!UniformLinear) {
    const std::uint32_t center_validity = guide_validity(center);
    const std::uint32_t sample_validity = guide_validity(sample);
    if (center_validity == kInvalidGuide ||
        sample_validity == kInvalidGuide) {
        return false;
    }
    if (!responsive_compatible(center, sample)) return false;
    if (center_validity == 0u && sample_validity == 0u) {
        weight = 1.0f;
        return true;
    }
    if (center_validity == 0u || sample_validity == 0u ||
        center.instance != sample.instance) {
        return false;
    }
    }
    const Vec3 delta = sample.point - center.point;
    const Vec3 sample_normal = guide_normal(sample);
    const float normal_weight = pow32(
        fmaxf(dot(center_normal, sample_normal), 0.0f));
    const Vec3 sample_albedo = guide_albedo(sample);
#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
    geometry_exponent = -fabsf(dot(delta,center_normal))*rsqrtf(fmaxf(dot(delta,delta),1.e-6f)) -
        length(center_albedo - sample_albedo) * 4.0f;
    weight = normal_weight;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    geometry_exponent+=measurement_aperture_exponent(params,sample,center_angular);
#endif
#else

    // execution fusion belongs only to its explicitly typed modules.
    const float depth_weight=expf(-fabsf(dot(delta,center_normal))*rsqrtf(fmaxf(dot(delta,delta),1.e-6f)));
    const float albedo_weight=expf(-length(center_albedo-sample_albedo)*4.0f);
    weight=normal_weight*depth_weight*albedo_weight*angular_compatibility_weight<Width>(
        params,center,sample,sample_x,sample_y,center_angular);
#endif
    if (!isfinite(weight)) {
        fail(params);
        weight = 0.0f;
    }
    return true;
}

#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
struct IrradianceFit {
    float numerator_delta{},denominator_delta{},variance_delta{};
    __device__ __forceinline__ void add(float center,float donor,float color,float weight,float variance) {
        const float d=donor-center;
        numerator_delta=fmaf(weight*d,color,numerator_delta);
        denominator_delta=fmaf(weight*d,donor+center,denominator_delta);
        variance_delta=fmaf(weight*weight*d*(donor+center),variance,variance_delta);
    }
    __device__ __forceinline__ float estimate(float response,float sum,float mass,float fallback) const {
        const float denom=fmaf(response*response,mass,denominator_delta);
        return denom>1.e-12f?fmaxf(0.f,response*fmaf(response,sum,numerator_delta)/denom):fallback;
    }
    __device__ __forceinline__ float variance(float response,float sum,float mass,float prior) const {
        const float denom=fmaf(response*response,mass,denominator_delta);
        if(denom<=1.e-12f)return prior;
        const float gain=response/denom;
        return fmaxf(0.f,fmaf(response*response,sum,variance_delta))*gain*gain;
    }
};
__device__ __forceinline__ float diffuse_response(float a) {
    return a*rsqrtf(fmaf(a,a,.01f));
}
#endif
template<int Width, int SharedWidth, bool Terminal=false, bool CachedLuma=false, bool NeighborWitness=false,bool UniformLinear=false,bool Irradiance=false>
__device__ __forceinline__ Half4 atrous_shared_site(
    const EncodedAtrousNativeParams &params,
    const Half4 *source,
    const SharedGuide *guides, const Vec3 *normals,
    int center_x,
    int center_y,
    int global_center_x,
    int global_center_y,
    int step, int global_step, const float *source_luma=nullptr) {
    const int center_index = center_y * SharedWidth + center_x;
    Vec4 center_value = unpack(source[center_index]);
    const bool statistical = params.statistical_reconstruction != 0u && global_step > 1;
    const bool center_known = !statistical || center_value.w >= 0.0f;
    bool variance_known = center_known;
    if (statistical) center_value.w = fabsf(center_value.w);
    const Guide center = load_tile_guide(guides,normals,center_index);
    const Vec3 center_normal = guide_normal(center);
    const Vec3 center_albedo = guide_albedo(center);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    Vec3 response{};IrradianceFit fits[3]{};bool material_varies=false;
    if constexpr(Irradiance)response={diffuse_response(center_albedo.x),diffuse_response(center_albedo.y),diffuse_response(center_albedo.z)};
#endif
    const AngularCenterState center_angular = UniformLinear?AngularCenterState{}:
        make_angular_center_state<Width>(
            params, center, global_center_x, global_center_y);
    const float center_luma = CachedLuma ? source_luma[center_index] : luminance(
        {center_value.x, center_value.y, center_value.z});
    const float phi = 32.0f * sqrtf(fmaxf(center_value.w, 1.0e-4f)) +
        1.0e-3f;
    const float inverse_phi = 1.0f / phi;
    Vec3 color_sum = make_vec3(0.0f);
    float variance_sum = 0.0f, carrier_sum=0.f;
    const unsigned channel=unsigned(global_center_x)%3u;
    float weight_sum = 0.0f;
    for (int oy = -2; oy <= 2; ++oy) {
#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
        #pragma unroll
#endif
        for (int ox = -2; ox <= 2; ++ox) {
            const int sx = center_x + ox * step;
            const int sy = center_y + oy * step;
            const int sample_index = sy * SharedWidth + sx;
            Vec4 sample_value = unpack(source[sample_index]);
            const bool sample_known = !statistical || sample_value.w >= 0.0f;
            if (statistical) sample_value.w = fabsf(sample_value.w);
            if (ox == 0 && oy == 0) {
                if constexpr(NeighborWitness)continue;
                const float geometry_weight = pow32(
                    fmaxf(dot(center_normal, center_normal), 0.0f));
                const float weight =
                    0.375f * 0.375f * geometry_weight;
                if constexpr(Terminal) {
                    const float value=channel==0u?sample_value.x:(channel==1u?sample_value.y:sample_value.z);
                    carrier_sum=fmaf(value,weight,carrier_sum);
                } else {
                color_sum += Vec3{sample_value.x, sample_value.y,
                                  sample_value.z} * weight;
                variance_sum += sample_value.w * weight * weight;
                }
                weight_sum += weight;
                continue;
            }
            const Guide sample = load_tile_guide(guides,normals,sample_index);
            float geometry_weight = 0.0f;
            float geometry_exponent = 0.0f;
            if (!atrous_geometry<Width,UniformLinear>(
                    params, center, center_normal, center_albedo,
                    sample, global_center_x + ox * global_step,
                    global_center_y + oy * global_step,
                    center_angular, geometry_weight, geometry_exponent)) {
                continue;
            }
            const int ax = ox < 0 ? -ox : ox;
            const int ay = oy < 0 ? -oy : oy;
            const float kx = ax == 0 ? 0.375f :
                (ax == 1 ? 0.25f : 0.0625f);
            const float ky = ay == 0 ? 0.375f :
                (ay == 1 ? 0.25f : 0.0625f);
            const float sample_luma = CachedLuma ? source_luma[sample_index] : luminance(
                {sample_value.x, sample_value.y, sample_value.z});
            const bool directional_response = !UniformLinear &&
                params.responsive_angular_radiometry != 0u &&
                guide_path_kind(center) == kGuidePathResponsive &&
                guide_path_kind(sample) == kGuidePathResponsive &&
                (params.responsive_radiometry_glossy_only == 0u ||
                 (guide_response_glossy_only(center) &&
                  guide_response_glossy_only(sample)));
            const float difference = center_luma - sample_luma;
            float residual = statistical && center_known && sample_known
                ? 0.5f * difference * difference / fmaxf(center_value.w + sample_value.w, 1.0e-20f)
                : fabsf(difference) * inverse_phi;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
            if constexpr(Irradiance) {
                if(statistical && center_known && sample_known && (sample.albedo.x!=center.albedo.x ||
                   sample.albedo.y!=center.albedo.y || sample.albedo.z!=center.albedo.z)) {
                    const auto a=guide_albedo(sample);
                    const Vec3 bs{diffuse_response(a.x),diffuse_response(a.y),diffuse_response(a.z)};
                    // Cross multiplication keeps a truly black donor defined.
                    const float cr=luminance({center_value.x*bs.x,center_value.y*bs.y,center_value.z*bs.z});
                    const float sr=luminance({sample_value.x*response.x,sample_value.y*response.y,sample_value.z*response.z});
                    const float sc=center_luma>0.f?cr/center_luma:luminance(bs);
                    const float ss=sample_luma>0.f?sr/sample_luma:luminance(response);
                    const float d=cr-sr,v=fmaf(center_value.w,sc*sc,sample_value.w*ss*ss);
                    residual=.5f*d*d/fmaxf(v,1.e-20f);
                }
            }
#endif
#ifdef LFS_ENCODED_PLAIN_CURRENT
            // exp(-plane) * exp(-albedo) * exp(-luminance): one evaluation
            // of the same mathematical weight, with FP32 roundoff verified.
            const float luminance_weight = expf(geometry_exponent - residual);
#elif !defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
            const float luminance_weight=directional_response?1.0f:expf(-residual);
#else

            // uncertainty replaces its luminance-free rule when both are known.
            const float range_exponent = directional_response && !(statistical && center_known && sample_known)
                ? 0.0f : -residual;
            const float luminance_weight = expf(geometry_exponent + range_exponent);
#endif
            const float weight =
                kx * ky * geometry_weight * luminance_weight;
            if (weight > 0.0f && !sample_known) variance_known = false;
            if constexpr(Terminal) {
                const float value=channel==0u?sample_value.x:(channel==1u?sample_value.y:sample_value.z);
                carrier_sum=fmaf(value,weight,carrier_sum);
                if constexpr(NeighborWitness) {
                    const float sw=weight*value/fmaxf(sample_luma,1.e-20f);
                    variance_sum=fmaf(sw*sw,fabsf(sample_value.w),variance_sum);
                }
            } else {
            color_sum += Vec3{sample_value.x, sample_value.y,
                              sample_value.z} * weight;
            variance_sum += sample_value.w * weight * weight;
            }
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
            if constexpr(Irradiance) {
                if(center.albedo.x!=sample.albedo.x || center.albedo.y!=sample.albedo.y || center.albedo.z!=sample.albedo.z) {
                    const Vec3 a=guide_albedo(sample);
                    fits[0].add(response.x,diffuse_response(a.x),sample_value.x,weight,sample_value.w);
                    fits[1].add(response.y,diffuse_response(a.y),sample_value.y,weight,sample_value.w);
                    fits[2].add(response.z,diffuse_response(a.z),sample_value.z,weight,sample_value.w);
                    material_varies=true;
                }
            }
#endif
            weight_sum += weight;
        }
    }
    const float inverse = 1.0f / fmaxf(weight_sum, 1.0e-4f);
    if constexpr(Terminal) {
        const float value=carrier_sum*inverse;
        if(!isfinite(value)){fail(params);return pack({0.f,0.f,0.f,1.e-4f});}
        // The normal terminal writes one canonical component. Its suffix
        // variance and the other two color components have no consumer.
        if constexpr(NeighborWitness) {
            if(!(weight_sum>1.e-4f))return pack({0.f,0.f,0.f,-1.f});
            return pack({value,value,value,fminf(65504.f,variance_sum*inverse*inverse)});
        }
        return pack({value,value,value,1.e-4f});
    }
    Vec3 filtered = color_sum * inverse;
    float variance = fmaxf(
        weight_sum * weight_sum >= 1.0e-4f
            ? variance_sum * inverse * inverse
            : variance_sum / 1.0e-4f,
        1.0e-4f);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if constexpr(Irradiance) {
        if(material_varies) {
            filtered={fits[0].estimate(response.x,color_sum.x,weight_sum,filtered.x),
                      fits[1].estimate(response.y,color_sum.y,weight_sum,filtered.y),
                      fits[2].estimate(response.z,color_sum.z,weight_sum,filtered.z)};
            const float sx=sqrtf(fits[0].variance(response.x,variance_sum,weight_sum,variance));
            const float sy=sqrtf(fits[1].variance(response.y,variance_sum,weight_sum,variance));
            const float sz=sqrtf(fits[2].variance(response.z,variance_sum,weight_sum,variance));
            const float sigma=luminance({sx,sy,sz});variance=fmaxf(sigma*sigma,1.e-4f);
        }
    }
#endif
    if (!finite3(filtered) || !isfinite(variance)) {
        fail(params);
        return pack({0.0f, 0.0f, 0.0f, 1.0e-4f});
    }
    return pack({filtered.x, filtered.y, filtered.z, variance_known ? variance : -variance});
}

template<int Width, int Height, int SharedWidth, bool CachedLuma=false,bool UniformLinear=false>
__device__ __forceinline__ Half4 prefilter_shared_site(
    const EncodedAtrousNativeParams &params, const Half4 *tile,
    const SharedGuide *guides, const Vec3 *normals, int center_x, int center_y, int x, int y, const float *source_luma=nullptr) {
    const int center_shared = center_y * SharedWidth + center_x;
    Vec4 center_value = unpack(tile[center_shared]);
    const Guide center = load_tile_guide(guides,normals,center_shared);
    const Vec3 center_normal = guide_normal(center);
    const AngularCenterState center_angular = UniformLinear?AngularCenterState{}:
        make_angular_center_state<Width>(params, center, x, y);

    const float center_luma = CachedLuma ? source_luma[center_shared] : safe_luminance(
        {center_value.x, center_value.y, center_value.z});
    Vec3 filtered_sum = make_vec3(0.0f);
    float filtered_weight = 0.0f;
    float moment1_sum = 0.0f;
    float moment2_sum = 0.0f;
    float geometry_sum = 0.0f;
    Vec3 neighbor_sum = make_vec3(0.0f);
    float neighbor_weight = 0.0f;
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            const int sx = x + ox;
            const int sy = y + oy;
            if (!UniformLinear && (sx < 0 || sx >= Width || sy < 0 || sy >= Height)) continue;
            const int local_x = center_x + ox;
            const int local_y = center_y + oy;
            const Vec4 sample_value = unpack(
                tile[local_y * SharedWidth + local_x]);
            const Guide sample = load_tile_guide(guides,normals,local_y * SharedWidth + local_x);
            const int ax = ox < 0 ? -ox : ox;
            const int ay = oy < 0 ? -oy : oy;
            const float kernel = ax == 0 && ay == 0 ? 0.25f :
                (ax == 0 || ay == 0 ? 0.125f : 0.0625f);
            const float sample_luma = CachedLuma ? source_luma[local_y*SharedWidth+local_x] : safe_luminance(
                {sample_value.x, sample_value.y, sample_value.z});
            const bool center_tap = ox == 0 && oy == 0;
            const float geometry = center_tap
                ? pow32(fmaxf(dot(center_normal, center_normal), 0.0f))
                : prefilter_geometry<Width,UniformLinear>(
                      params, center, center_normal, sample, sx, sy,
                      center_angular);
            const bool directional_response = !UniformLinear && !center_tap &&
                params.responsive_angular_radiometry != 0u &&
                guide_path_kind(center) == kGuidePathResponsive &&
                guide_path_kind(sample) == kGuidePathResponsive &&
                (params.responsive_radiometry_glossy_only == 0u ||
                 (guide_response_glossy_only(center) &&
                  guide_response_glossy_only(sample)));
            const float radiometric = directional_response
                ? 1.0f
                : expf(
                      -fabsf(center_luma - sample_luma) /
                          (center_luma + sample_luma + 0.002f) * 3.5f);
            const float total = center_tap
                ? kernel * geometry
                : kernel * geometry * radiometric;
            filtered_sum += Vec3{sample_value.x, sample_value.y,
                                 sample_value.z} * total;
            filtered_weight += total;

            moment1_sum += sample_luma * kernel * geometry;
            moment2_sum += sample_luma * sample_luma * kernel * geometry;
            geometry_sum += kernel * geometry;
            if ((ox != 0 || oy != 0) && geometry > 0.1f) {
                neighbor_sum += Vec3{sample_value.x, sample_value.y,
                                     sample_value.z} * geometry;
                neighbor_weight += geometry;

            }
        }
    }
    Vec3 firefly{center_value.x, center_value.y, center_value.z};
    bool compress_isolated = neighbor_weight > 0.5f;
#if defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
    // A new history retains the original single-frame heavy-tail protection.
    // Warm radial-log measurements must not be censored a second time here.
    // This is frame history availability, not per-pixel history acceptance.
    if (params.temporal_history_available != 0u &&
        guide_validity(center) == 1u && guide_response_glossy_only(center)) {
        compress_isolated = false;
    }
#endif
    if (compress_isolated) {
        const Vec3 mean = neighbor_sum / fmaxf(neighbor_weight, 1.0e-3f);
        const float mean_luma = fmaxf(safe_luminance(mean), 0.01f);
        const float isolation = center_luma / mean_luma;
        const float history_t = 1.0f / 48.0f;
        const float sigma_multiplier = 20.0f + history_t * (10.0f - 20.0f);
        const float temporal_threshold = fmaxf(center_luma, 0.01f) +
            sigma_multiplier * fmaxf(sqrtf(fmaxf(center_value.w, 0.0f)), 0.5f);
        const float spatial_threshold = mean_luma * 3.0f;
        const float blend = isolation > 3.0f ? 0.7f : 0.3f;
        const float threshold = fmaxf(
            temporal_threshold + blend *
                (spatial_threshold - temporal_threshold), 0.1f);
        if (isolation > 3.0f && center_luma > threshold) {
            const float ratio = center_luma / threshold;
            const float compressed =
                1.0f + asinhf((ratio - 1.0f) * 2.0f) / 2.0f * 0.85f;
            const float scale = fminf(
                threshold * compressed / fmaxf(center_luma, 1.0e-3f), 1.0f);
            firefly = firefly * scale;
        }
    }
    Vec3 filtered = filtered_sum / fmaxf(filtered_weight, 1.0e-4f);
    const float spatial_m1 = moment1_sum / fmaxf(geometry_sum, 1.0e-4f);
    const float spatial_variance = fmaxf(
        moment2_sum / fmaxf(geometry_sum, 1.0e-4f) -
            spatial_m1 * spatial_m1,
        center_luma * 0.08f);
    float output_variance = fmaxf(
        center_value.w + 0.3f *
            (fmaxf(spatial_variance, 0.001f) - center_value.w),
        0.001f);
    const Vec3 output = clamp3(
        firefly + (filtered - firefly) * 0.24f,
        0.0f, 500.0f);
    return pack({output.x, output.y, output.z, output_variance});
}

template<int Width, int Height>
__device__ __forceinline__ void dispatch_init_prefilter(
    EncodedAtrousNativeParams params,
    Half4 *tile,
    SharedGuide *guides, Vec3 *normals) {
    constexpr int TileWidth = 32;
    constexpr int TileHeight = LFS_ENCODED_ATROUS_TILE_ROWS;
    constexpr int SharedWidth = TileWidth + 2;
    constexpr int SharedHeight = TileHeight + 2;
    const int thread = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    for (int item = thread; item < SharedWidth * SharedHeight;
         item += TileWidth * TileHeight) {
        const int local_x = item % SharedWidth;
        const int local_y = item / SharedWidth;
        const int x = static_cast<int>(blockIdx.x) * TileWidth + local_x - 1;
        const int y = static_cast<int>(blockIdx.y) * TileHeight + local_y - 1;
        if (x >= 0 && x < Width && y >= 0 && y < Height) {
            const InitGuideValue initialized =
                initialize_guide_site<Width>(params, x, y);
            tile[item] = initialized.signal;
            store_tile_guide(guides,normals,item,initialized.guide);
        } else {
            tile[item] = pack({0.0f, 0.0f, 0.0f, 0.0f});
            Guide invalid{};set_guide_validity(invalid,kInvalidGuide);
            store_tile_guide(guides,normals,item,invalid);
        }
    }
    __syncthreads();

    const int x = static_cast<int>(blockIdx.x) * TileWidth + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * TileHeight + threadIdx.y;
    if (x >= Width || y >= Height) return;
    const int center_shared =
        (static_cast<int>(threadIdx.y) + 1) * SharedWidth +
        static_cast<int>(threadIdx.x) + 1;
    store_signal(reinterpret_cast<Half4 *>(params.scratch_a), y * Width + x,
        prefilter_shared_site<Width, Height, SharedWidth>(params, tile, guides, normals,
            static_cast<int>(threadIdx.x) + 1,
            static_cast<int>(threadIdx.y) + 1, x, y));
}


__device__ __forceinline__ bool reflective_site(unsigned metadata) {
    const unsigned cls = (metadata >> denoise_native::kPackedSurfaceFlagsShift) &
                         denoise_native::kPackedSurfaceFlagsMask;
    return (metadata & 3u) == 1u && (cls == kSurfaceGlossy || cls == kSurfaceNearSpecular);
}

#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
#include "measured_density_precision.cuh"
#endif
#ifdef LFS_ENCODED_PLAIN_CURRENT
__device__ __forceinline__ float selected(Vec3 c,unsigned k){return k==0u?c.x:(k==1u?c.y:c.z);}
#include "present_encoded_carrier.cuh"
#endif
#include "measured_radiometric_plane.cuh"
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
#include "plane_radiance_moments.cuh"
#endif
template<int Width, int Height, int Step, int TileWidth, int TileHeight, int Spacing = 1,bool CarrierOnly=false,bool RouteDiffuse=false,bool CacheGrid=false>
__device__ __forceinline__ void dispatch_atrous(
    EncodedAtrousNativeParams params,
    const Half4 *global_source,
    Half4 *global_destination,
    Half4 *tile,
    SharedGuide *guides, Vec3 *normals,float *luma=nullptr) {
    constexpr int Halo = (Step / Spacing) * 2;
    constexpr int SharedWidth = TileWidth + Halo * 2;
    constexpr int SharedHeight = TileHeight + Halo * 2;
    const int thread = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
    if constexpr (Step == 2) {
        const unsigned block_x = blockIdx.x * TileWidth * Spacing + blockIdx.z % Spacing;
        const unsigned block_y = blockIdx.y * TileHeight * Spacing + blockIdx.z / Spacing;
#ifdef LFS_ENCODED_PLAIN_CURRENT
        if constexpr (CarrierOnly) {
            // Both 32x16 output tiles must exclude a grid carrier. Mixed
            // subtiles keep their original per-parity center/halo evaluation.
            if (!params.resident_storage && params.register_motion && params.deterministic_environment) {
                const unsigned group=(block_y/32u)*unsigned((Width+127)/128)+block_x/128u;
                const unsigned shift=((block_y/16u)&1u)*4u+((block_x/32u)&2u);
                if ((reinterpret_cast<const unsigned*>(params.frame_status)[1u+group]&(3u<<shift))==0u) return;
            }
        }
#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        const unsigned group=(block_y/32u)*unsigned((Width+127)/128)+block_x/128u;
        const unsigned coverage = reinterpret_cast<const unsigned*>(params.frame_status)[1u+group];
        if constexpr (RouteDiffuse) {
            if ((coverage & (1u << regular_r2_bit(block_x,block_y))) == 0u) return;
            if constexpr(CacheGrid && TileWidth==32 && TileHeight==8 && Spacing==2 && Width==11520) {
                if(params.register_motion==1u && !params.resident_storage && !params.hold_same && params.regular_repeat_state &&
                   reconstruct_lambertian_plane<Width,Height>(params,global_source,global_destination,tile,guides,normals,luma))return;
            }
        } else if (coverage == 0u) return;
#endif
        // Routed Q receives exact ownership for this block and parity from M.
        if constexpr (!RouteDiffuse) {
        const unsigned gx=block_x+threadIdx.x*Spacing, gy=block_y+threadIdx.y*Spacing;
        const auto *obs=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(params.packed_observation);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        bool has_grid_site=false;
        if(gx<Width && gy<Height) {
            const unsigned md=obs[gy*Width+gx].metadata;
            has_grid_site=!reflective_site(md);
            if(!params.resident_storage && params.register_motion && params.deterministic_environment && (md&3u)==ssat_native::kPackedSurfaceDomainEnvironment)
                has_grid_site=false;
        }
#else
        bool has_grid_site=false;
        if(gx<Width && gy<Height) {
            const unsigned i=gy*Width+gx,md=obs[i].metadata;
            has_grid_site=!reflective_site(md);
#ifdef LFS_ENCODED_PLAIN_CURRENT
            if(!params.resident_storage && params.register_motion && params.deterministic_environment && (md&3u)==ssat_native::kPackedSurfaceDomainEnvironment) {
                // As in the regular terminal presenter, a primary deterministic
                // miss is the original raw radiance. No spatial state consumes it.
                present_encoded_carrier<false,Width>(params,i,md,pack({0.f,0.f,0.f,1.e-4f}));
                has_grid_site=false;
            }
#endif
        }
#endif
        if (!__syncthreads_or(has_grid_site)) return;
        }
    }
#endif
    for (int item = thread; item < SharedWidth * SharedHeight;
         item += TileWidth * TileHeight) {
        const int local_x = item % SharedWidth;
        const int local_y = item / SharedWidth;
        const int x = (static_cast<int>(blockIdx.x) * TileWidth + local_x - Halo) * Spacing + blockIdx.z % Spacing;
        const int y = (static_cast<int>(blockIdx.y) * TileHeight + local_y - Halo) * Spacing + blockIdx.z / Spacing;
        if (x >= 0 && x < Width && y >= 0 && y < Height) {
            tile[item] = load_signal(global_source, y * Width + x);
            store_tile_guide(guides,normals,item,load_guide<Width>(params, x, y));
        } else {
            tile[item] = pack({0.0f, 0.0f, 0.0f, 0.0f});
            Guide invalid{};set_guide_validity(invalid,kInvalidGuide);
            store_tile_guide(guides,normals,item,invalid);
        }
        if constexpr(CacheGrid) {
            const auto v=unpack(tile[item]);
            luma[item]=luminance({v.x,v.y,v.z});
        }
    }
    __syncthreads();
    bool uniform=false,varied_albedo=false;
    bool flat_constant=false;
    if constexpr(CacheGrid) {
        bool match=true,albedo_match=true;
        const auto ref=load_tile_guide(guides,normals,Halo*SharedWidth+Halo);
        const unsigned instance=ref.instance;
        for(int k=thread;k<SharedWidth*SharedHeight;k+=TileWidth*TileHeight) {
            const auto v=load_tile_guide(guides,normals,k);
            match=match && guide_validity(v)==1u && guide_path_kind(v)==0u && v.instance==instance;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
            albedo_match=albedo_match && ref.albedo.x==v.albedo.x && ref.albedo.y==v.albedo.y && ref.albedo.z==v.albedo.z;
#endif
        }
        uniform=__syncthreads_and(match);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        varied_albedo= !__syncthreads_and(albedo_match);
        if constexpr(Step==2 && RouteDiffuse && (Width==4608 || Width==11520)) {
            if(uniform && !varied_albedo && !params.hold_same && params.register_motion==1u)
                flat_constant=one_ray_plane<SharedWidth,SharedHeight,TileWidth*TileHeight>(guides,normals,Halo*SharedWidth+Halo,thread);
        }
#endif
    }
    const int x = (static_cast<int>(blockIdx.x) * TileWidth + threadIdx.x) * Spacing + blockIdx.z % Spacing;
    const int y = (static_cast<int>(blockIdx.y) * TileHeight + threadIdx.y) * Spacing + blockIdx.z / Spacing;
    if (x >= Width || y >= Height) return;
#if defined(LFS_ENCODED_GLOSSY_MEASUREMENT)||defined(LFS_ENCODED_PLAIN_CURRENT)
    if constexpr (Step==2) {
        const auto *obs=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(params.packed_observation);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        const unsigned md=obs[y*Width+x].metadata;
        if(reflective_site(md)) {
            if constexpr(RouteDiffuse) {
                if((md&511u)==(1u|(kSurfaceNearSpecular<<2u))) {
                    const auto *g=reinterpret_cast<const denoise_native::TemporalSample32*>(params.regular_measurement_geometry);
                    if(g[y*Width+x].sigma_albedo.x>=to_half(1.f/3.f)) {
                        const auto w=atrous_shared_site<Width,SharedWidth,true,false,true>(params,tile,guides,normals,
                            int(threadIdx.x)+Halo,int(threadIdx.y)+Halo,x,y,Step/Spacing,Step);
                        store_signal(global_destination,y*Width+x,w);
                    }
                }
            }
            return;
        }
        if(!params.resident_storage && params.register_motion && params.deterministic_environment && (md&3u)==ssat_native::kPackedSurfaceDomainEnvironment)
            return;
#else
        const unsigned md=obs[y*Width+x].metadata;
        if(reflective_site(md))return;
#ifdef LFS_ENCODED_PLAIN_CURRENT
        if(!params.resident_storage && params.register_motion && params.deterministic_environment && (md&3u)==ssat_native::kPackedSurfaceDomainEnvironment)return;
#endif
#endif
    }
#endif
    const int sx=int(threadIdx.x)+Halo,sy=int(threadIdx.y)+Halo;
    Half4 result{};
    if constexpr(CacheGrid) {
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        const auto md=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(params.packed_observation)[y*Width+x].metadata;
        if(flat_constant && (md&511u)==(1u|(kSurfaceDiffuse<<2u)))
            result=integrate_radiometric_plane<SharedWidth,true,false>(tile,luma,guides,normals,sx,sy);
        else if(varied_albedo && !params.hold_same && params.register_motion && !params.resident_storage && (md&511u)==(1u|(kSurfaceDiffuse<<2u)))
            result=atrous_shared_site<Width,SharedWidth,CarrierOnly,true,false,false,true>(
                params,tile,guides,normals,sx,sy,x,y,Step/Spacing,Step,luma);
        else
#endif
        if(uniform)result=atrous_shared_site<Width,SharedWidth,CarrierOnly,true,false,true>(
            params,tile,guides,normals,sx,sy,x,y,Step/Spacing,Step,luma);
        else result=atrous_shared_site<Width,SharedWidth,CarrierOnly,true>(
            params,tile,guides,normals,sx,sy,x,y,Step/Spacing,Step,luma);
    } else result=atrous_shared_site<Width,SharedWidth,CarrierOnly>(
        params,tile,guides,normals,sx,sy,x,y,Step/Spacing,Step);
#ifdef LFS_ENCODED_PLAIN_CURRENT
    if constexpr(Step==2) {
        const unsigned index=y*Width+x;
        const auto md=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(params.packed_observation)[index].metadata;
        if(params.resident_storage && params.register_motion && register_motion_surface(md))
            store_signal(global_destination,index,result);
        present_encoded_carrier<false,Width>(params,index,md,result);
        return;
    }
#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if constexpr(Step==2 && RouteDiffuse) {
        if(params.register_motion && !params.hold_same) {
            const unsigned i=y*Width+x;
            const auto o=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(params.packed_observation)[i];
            const auto center=tile[(threadIdx.y+Halo)*SharedWidth+threadIdx.x+Halo];
            if((o.metadata&511u)==(1u|(kSurfaceDiffuse<<2u)) && from_half(center.w)>=0.f) {
                const unsigned ch=i%3u;
                const auto col=unpack(result);
                const float z=ch==0u?col.x:(ch==1u?col.y:col.z);
                const auto albedo=o.linear_albedo_rgb_diffuse_factor;
                const float a=from_half(ch==0u?albedo.x:(ch==1u?albedo.y:albedo.z));
                auto *panel=reinterpret_cast<float*>(params.canonical);
                const float physical=from_half(to_half(z*params.final_energy))*sqrtf(fmaf(a,a,.01f));
                panel[4u*(i/3u)+ch]=physical;
                if(ch==2u)panel[4u*(i/3u)+3u]=1.f;
            }
        }
    }
#endif
    store_signal(global_destination,y*Width+x,result);
}

#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)

// Fuse only execution: materialize the original Half4 prefilter rounding in
// shared memory, then evaluate the unchanged step1 stencil. Halo work and
// shared bytes are explicit; no intermediate global prefilter write is needed.
#include "unresolved_observation_tile.cuh"
template<int Width, int Height,bool DirectEnvironment=false,bool CachedLuma=false,bool CaptureHeld=false>
__device__ __forceinline__ void dispatch_init_step1(
    const EncodedAtrousNativeParams &params,
    Half4 *raw, Half4 *filtered, SharedGuide *guides, Vec3 *normals,
    float *raw_luma=nullptr,float *filtered_luma=nullptr) {
    constexpr int SW = 38, SH = 22, Count = SW * SH;
    const int thread = threadIdx.y * 32 + threadIdx.x;
    const int base_x = blockIdx.x * 32 - 3;
    const int base_y = blockIdx.y * 16 - 3;
#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
    if constexpr(DirectEnvironment && !CaptureHeld && Width==11520) {
        if(!params.resident_storage && params.register_motion && !params.hold_same && params.deterministic_environment) {
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
            if(measure_uniform_diffuse_plane<Width,Height,true>(params,raw,filtered,guides))return;
#else
            // The plane owns a plain grid carrier. Publish the same raster
            // coverage bit normally computed by the per-site initialization.
            if(measure_uniform_diffuse_plane<Width,Height,false>(params,raw,filtered,guides)) {
                if(thread==0) {
                    const unsigned columns=(Width+127u)/128u;
                    const unsigned group=(unsigned(blockIdx.y)/2u)*columns+unsigned(blockIdx.x)/4u;
                    const unsigned bit=(unsigned(blockIdx.y)&1u)*4u+(unsigned(blockIdx.x)&3u);
                    atomicOr(reinterpret_cast<unsigned*>(params.frame_status)+1u+group,1u<<bit);
                }
                return;
            }
#endif
        }
    }
#endif
    bool needs_grid=false;
    for (int i = thread; i < Count; i += 512) {
        const int x = base_x + i % SW, y = base_y + i / SW;
        if (x >= 0 && x < Width && y >= 0 && y < Height) {
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
            constexpr bool capture_acquired=DirectEnvironment && !CaptureHeld && (Width==4608 || Width==11520);
            const unsigned owner_x=unsigned(i%SW)-3u,owner_y=unsigned(i/SW)-3u;
            const auto value=initialize_guide_site<Width,true,capture_acquired>(
                params,x,y,owner_x<32u && owner_y<16u);
#else
            const InitGuideValue value = initialize_guide_site<Width,true>(params, x, y);
#endif
            raw[i] = value.signal;
            store_tile_guide(guides,normals,i,value.guide);
            if constexpr (CachedLuma) {
                // A neighboring ES1 halo does not own any center in this block.
                const unsigned tx=unsigned(i%SW)-3u,ty=unsigned(i/SW)-3u;
                needs_grid|=value.grid_site && tx<32u && ty<16u;
            }
        } else {
            raw[i] = {};
            Guide invalid{};set_guide_validity(invalid,kInvalidGuide);
            store_tile_guide(guides,normals,i,invalid);
        }
        if constexpr (CachedLuma) {
            const Vec4 v=unpack(raw[i]);
            raw_luma[i]=safe_luminance({v.x,v.y,v.z});
        }
    }
    if constexpr (CachedLuma) {
        const bool block_needs_grid=__syncthreads_or(needs_grid);
        if (thread==0 && block_needs_grid) {
            const unsigned columns=(Width+127u)/128u;
            const unsigned group=(unsigned(blockIdx.y)/2u)*columns+unsigned(blockIdx.x)/4u;
            const unsigned bit=(unsigned(blockIdx.y)&1u)*4u+(unsigned(blockIdx.x)&3u);
            atomicOr(reinterpret_cast<unsigned*>(params.frame_status)+1u+group,1u<<bit);
        }
    } else {
        __syncthreads();
    }
#ifdef LFS_ENCODED_PLAIN_CURRENT
    if constexpr(DirectEnvironment && !CaptureHeld && CachedLuma && (Width==4608 || Width==11520)) {
        if(!params.resident_storage && params.register_motion && !params.hold_same && params.deterministic_environment &&
           measure_uniform_linear_tile<Width,Height>(params,raw,filtered,guides,normals))return;
    }
#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if constexpr(DirectEnvironment && !CaptureHeld && !CachedLuma && (Width==4608 || Width==11520)) {
        if(!params.resident_storage && params.register_motion && !params.hold_same && params.deterministic_environment &&
           measure_uniform_linear_tile<Width,Height>(params,raw,filtered,guides,normals))return;
    }
#endif
    if constexpr (CachedLuma) {
        // Preserve both rounded outputs for this thread until the prefilter
        // has consumed the whole raw/luminance tile. Switching union storage
        // after the barrier also protects donors belonging to another warp.
        Half4 pending[2]{};
        #pragma unroll
        for (int pass=0;pass<2;++pass) {
            const int i=thread+pass*512;
            if (i>=36*20) continue;
            const int sx=1+i%36,sy=1+i/36,local=sy*SW+sx;
            const int x=base_x+sx,y=base_y+sy;
            if (x<0||x>=Width||y<0||y>=Height) continue;
            const bool direct=params.register_motion && params.deterministic_environment &&
                guide_validity(load_tile_guide(guides,normals,local))==0u;
            pending[pass]=direct?raw[local]:prefilter_shared_site<Width,Height,SW,true>(
                params,raw,guides,normals,sx,sy,x,y,raw_luma);
        }
        __syncthreads();
        #pragma unroll
        for (int pass=0;pass<2;++pass) {
            const int i=thread+pass*512;
            if (i>=36*20) continue;
            const int local=(1+i/36)*SW+1+i%36;
            const auto value=pending[pass];
            const auto v=unpack(value);
            filtered[local]=value;
            filtered_luma[local]=luminance({v.x,v.y,v.z});
        }
    } else {
    for (int i = thread; i < 36 * 20; i += 512) {
        const int sx = 1 + i % 36, sy = 1 + i / 36;
        const int x = base_x + sx, y = base_y + sy;
        if constexpr(DirectEnvironment) {
        // Registered deterministic background is presented from the original
        // ray. Geometry excludes environment donors from every surface stencil.
        const int local=sy*SW+sx;
        if(params.register_motion && params.deterministic_environment &&
           guide_validity(load_tile_guide(guides,normals,local))==0u) {
            filtered[local]=raw[local];
            continue;
        }
        }
        filtered[sy * SW + sx] = x >= 0 && x < Width && y >= 0 && y < Height
            ? prefilter_shared_site<Width, Height, SW>(params, raw, guides, normals, sx, sy, x, y)
            : Half4{};
    }
    }
    __syncthreads();
    const int x = base_x + 3 + threadIdx.x, y = base_y + 3 + threadIdx.y;
    if (x < Width && y < Height) {
        if constexpr(DirectEnvironment) {
        const int local=(int(threadIdx.y)+3)*SW+int(threadIdx.x)+3;
        if(params.register_motion && params.deterministic_environment &&
           guide_validity(load_tile_guide(guides,normals,local))==0u) {
            store_signal(reinterpret_cast<Half4*>(params.scratch_b),y*Width+x,CachedLuma?filtered[local]:raw[local]);
            return;
        }
        }
        auto step1_params=params;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        // The separate path switches only the recursive angular scope after
        // prefilter. Preserve that boundary in this shared Half4 execution.
        step1_params.angular_compatibility_responsive_only=1u;
#endif
        auto measured=atrous_shared_site<Width, SW,false,CachedLuma>(step1_params, filtered, guides, normals,
            threadIdx.x + 3, threadIdx.y + 3, x, y, 1, 1, filtered_luma);
        const unsigned i=y*Width+x;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        if constexpr(Width==4608 || Width==11520) {
            if(!params.resident_storage && params.register_motion) {
                const Guide center=load_tile_guide(guides,normals,(int(threadIdx.y)+3)*SW+int(threadIdx.x)+3);
                if(guide_path_kind(center)!=kGuidePathResponsive)measured.w|=0x8000u;
            }
        }
#endif
        store_signal(reinterpret_cast<Half4*>(params.scratch_b),i,measured);
        if constexpr(CaptureHeld) {
            auto &fresh=reinterpret_cast<Half4*>(params.hold_state)[i];
            fresh.x=measured.x;fresh.y=measured.y;fresh.z=measured.z;
        }
    }
}
#endif

template<int Width, int Height>
__device__ __forceinline__ bool valid_params(
    const EncodedAtrousNativeParams &params) {
    return
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        glossy_measurement_compatible(params) &&
#endif
        params.radiance != 0u && params.packed_observation != 0u &&
        params.static_view_id != 0u &&
        params.static_inverse_ray_length != 0u &&
        params.static_view_x != 0u && params.static_focal_x != 0u &&
        params.static_focal_y != 0u && params.scratch_a != 0u &&
        params.scratch_b != 0u &&
        params.frame_status != 0u &&
        params.static_view_count != 0u &&
        params.count == static_cast<std::uint32_t>(Width * Height) &&
        retained_current_compatible(params) &&
        params.focal_distance > 0.0f &&
        isfinite(params.focal_distance);
}

#ifdef LFS_ENCODED_PLAIN_CURRENT
// Host and device validate before folding uniform disabled features away.
__device__ __forceinline__ EncodedAtrousNativeParams plain_parameters(
    EncodedAtrousNativeParams p) {

    p.angular_compatibility_spatial = 0u;
    p.responsive_angular_radiometry = 0u;
    p.angular_compatibility_aperture_only = 0u;

    p.responsive_radiometry_glossy_only = 0u;
    p.angular_compatibility_responsive_only = 0u;

    return p;
}

#endif

#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
__device__ __forceinline__ EncodedAtrousNativeParams measurement_parameters(EncodedAtrousNativeParams p) {

    p.angular_compatibility_spatial=1u;p.responsive_angular_radiometry=1u;
    p.angular_compatibility_aperture_only=1u;p.responsive_radiometry_glossy_only=1u;
    p.statistical_reconstruction=1u;return p;
}
#define LFS_GENERAL_PARAMETERS(params) measurement_parameters(params)
#else
#define LFS_GENERAL_PARAMETERS(params) params
#endif

} // namespace

#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
#define LFS_STEP2_ENTRY(Name) lfs_encoded_atrous_step2_coverage_##Name
#define LFS_STEP2_ROWS 8
#define LFS_STEP2_SPACING 2
#define LFS_STEP2_HALO2 4
#else
#define LFS_STEP2_ENTRY(Name) lfs_encoded_atrous_step2_##Name
#define LFS_STEP2_ROWS LFS_ENCODED_ATROUS_TILE_ROWS
#define LFS_STEP2_SPACING 1
#define LFS_STEP2_HALO2 8
#endif

#ifdef LFS_ENCODED_PLAIN_CURRENT

#define LFS_DEFINE_PROFILE(Name, Width, Height)                                      \
extern "C" __global__ void lfs_encoded_atrous_init_prefilter_##Name(                \
    EncodedAtrousNativeParams params) {                                              \
    __shared__ Half4 tile[(32 + 2) * (LFS_ENCODED_ATROUS_TILE_ROWS + 2)];                                     \
    __shared__ SharedGuide guides[(32 + 2) * (LFS_ENCODED_ATROUS_TILE_ROWS + 2)]; LFS_SHARED_NORMALS(normals, (32 + 2) * (LFS_ENCODED_ATROUS_TILE_ROWS + 2))                                   \
    if (!valid_params<Width, Height>(params) || !plain_current_compatible(params)) {                                      \
        if (threadIdx.x == 0u && threadIdx.y == 0u) fail(                            \
            params, kFrameStatusInvalidLayout);                                     \
        return;                                                                      \
    }                                                                                \
    dispatch_init_prefilter<Width, Height>(plain_parameters(params), tile, guides, normals);                    \
}                                                                                    \
extern "C" __global__ void lfs_encoded_atrous_init_step1_##Name( \
    EncodedAtrousNativeParams params) { \
    __shared__ Half4 raw[38 * 22], filtered[38 * 22]; \
    __shared__ SharedGuide guides[38 * 22]; LFS_SHARED_NORMALS(normals, 38 * 22) \
    if (!valid_params<Width, Height>(params) || !plain_current_compatible(params)) { \
        if (threadIdx.x == 0u && threadIdx.y == 0u) fail(params, kFrameStatusInvalidLayout); \
        return; \
    } \
    dispatch_init_step1<Width,Height>(plain_parameters(params),raw,filtered,guides,normals); \
} \
extern "C" __global__ void lfs_encoded_atrous_step1_##Name(                         \
    EncodedAtrousNativeParams params) {                                              \
    __shared__ Half4 tile[(32 + 4) * (8 + 4)];                                     \
    __shared__ SharedGuide guides[(32 + 4) * (8 + 4)]; LFS_SHARED_NORMALS(normals, (32 + 4) * (8 + 4))                                   \
    if (!valid_params<Width, Height>(params) || !plain_current_compatible(params)) {                                      \
        if (threadIdx.x == 0u && threadIdx.y == 0u) fail(                            \
            params, kFrameStatusInvalidLayout);                                     \
        return;                                                                      \
    }                                                                                \
    dispatch_atrous<Width, Height, 1, 32, 8>(plain_parameters(params),                               \
        reinterpret_cast<const Half4 *>(params.scratch_a),                           \
        reinterpret_cast<Half4 *>(params.scratch_b), tile, guides, normals);                  \
}                                                                                    \
extern "C" __global__ void LFS_STEP2_ENTRY(Name)(                         \
    EncodedAtrousNativeParams params) {                                              \
    __shared__ Half4 tile[(32 + 4) * (8 + 4)];                                     \
    __shared__ SharedGuide guides[(32 + 4) * (8 + 4)]; LFS_SHARED_NORMALS(normals, (32 + 4) * (8 + 4))                                   \
    if (!valid_params<Width, Height>(params) || !plain_current_compatible(params)) {                                      \
        if (threadIdx.x == 0u && threadIdx.y == 0u) fail(                            \
            params, kFrameStatusInvalidLayout);                                     \
        return;                                                                      \
    }                                                                                \
    dispatch_atrous<Width, Height, 2, 32, 8, 2>(plain_parameters(params),                               \
        reinterpret_cast<const Half4 *>(params.scratch_b),                           \
        reinterpret_cast<Half4 *>(params.scratch_a), tile, guides, normals);                  \
}
#else
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
#define LFS_QUALITY_FUSED_ENTRY(Name,Width,Height) \
extern "C" __global__ void lfs_encoded_atrous_init_step1_##Name(EncodedAtrousNativeParams p){ \
 __shared__ Half4 raw[38*22],filtered[38*22];__shared__ SharedGuide guides[38*22];LFS_SHARED_NORMALS(normals,38*22) \
 if(!valid_params<Width,Height>(p)){if(threadIdx.x==0u&&threadIdx.y==0u)fail(p,kFrameStatusInvalidLayout);return;} \
 dispatch_init_step1<Width,Height>(measurement_parameters(p),raw,filtered,guides,normals); }
LFS_QUALITY_FUSED_ENTRY(2k,4608,2048)
LFS_QUALITY_FUSED_ENTRY(4k,11520,2160)
LFS_QUALITY_FUSED_ENTRY(8k,23040,4320)
#undef LFS_QUALITY_FUSED_ENTRY
#endif

#define LFS_DEFINE_PROFILE(Name, Width, Height)                                      \
extern "C" __global__ void lfs_encoded_atrous_init_prefilter_##Name(                \
    EncodedAtrousNativeParams params) {                                              \
    __shared__ Half4 tile[(32 + 2) * (LFS_ENCODED_ATROUS_TILE_ROWS + 2)];                                     \
    __shared__ SharedGuide guides[(32 + 2) * (LFS_ENCODED_ATROUS_TILE_ROWS + 2)]; LFS_SHARED_NORMALS(normals, (32 + 2) * (LFS_ENCODED_ATROUS_TILE_ROWS + 2))                                   \
    if (!valid_params<Width, Height>(params)) {                                      \
        if (threadIdx.x == 0u && threadIdx.y == 0u) fail(                            \
            params, kFrameStatusInvalidLayout);                                     \
        return;                                                                      \
    }                                                                                \
    dispatch_init_prefilter<Width, Height>(LFS_GENERAL_PARAMETERS(params), tile, guides, normals);                    \
}                                                                                    \
extern "C" __global__ void lfs_encoded_atrous_step1_##Name(                         \
    EncodedAtrousNativeParams params) {                                              \
    __shared__ Half4 tile[(32 + 4) * (LFS_ENCODED_ATROUS_TILE_ROWS + 4)];                                     \
    __shared__ SharedGuide guides[(32 + 4) * (LFS_ENCODED_ATROUS_TILE_ROWS + 4)]; LFS_SHARED_NORMALS(normals, (32 + 4) * (LFS_ENCODED_ATROUS_TILE_ROWS + 4))                                   \
    if (!valid_params<Width, Height>(params)) {                                      \
        if (threadIdx.x == 0u && threadIdx.y == 0u) fail(                            \
            params, kFrameStatusInvalidLayout);                                     \
        return;                                                                      \
    }                                                                                \
    dispatch_atrous<Width, Height, 1, 32, LFS_ENCODED_ATROUS_TILE_ROWS>(LFS_GENERAL_PARAMETERS(params),                               \
        reinterpret_cast<const Half4 *>(params.scratch_a),                           \
        reinterpret_cast<Half4 *>(params.scratch_b), tile, guides, normals);                  \
}                                                                                    \
extern "C" __global__ void LFS_STEP2_ENTRY(Name)(                         \
    EncodedAtrousNativeParams params) {                                              \
    __shared__ Half4 tile[(32 + LFS_STEP2_HALO2) * (LFS_STEP2_ROWS + LFS_STEP2_HALO2)];                                     \
    __shared__ SharedGuide guides[(32 + LFS_STEP2_HALO2) * (LFS_STEP2_ROWS + LFS_STEP2_HALO2)]; LFS_SHARED_NORMALS(normals, (32 + LFS_STEP2_HALO2) * (LFS_STEP2_ROWS + LFS_STEP2_HALO2))                                   \
    if (!valid_params<Width, Height>(params)) {                                      \
        if (threadIdx.x == 0u && threadIdx.y == 0u) fail(                            \
            params, kFrameStatusInvalidLayout);                                     \
        return;                                                                      \
    }                                                                                \
    dispatch_atrous<Width, Height, 2, 32, LFS_STEP2_ROWS, LFS_STEP2_SPACING>(LFS_GENERAL_PARAMETERS(params),                               \
        reinterpret_cast<const Half4 *>(params.scratch_b),                           \
        reinterpret_cast<Half4 *>(params.scratch_a), tile, guides, normals);                  \
}

#endif

LFS_DEFINE_PROFILE(2k, 4608, 2048)
LFS_DEFINE_PROFILE(4k, 11520, 2160)
LFS_DEFINE_PROFILE(8k, 23040, 4320)

#undef LFS_DEFINE_PROFILE

#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
#ifdef LFS_ENCODED_PLAIN_CURRENT
#define LFS_REGISTERED_INIT_PARAMS(p) plain_parameters(p)
#else
#define LFS_REGISTERED_INIT_PARAMS(p) measurement_parameters(p)
#endif
#ifdef LFS_ENCODED_PLAIN_CURRENT
union alignas(8) EsSignalTile {
    unsigned char storage[38*22*sizeof(Half4)];
    float luma[38*22];
    __device__ Half4 *colors() {return reinterpret_cast<Half4*>(storage);}
};
#define LFS_REGULAR_ES1(Name,Width,Height) \
extern "C" __global__ void lfs_regular_deterministic_es1_##Name(EncodedAtrousNativeParams p){ \
 __shared__ EsSignalTile raw,filtered; __shared__ SharedGuide guides[38*22]; LFS_SHARED_NORMALS(normals,38*22) \
 if(!valid_params<Width,Height>(p)){if(threadIdx.x==0u&&threadIdx.y==0u)fail(p,kFrameStatusInvalidLayout);return;} \
 dispatch_init_step1<Width,Height,true,true>(LFS_REGISTERED_INIT_PARAMS(p),raw.colors(),filtered.colors(),guides,normals,filtered.luma,raw.luma); }
#else
#define LFS_REGULAR_ES1(Name,Width,Height) \
extern "C" __global__ void lfs_regular_deterministic_es1_##Name(EncodedAtrousNativeParams p){ \
 __shared__ Half4 raw[38*22],filtered[38*22];__shared__ SharedGuide guides[38*22];LFS_SHARED_NORMALS(normals,38*22) \
 if(!valid_params<Width,Height>(p)){if(threadIdx.x==0u&&threadIdx.y==0u)fail(p,kFrameStatusInvalidLayout);return;} \
 dispatch_init_step1<Width,Height,true>(LFS_REGISTERED_INIT_PARAMS(p),raw,filtered,guides,normals); }
#endif
LFS_REGULAR_ES1(2k,4608,2048)
LFS_REGULAR_ES1(4k,11520,2160)
LFS_REGULAR_ES1(8k,23040,4320)
#undef LFS_REGULAR_ES1
#define LFS_HELD_TYPED(Name,Width,Height) \
extern "C" __global__ void lfs_capture_held_es1_##Name(EncodedAtrousNativeParams p){ \
 __shared__ Half4 raw[38*22],filtered[38*22];__shared__ SharedGuide guides[38*22];LFS_SHARED_NORMALS(normals,38*22) \
 if(!valid_params<Width,Height>(p)||!p.hold_state){if(threadIdx.x==0u&&threadIdx.y==0u)fail(p,kFrameStatusInvalidLayout);return;} \
 dispatch_init_step1<Width,Height,false,false,true>(LFS_REGISTERED_INIT_PARAMS(p),raw,filtered,guides,normals); }
LFS_HELD_TYPED(2k,4608,2048)
LFS_HELD_TYPED(4k,11520,2160)
#undef LFS_HELD_TYPED
#ifdef LFS_ENCODED_PLAIN_CURRENT
#define LFS_HELD_ENV(Name,Width,Height) \
extern "C" __global__ void lfs_capture_held_environment_es1_##Name(EncodedAtrousNativeParams p){ \
 __shared__ EsSignalTile raw,filtered;__shared__ SharedGuide guides[38*22];LFS_SHARED_NORMALS(normals,38*22) \
 if(!valid_params<Width,Height>(p)||!p.hold_state){if(threadIdx.x==0u&&threadIdx.y==0u)fail(p,kFrameStatusInvalidLayout);return;} \
 dispatch_init_step1<Width,Height,true,true,true>(LFS_REGISTERED_INIT_PARAMS(p),raw.colors(),filtered.colors(),guides,normals,filtered.luma,raw.luma); }
#else
#define LFS_HELD_ENV(Name,Width,Height) \
extern "C" __global__ void lfs_capture_held_environment_es1_##Name(EncodedAtrousNativeParams p){ \
 __shared__ Half4 raw[38*22],filtered[38*22];__shared__ SharedGuide guides[38*22];LFS_SHARED_NORMALS(normals,38*22) \
 if(!valid_params<Width,Height>(p)||!p.hold_state){if(threadIdx.x==0u&&threadIdx.y==0u)fail(p,kFrameStatusInvalidLayout);return;} \
 dispatch_init_step1<Width,Height,true,false,true>(LFS_REGISTERED_INIT_PARAMS(p),raw,filtered,guides,normals); }
#endif
LFS_HELD_ENV(2k,4608,2048)
LFS_HELD_ENV(4k,11520,2160)
#undef LFS_HELD_ENV
#undef LFS_REGISTERED_INIT_PARAMS
#endif
#ifdef LFS_ENCODED_PLAIN_CURRENT
#ifdef LFS_ENCODED_PLAIN_CURRENT
extern "C" __global__ void lfs_present_panel_samples(EncodedAtrousNativeParams p) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p.count || *reinterpret_cast<const unsigned*>(p.frame_status))return;
    auto *compact=reinterpret_cast<unsigned*>(p.panel_samples);
    const unsigned packet=compact[i],encoded=packet>>16u;
    if(encoded==0x7e00u)return;
    const unsigned slot=4u*(i/3u)+i%3u;
    auto *panel=reinterpret_cast<float*>(p.canonical);
    const float now=panel[slot];
    if(p.hold_same || !(packet&0x8000u) || !(encoded&0x7fffu)) {
        compact[i]=unsigned(to_half(now))|0x8000u;
        return;
    }
    panel[slot]=combine_previous_footprint(p,i,now);
}
#endif
#define LFS_REGULAR_R2(Name,Width,Height) \
extern "C" __global__ void lfs_regular_carrier_r2_##Name(EncodedAtrousNativeParams p){ \
 __shared__ float r2_luma[(Width==4608 || Width==11520)?36*12:1];__shared__ Half4 tile[36*12];__shared__ SharedGuide guides[36*12];LFS_SHARED_NORMALS(normals,36*12) \
 if(!valid_params<Width,Height>(p)||!plain_current_compatible(p)){if(threadIdx.x==0u&&threadIdx.y==0u)fail(p,kFrameStatusInvalidLayout);return;} \
 dispatch_atrous<Width,Height,2,32,8,2,true,false,(Width==4608 || Width==11520)>(plain_parameters(p),reinterpret_cast<const Half4*>(p.scratch_b),reinterpret_cast<Half4*>(p.scratch_a),tile,guides,normals,r2_luma); }
LFS_REGULAR_R2(2k,4608,2048)
LFS_REGULAR_R2(4k,11520,2160)
LFS_REGULAR_R2(8k,23040,4320)
#undef LFS_REGULAR_R2
#endif

#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
#define LFS_REGULAR_ROUTED_R2(Name,Width,Height) \
extern "C" __global__ void lfs_regular_routed_r2_##Name(EncodedAtrousNativeParams p){ \
 __shared__ float r2_luma[(Width==4608 || Width==11520)?36*12:1];__shared__ Half4 tile[36*12];__shared__ SharedGuide guides[36*12];LFS_SHARED_NORMALS(normals,36*12) \
 if(!valid_params<Width,Height>(p)){if(threadIdx.x==0u&&threadIdx.y==0u)fail(p,kFrameStatusInvalidLayout);return;} \
 dispatch_atrous<Width,Height,2,32,8,2,false,true,(Width==4608 || Width==11520)>(measurement_parameters(p),reinterpret_cast<const Half4*>(p.scratch_b),reinterpret_cast<Half4*>(p.scratch_a),tile,guides,normals,r2_luma); }
LFS_REGULAR_ROUTED_R2(2k,4608,2048)
LFS_REGULAR_ROUTED_R2(4k,11520,2160)
LFS_REGULAR_ROUTED_R2(8k,23040,4320)
#undef LFS_REGULAR_ROUTED_R2
#endif


#ifdef LFS_ENCODED_PLAIN_CURRENT
namespace {
#include "refinement_guide.cuh"
#include "view_chart_refinement.cuh"
}
#include "view_chart_cached.cuh"
#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
#include "ssat/phase2/native/refinement_tiled.cuh"
#include "ssat/phase2/native/view_chart_final.cuh"
#endif


#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
#include "stationary_view_reconstruction.cuh"
#include "registered_view_reconstruction.cuh"
#include "resident_view_reconstruction.cuh"
#include "fixed_motion_transition.cuh"
#include "held_es_reconstruction.cuh"
#ifdef LFS_ENCODED_PLAIN_CURRENT
#include "stationary_filter_weights.cuh"
#else
#include "stationary_aperture_weights.cuh"
#include "stationary_cell_weights.cuh"
#include "held_view_cells.cuh"
#endif
#endif
} // namespace lfs::denoise::encoded_atrous::native
