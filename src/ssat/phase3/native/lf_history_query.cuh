#pragma once
#include "physical_lf_taa_native_params.h"
#include "physical_lf_taa_native_current_adapter.h"
#include "ssat/native/denoise_native_types.h"
#include "ssat/native/canonical_transfer.cuh"
#include "ssat/native/packed_surface_observation_contract.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstdint>

namespace native = lfs::denoise::physical_lf::native;
namespace native_current = lfs::denoise::physical_lf::native_current;
namespace denoise_native = lfs::denoise::native;
namespace ssat_native = lfs::ssat::native;

namespace {

constexpr std::uint32_t kSurfaceValidBit = 0u;
constexpr std::uint32_t kDomainBit = 1u;
constexpr std::uint32_t kClassShift = 2u;
constexpr std::uint32_t kClassMask = 0x1fu;
constexpr std::uint32_t kAgeShift = 7u;
constexpr std::uint32_t kConfidenceShift = 15u;
constexpr std::uint32_t kResponseDirectionHistoryBit = 23u;
constexpr std::uint32_t kResponseMedianHistoryBit = 24u;
constexpr std::uint32_t kResponseLogHistoryBit = 25u;
constexpr std::uint32_t kResponseBranchHistoryBit = 26u;
constexpr std::uint32_t kResponseBranchValidBit = 27u;
constexpr std::uint32_t kResponseBranchTransmissionBit = 28u;
constexpr std::uint32_t kComponentStableConfidenceShift = 23u;
constexpr std::uint32_t kComponentTransportedConfidenceShift = 27u;
constexpr std::uint32_t kRigidMotionValid = 1u << 0u;
constexpr std::uint32_t kRigidMotionIdentity = 1u << 1u;
constexpr std::uint32_t kSurfaceDiffuse = 1u << 0u;
constexpr std::uint32_t kSurfaceGlossy = 1u << 1u;
constexpr std::uint32_t kSurfaceNearSpecular = 1u << 2u;
constexpr std::uint32_t kSurfaceTransmission = 1u << 3u;
constexpr std::uint32_t kSurfaceEmissive = 1u << 4u;
constexpr std::uint32_t kResponsiveSurfaceMask =
    kSurfaceGlossy | kSurfaceNearSpecular |
    kSurfaceTransmission | kSurfaceEmissive;
constexpr int kCurrentTileWidth = 32;
constexpr int kCurrentTileHeight = 8;
// The dynamic Quality kernel uses 82 registers on sm_120. Two-row blocks
// reduce register-allocation fragmentation while retaining the full query.
constexpr int kQualityDynamicTileHeight = 2;
constexpr int kPerformanceCurrentTileHeight = 4;
constexpr int kCurrentHaloX = 3;
constexpr int kCurrentHaloY = 1;
constexpr int kCurrentSharedWidth =
    kCurrentTileWidth + 2 * kCurrentHaloX;
template<int TileHeight>
constexpr int kCurrentSharedCount = kCurrentSharedWidth *
    (TileHeight + 2 * kCurrentHaloY);
constexpr std::uint32_t kTimestampStarted = 1u << 0u;
constexpr std::uint32_t kTimestampEnded = 1u << 1u;
constexpr std::uint32_t kSubstageCompactPackEnded = 1u << 0u;
constexpr std::uint32_t kSubstageTemporalResolveEnded = 1u << 1u;

template<typename T>
__device__ __forceinline__ T *pointer(std::uint64_t address) {
    return reinterpret_cast<T *>(static_cast<std::uintptr_t>(address));
}

__device__ __forceinline__ float half_to_float(std::uint16_t value) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
    return result;
}

__device__ __forceinline__ std::uint16_t float_to_half(float value) {
    std::uint16_t result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ bool half_is_finite(std::uint16_t value) {
    return (value & 0x7c00u) != 0x7c00u;
}

__device__ __forceinline__ bool finitef(float value) {
    return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ void mark_frame_status(
    const native::Phase3Params &params,
    std::uint32_t status) {
    if (params.frame_status != 0u) {
        atomicOr(pointer<std::uint32_t>(params.frame_status), status);
    }
}

__device__ __forceinline__ float sqrt_rn(float value) {
    float result;
    asm("sqrt.rn.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float calibrated_tangent(float angle) {
    // Phase3 calibration constrains angle to (-pi/2, pi/2). Reducing by eight
    // keeps the polynomial in a narrow interval. Carry the double-angle
    // recurrence as n/d, delaying division until all three steps are complete:
    // tan(2a) = 2*n*d / (d*d - n*n). This preserves the calibrated function
    // while removing two dependent divisions; it does not change ViewId.
    const float x = angle * 0.125f;
    const float x2 = x * x;
    float tangent = x * fmaf(
        x2,
        fmaf(
            x2,
            fmaf(x2, fmaf(x2, 62.0f / 2835.0f, 17.0f / 315.0f),
                 2.0f / 15.0f),
            1.0f / 3.0f),
        1.0f);
    float denominator = 1.0f;
    #pragma unroll
    for (int iteration = 0; iteration < 3; ++iteration) {
        const float numerator = (2.0f * tangent) * denominator;
        denominator = denominator * denominator - tangent * tangent;
        tangent = numerator;
    }
    return tangent / denominator;
}

__device__ __forceinline__ float clampf(float value, float low, float high) {
    return fminf(fmaxf(value, low), high);
}

__device__ __forceinline__ float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 subtract(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 multiply(float3 value, float scalar) {
    return make_float3(value.x * scalar, value.y * scalar, value.z * scalar);
}

__device__ __forceinline__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 normalize3(float3 value) {
    return multiply(value, rsqrtf(fmaxf(dot3(value, value), 1.0e-20f)));
}

__device__ __forceinline__ float3 xyz(native::Float4 value) {
    return make_float3(value.x, value.y, value.z);
}

__device__ __forceinline__ float3 transform_vector(
    const native::Matrix4x4 &matrix,
    float3 value) {
    return make_float3(
        matrix.columns[0].x * value.x + matrix.columns[1].x * value.y +
            matrix.columns[2].x * value.z,
        matrix.columns[0].y * value.x + matrix.columns[1].y * value.y +
            matrix.columns[2].y * value.z,
        matrix.columns[0].z * value.x + matrix.columns[1].z * value.y +
            matrix.columns[2].z * value.z);
}

__device__ __forceinline__ float3 transform_point(
    const native::Matrix4x4 &matrix,
    float3 value) {
    return add(transform_vector(matrix, value), xyz(matrix.columns[3]));
}

__device__ __forceinline__ float3 transform_motion_vector(
    const native::RigidInstanceMotionRecord &record,
    float3 value) {
    const float3 r0 = xyz(record.current_world_to_previous_world_row0);
    const float3 r1 = xyz(record.current_world_to_previous_world_row1);
    const float3 r2 = xyz(record.current_world_to_previous_world_row2);
    return make_float3(dot3(r0, value), dot3(r1, value), dot3(r2, value));
}

__device__ __forceinline__ float3 transform_motion_point(
    const native::RigidInstanceMotionRecord &record,
    float3 value) {
    const float3 transformed = transform_motion_vector(record, value);
    return make_float3(
        transformed.x + record.current_world_to_previous_world_row0.w,
        transformed.y + record.current_world_to_previous_world_row1.w,
        transformed.z + record.current_world_to_previous_world_row2.w);
}

__device__ __forceinline__ std::uint32_t surface_valid(
    std::uint32_t metadata) {
    return (metadata >> kSurfaceValidBit) & 1u;
}

__device__ __forceinline__ std::uint32_t signal_domain(
    std::uint32_t metadata) {
    return (metadata >> kDomainBit) & 1u;
}

__device__ __forceinline__ std::uint32_t surface_class(
    std::uint32_t metadata) {
    return (metadata >> kClassShift) & kClassMask;
}

__device__ __forceinline__ std::uint32_t history_age(
    std::uint32_t metadata) {
    return (metadata >> kAgeShift) & 0xffu;
}

__device__ __forceinline__ std::uint32_t pack_metadata(
    std::uint32_t valid,
    std::uint32_t domain,
    std::uint32_t classification,
    std::uint32_t age,
    std::uint32_t confidence) {
    return ((valid & 1u) << kSurfaceValidBit) |
        ((domain & 1u) << kDomainBit) |
        ((classification & kClassMask) << kClassShift) |
        ((age & 0xffu) << kAgeShift) |
        ((confidence & 0xffu) << kConfidenceShift);
}

__device__ __forceinline__ std::uint32_t pack_normal(float3 normal) {
    const float denominator = fmaxf(
        fabsf(normal.x) + fabsf(normal.y) + fabsf(normal.z), 1.0e-20f);
    const float projected_x = normal.x / denominator;
    const float projected_y = normal.y / denominator;
    const float sign_x = projected_x >= 0.0f ? 1.0f : -1.0f;
    const float sign_y = projected_y >= 0.0f ? 1.0f : -1.0f;
    const float encoded_x = normal.z >= 0.0f
        ? projected_x
        : (1.0f - fabsf(projected_y)) * sign_x;
    const float encoded_y = normal.z >= 0.0f
        ? projected_y
        : (1.0f - fabsf(projected_x)) * sign_y;
    const std::uint32_t x = static_cast<std::uint32_t>(clampf(
        roundf((encoded_x * 0.5f + 0.5f) * 65535.0f), 0.0f, 65535.0f));
    const std::uint32_t y = static_cast<std::uint32_t>(clampf(
        roundf((encoded_y * 0.5f + 0.5f) * 65535.0f), 0.0f, 65535.0f));
    return x | (y << 16u);
}

__device__ __forceinline__ native::PackedDenoiserPathGuide
load_denoiser_path_guide(
    const native::Phase3Params &params,
    std::uint32_t index) {
    if (params.current_denoiser_path_guide == 0u) return {};
    return pointer<const native::PackedDenoiserPathGuide>(
        params.current_denoiser_path_guide)[index];
}

__device__ __forceinline__ std::uint32_t path_guide_class(
    const native::PackedDenoiserPathGuide &guide) {
    return (guide.metadata >> 20u) & 0x3u;
}

__device__ __forceinline__ bool path_guide_valid(
    const native::PackedDenoiserPathGuide &guide) {
    return ((guide.metadata >> 23u) & 1u) != 0u;
}

__device__ __forceinline__ bool responsive_path_guide(
    const native::PackedDenoiserPathGuide &guide) {
    return path_guide_valid(guide) && path_guide_class(guide) < 2u;
}

__device__ __forceinline__ bool diffuse_risk_path_guide(
    const native::PackedDenoiserPathGuide &guide) {
    return path_guide_valid(guide) && path_guide_class(guide) == 2u;
}

__device__ __forceinline__ float3 unpack_oct8(std::uint32_t packed) {
    float x = static_cast<float>(packed & 0xffu) *
        (2.0f / 255.0f) - 1.0f;
    float y = static_cast<float>((packed >> 8u) & 0xffu) *
        (2.0f / 255.0f) - 1.0f;
    const float z = 1.0f - fabsf(x) - fabsf(y);
    const float fold = clampf(-z, 0.0f, 1.0f);
    x += x >= 0.0f ? -fold : fold;
    y += y >= 0.0f ? -fold : fold;
    return normalize3(make_float3(x, y, z));
}

__device__ __forceinline__ float path_guide_hit_log(
    const native::PackedDenoiserPathGuide &guide) {
    return half_to_float(static_cast<std::uint16_t>(guide.payload1 >> 16u));
}

__device__ __forceinline__ float path_guide_hit_t(
    const native::PackedDenoiserPathGuide &guide) {
    return exp2f(path_guide_hit_log(guide)) - 1.0f;
}

__device__ __forceinline__ float path_guide_eta(
    const native::PackedDenoiserPathGuide &guide) {
    const float eta_log =
        static_cast<float>((guide.metadata >> 13u) & 0x7fu) *
            (4.0f / 127.0f) - 2.0f;
    return exp2f(eta_log);
}

__device__ __forceinline__ float packed_response_eta(
    std::uint32_t auxiliary) {
    const std::uint32_t eta_code = auxiliary & 0x1fu;
    if (eta_code == 0u) return 0.0f;
    const float eta_log = static_cast<float>(eta_code - 1u) *
        (4.0f / 30.0f) - 2.0f;
    return exp2f(eta_log);
}

__device__ __forceinline__ bool deterministic_response_class_allowed(
    const native::Phase3Params &params,
    std::uint32_t classification) {
    if ((classification & kSurfaceDiffuse) != 0u ||
        (classification &
         (kSurfaceGlossy | kSurfaceNearSpecular | kSurfaceTransmission)) ==
            0u) {
        return false;
    }
    if (params.response_direction_glossy_only == 0u) return true;
    return (classification & kSurfaceGlossy) != 0u &&
        (classification &
         (kSurfaceNearSpecular | kSurfaceTransmission)) == 0u;
}

__device__ __forceinline__ native::CurrentTemporalInput compact_current(
    const native::Phase3Params &params,
    std::uint32_t index,
    std::uint32_t channel, const native::Half4 &value) {
    const std::uint32_t scalar_index = index * 4u + channel;
    float physical{};
    float compact_demodulated{};
    {
        const std::uint16_t encoded = channel == 0u
            ? value.x : (channel == 1u ? value.y : value.z);
        compact_demodulated = half_to_float(encoded);
        float compact_variance = fabsf(half_to_float(value.w));

        if (!half_is_finite(value.x) || !half_is_finite(value.y) ||
            !half_is_finite(value.z) || !half_is_finite(value.w) ||
            !finitef(compact_demodulated) ||
            !finitef(compact_variance) || compact_demodulated < 0.0f ||
            compact_variance < 0.0f) {
            mark_frame_status(
                params, native::kPhase3FrameStatusInvalidReconstruction);
        }
    }
    std::uint32_t validity{};
    std::uint32_t flags{};
    float albedo{};
    float depth{};
    float diffuse_factor{};
    native::Float4 normal{};
    std::uint64_t surface_key = ~std::uint64_t{0u};
    std::uint32_t packed_normal{};
    bool guide_nonfinite{};
    {
        const denoise_native::PackedSurfaceObservationV1 observation =
            pointer<const denoise_native::PackedSurfaceObservationV1>(
                params.current_packed_observation)[index];
        const auto contract =
            ssat_native::validate_packed_surface_observation_v1(observation);
        if (contract.status != 0u) {
            mark_frame_status(
                params, native::kPhase3FrameStatusInvalidReconstruction);
        }
        validity = contract.domain ==
            ssat_native::kPackedSurfaceDomainSurface ? 1u : 0u;
        flags = contract.surface_flags & kClassMask;
        const auto packed_albedo =
            observation.linear_albedo_rgb_diffuse_factor;
        const std::uint16_t encoded_albedo = channel == 0u
            ? packed_albedo.x
            : (channel == 1u ? packed_albedo.y : packed_albedo.z);
        albedo = half_to_float(encoded_albedo);
        diffuse_factor = half_to_float(packed_albedo.w);
        depth = observation.hit_distance;
        surface_key = observation.surface_key;
        packed_normal = observation.packed_geometric_normal_oct16;
        guide_nonfinite = contract.status != 0u;
    }
    {
        physical = compact_demodulated *
            native_current::encoded_atrous_modulation(validity, albedo);
    }
    const bool input_nonfinite = guide_nonfinite || !finitef(physical) ||
        !finitef(albedo) || !finitef(diffuse_factor) ||
        (validity != 0u &&
         (!finitef(depth) ||
          (1u == 0u &&
           (!finitef(normal.x) || !finitef(normal.y) ||
            !finitef(normal.z)))));
    if (input_nonfinite) {
        mark_frame_status(params, native::kPhase3FrameStatusNonfinite);
    }
    const native::PackedDenoiserPathGuide denoiser_guide =
        load_denoiser_path_guide(params, index);

    const bool direct_only = (flags & kResponsiveSurfaceMask) != 0u;
    const bool diffuse = (flags & kSurfaceDiffuse) != 0u;
    const std::uint32_t domain =
        validity != 0u && diffuse && !direct_only &&
            diffuse_factor >= 0.5f &&
            albedo >= 0.05f
        ? 1u
        : 0u;

    native::CurrentTemporalInput current{};
    if (validity != 0u) {
        current.surface_key = surface_key;
        current.packed_normal = packed_normal;
    } else {
        current.surface_key = ~std::uint64_t{0u};
    }
    current.depth = depth;
    current.metadata = validity | (domain << kDomainBit) |
        ((flags & kClassMask) << kClassShift);
    current.signal_albedo = {
        float_to_half(clampf(physical, 0.0f, 500.0f)),
        float_to_half(clampf(albedo, 0.0f, 500.0f))};
    return current;
}

__device__ __forceinline__ float3 unpack_normal(std::uint32_t packed) {
    const float x = static_cast<float>(packed & 0xffffu) /
        65535.0f * 2.0f - 1.0f;
    const float y = static_cast<float>((packed >> 16u) & 0xffffu) /
        65535.0f * 2.0f - 1.0f;
    const float z = 1.0f - fabsf(x) - fabsf(y);
    const float fold = clampf(-z, 0.0f, 1.0f);
    return normalize3(make_float3(
        x + (x >= 0.0f ? -fold : fold),
        y + (y >= 0.0f ? -fold : fold),
        z));
}

__device__ __forceinline__ float view_position(
    const native::Phase3Params &params,
    std::uint32_t view_id) {
    if (params.extended_view_positions != 0u &&
        view_id < params.extended_view_count) {
        return pointer<const float>(
            params.extended_view_positions)[view_id];
    }
    return params.view_positions[view_id];
}

__device__ __forceinline__ float inverse_view_spacing(
    const native::Phase3Params &params,
    std::uint32_t view_id) {
    if (params.extended_inverse_view_spacings != 0u &&
        view_id < params.extended_view_count) {
        return pointer<const float>(
            params.extended_inverse_view_spacings)[view_id];
    }
    return params.inverse_view_spacings[view_id];
}

__device__ __forceinline__ std::uint32_t producer_view_id(
    const native::Phase3Params &params,
    std::uint32_t encoded_x,
    std::uint32_t encoded_y,
    float slant_tangent) {
    const float d = static_cast<float>(encoded_x) +
        3.0f * static_cast<float>(encoded_y) * slant_tangent +
        params.offset;
    const float branch = floorf(d * params.inverse_pitch);
    const float wrapped = d - branch * params.pitch;
    const std::uint32_t views = max(params.view_count, 1u);
    const std::uint32_t raw = min(
        static_cast<std::uint32_t>(floorf(
            wrapped * params.phase_view_scale)), views - 1u);
    return views - 1u - raw;
}

__device__ __forceinline__ std::uint32_t producer_view_id(
    const native::Phase3Params &params,
    std::uint32_t encoded_x,
    std::uint32_t encoded_y) {
    return producer_view_id(
        params, encoded_x, encoded_y, params.slant_tangent);
}

struct LocalRay {
    float3 origin;
    float3 direction;
    std::uint32_t view_id;
};

__device__ __forceinline__ LocalRay analytical_local_ray(
    const native::Phase3Params &params,
    std::uint32_t encoded_x,
    std::uint32_t encoded_y,
    std::uint32_t view_id) {
    const float lattice_width = params.focal_lattice_width;
    const float lattice_height = params.focal_lattice_height;
    const float3 focal = make_float3(
        -params.focal_width * 0.5f +
            (static_cast<float>(encoded_x) + 0.5f) * lattice_width,
        params.focal_height * 0.5f -
            (static_cast<float>(encoded_y) + 0.5f) * lattice_height,
        params.focal_distance);
    const float3 origin = make_float3(view_position(params, view_id), 0.0f, 0.0f);
    return {origin, normalize3(subtract(focal, origin)), view_id};
}

__device__ __forceinline__ LocalRay analytical_local_ray(
    const native::Phase3Params &params,
    std::uint32_t encoded_x,
    std::uint32_t encoded_y) {
    return analytical_local_ray(
        params, encoded_x, encoded_y,
        producer_view_id(params, encoded_x, encoded_y));
}

struct AngularNeighborhoodFilter {
    float3 center_direction{};
    float inverse_two_alpha_squared{};
    float footprint_current_weight{};
    bool enabled{};
};

__device__ __forceinline__ const native::RigidInstanceMotionRecord *
rigid_motion_record(
    const native::Phase3Params &params,
    std::uint32_t instance_id) {
    if (params.rigid_instance_motion == 0u ||
        params.rigid_instance_stride_bytes !=
            sizeof(native::RigidInstanceMotionRecord) ||
        instance_id >= params.rigid_instance_count) {
        return nullptr;
    }
    const auto *bytes = pointer<const std::uint8_t>(
        params.rigid_instance_motion);
    return reinterpret_cast<const native::RigidInstanceMotionRecord *>(
        bytes + static_cast<std::size_t>(instance_id) *
            params.rigid_instance_stride_bytes);
}

struct StableResponseHit {
    float hit_t{};
    std::uint32_t secondary_instance{};
    bool valid{};
};

template<int Radius, bool StaticEntry>
struct QueryState {
    std::uint32_t index{};
    std::uint32_t encoded_x{};
    std::uint32_t encoded_y{};
    std::uint32_t channel{};
    std::uint32_t current_view{};
    std::uint32_t current_valid{};
    std::uint32_t current_domain{};
    std::uint32_t current_class{};
    native::PackedDenoiserPathGuide path_guide{};
    float current_roughness{1.0f};
    float responsive_hit_log{};
    float responsive_previous_hit_log{};
    float responsive_direction_eta{};
    float contribution_dominance{};
    bool responsive_guide{};
    bool responsive_path{};
    bool deterministic_response_path{};
    bool response_vmb_attempted{};
    bool response_vmb_solved{};
    bool response_stable_hit_solved{};
    bool response_environment{};
    bool response_direction_valid{};
    bool diffuse_risk_path{};
    native::CurrentTemporalInput current{};
    float spatial_prior_physical{};
    float spatial_prior_signal{};
    float current_physical{};
    float current_albedo{};
    float current_signal{};
    float current_estimator_variance{};
    float current_effective_count{1.0f};
    float current_phase2_confidence{1.0f};
    float query_lattice_x{};
    float query_lattice_y{};
    float query_u_x{};
    float continuous_view{};
    float inverse_u_spacing{1.0f};
    float projection_residual_squared{};
    float jacobian_condition{};
    std::uint32_t generated_taps{};
    float3 query_direction_previous_local{};
    float3 response_direction_current_world{};
    float3 response_direction_previous_world{};
    float3 predicted_previous_position_local{};
    float3 predicted_previous_normal_local{};
    float lf_displacement{};
    bool query_valid{};
    bool exact_identity_query{};
    bool projection_invalid{};
    bool support_empty{};
};

struct AngularCoordinate {
    float continuous_view{};
    float inverse_spacing{1.0f};
};

template<bool Calibrated=false>
__device__ __forceinline__ AngularCoordinate angular_coordinate(
    const native::Phase3Params &params,
    float query_u) {
    AngularCoordinate result{};
    const int count = static_cast<int>(max(params.view_count, 1u));
    if (count <= 1) return result;
    if constexpr(Calibrated) {
        if(params.array_angle_radians>0.f && params.focal_distance>0.f && finitef(query_u)) {
            const float first=view_position(params,0u),last=view_position(params,unsigned(count-1));
            if(query_u<=first) {result.inverse_spacing=inverse_view_spacing(params,0u);return result;}
            if(query_u>=last) {result.continuous_view=float(count-1);result.inverse_spacing=inverse_view_spacing(params,unsigned(count-1));return result;}
            const float tangent=query_u*params.inverse_focal_distance;
            const int anchor=max(0,min(count-1,__float2int_rn(fmaf(tangent,params.inverse_array_angle_radians,.5f)*float(count-1))));
            const float a=view_position(params,unsigned(anchor))*params.inverse_focal_distance;
            const float delta=(tangent-a)/fmaf(tangent,a,1.f);
            if(fabsf(delta)<=.125f) {
                // atan(t)-atan(a)=atan((t-a)/(1+at)). The alternating
                // series remainder is <= |delta|^11/11 on this interval.
                const float x=delta*delta;
                const float rad=delta*fmaf(x,fmaf(x,fmaf(x,fmaf(x,1.f/9.f,-1.f/7.f),1.f/5.f),-1.f/3.f),1.f);
                result.continuous_view=clampf(fmaf(rad,float(count-1)*params.inverse_array_angle_radians,float(anchor)),0.f,float(count-1));
                result.inverse_spacing=inverse_view_spacing(params,unsigned(floorf(result.continuous_view)));
                return result;
            }
        }
    }
    const float unit = params.inverse_array_angle_radians != 0.0f
        ? clampf(atan2f(query_u, params.focal_distance) *
                params.inverse_array_angle_radians + 0.5f, 0.0f, 1.0f)
        : 0.5f;
    result.continuous_view = unit * static_cast<float>(count - 1);
    const int lower = max(
        0, min(static_cast<int>(floorf(result.continuous_view)), count - 1));
    result.inverse_spacing = inverse_view_spacing(
        params, static_cast<std::uint32_t>(lower));
    return result;
}

__device__ __forceinline__ bool solve_planar_snell_point(
    float3 aperture,
    float3 primary_point,
    float3 primary_normal,
    float3 secondary_point,
    float eta,
    float3 &interface_point) {
    const float3 normal = normalize3(primary_normal);
    const float aperture_signed = dot3(
        subtract(aperture, primary_point), normal);
    const float secondary_signed = dot3(
        subtract(secondary_point, primary_point), normal);
    if (!finitef(aperture_signed) || !finitef(secondary_signed) ||
        !finitef(eta) || eta <= 0.0f ||
        fabsf(aperture_signed) <= 1.0e-6f ||
        fabsf(secondary_signed) <= 1.0e-6f ||
        aperture_signed * secondary_signed >= 0.0f) {
        return false;
    }
    const float3 aperture_plane = subtract(
        aperture, multiply(normal, aperture_signed));
    const float3 secondary_plane = subtract(
        secondary_point, multiply(normal, secondary_signed));
    const float3 lateral = subtract(secondary_plane, aperture_plane);
    const float lateral_length = sqrt_rn(fmaxf(dot3(lateral, lateral), 0.0f));
    if (!finitef(lateral_length)) return false;
    if (lateral_length <= 1.0e-6f) {
        interface_point = aperture_plane;
        return true;
    }
    const float a = fabsf(aperture_signed);
    const float b = fabsf(secondary_signed);
    float lower = 0.0f;
    float upper = 1.0f;
    float t = clampf(eta * a / fmaxf(b + eta * a, 1.0e-20f),
                     lower, upper);
    #pragma unroll
    for (int iteration = 0; iteration < 2; ++iteration) {
        const float left = t * lateral_length;
        const float right = (1.0f - t) * lateral_length;
        const float left_length = sqrt_rn(a * a + left * left);
        const float right_length = sqrt_rn(b * b + right * right);
        const float sin_incident = left / fmaxf(left_length, 1.0e-20f);
        const float sin_transmitted = right /
            fmaxf(right_length, 1.0e-20f);
        const float residual = sin_incident - eta * sin_transmitted;
        if (residual > 0.0f) upper = t;
        else lower = t;
        const float derivative = lateral_length *
            (a * a /
                 fmaxf(left_length * left_length * left_length, 1.0e-20f) +
             eta * b * b /
                 fmaxf(right_length * right_length * right_length, 1.0e-20f));
        const float candidate = derivative > 1.0e-8f
            ? t - residual / derivative
            : 0.5f * (lower + upper);
        t = candidate > lower && candidate < upper && finitef(candidate)
            ? candidate
            : 0.5f * (lower + upper);
    }
    t = clampf(t, lower, upper);
    interface_point = add(aperture_plane, multiply(lateral, t));
    return finitef(interface_point.x) && finitef(interface_point.y) &&
        finitef(interface_point.z);
}

__device__ __forceinline__ bool intersect_forward_plane(
    float3 ray_origin,
    float3 ray_direction,
    float3 plane_point,
    float3 plane_normal,
    float3 &intersection) {
    const float denominator = dot3(ray_direction, plane_normal);
    if (!finitef(denominator) || fabsf(denominator) <= 1.0e-6f) {
        return false;
    }
    const float distance = dot3(
        subtract(plane_point, ray_origin), plane_normal) / denominator;
    if (!finitef(distance) || distance < 0.0f) return false;
    intersection = add(ray_origin, multiply(ray_direction, distance));
    return finitef(intersection.x) && finitef(intersection.y) &&
        finitef(intersection.z);
}

__device__ __forceinline__ float3 reflect_direction(
    float3 incident, float3 normal) {
    return normalize3(subtract(
        incident, multiply(normal, 2.0f * dot3(incident, normal))));
}

// eta is n_transmitted / n_incident, matching BSDFSample::eta.
__device__ __forceinline__ bool central_transmission_direction(
    float3 incident, float3 normal, float eta, float3 &transmitted) {
    incident = normalize3(incident);
    normal = normalize3(normal);
    if (!finitef(eta) || eta < 0.25f || eta > 4.0f) return false;
    if (dot3(incident, normal) > 0.0f) normal = multiply(normal, -1.0f);
    const float incident_normal = dot3(incident, normal);
    const float3 incident_tangent = subtract(
        incident, multiply(normal, incident_normal));
    const float3 transmitted_tangent = multiply(
        incident_tangent, 1.0f / eta);
    const float tangent_squared = dot3(
        transmitted_tangent, transmitted_tangent);
    if (!finitef(tangent_squared) || tangent_squared >= 1.0f) return false;
    transmitted = normalize3(add(
        transmitted_tangent,
        multiply(normal, -sqrt_rn(fmaxf(1.0f - tangent_squared, 0.0f)))));
    return finitef(transmitted.x) && finitef(transmitted.y) &&
        finitef(transmitted.z);
}

__device__ __forceinline__ bool inverse_central_transmission_direction(
    float3 transmitted, float3 normal, float eta, float3 &incident) {
    transmitted = normalize3(transmitted);
    normal = normalize3(normal);
    if (!finitef(eta) || eta < 0.25f || eta > 4.0f) return false;
    const float transmitted_normal = dot3(transmitted, normal);
    const float3 transmitted_tangent = subtract(
        transmitted, multiply(normal, transmitted_normal));
    const float3 incident_tangent = multiply(transmitted_tangent, eta);
    const float tangent_squared = dot3(incident_tangent, incident_tangent);
    if (!finitef(tangent_squared) || tangent_squared >= 1.0f) return false;
    const float normal_sign = transmitted_normal >= 0.0f ? 1.0f : -1.0f;
    incident = normalize3(add(
        incident_tangent,
        multiply(normal, normal_sign *
            sqrt_rn(fmaxf(1.0f - tangent_squared, 0.0f)))));
    return finitef(incident.x) && finitef(incident.y) && finitef(incident.z);
}

__device__ __forceinline__ bool primary_incident_to_lf_ray(
    float3 primary_point, float3 desired_incident,
    float3 &origin, float3 &direction) {
    desired_incident = normalize3(desired_incident);
    if (!finitef(desired_incident.z) ||
        fabsf(desired_incident.z) <= 1.0e-6f) return false;
    const float distance = primary_point.z / desired_incident.z;
    if (!finitef(distance) || distance <= 0.0f) return false;
    const float3 aperture = subtract(
        primary_point, multiply(desired_incident, distance));
    origin = make_float3(aperture.x, 0.0f, 0.0f);
    direction = normalize3(subtract(primary_point, origin));
    return finitef(origin.x) && finitef(direction.x) &&
        finitef(direction.y) && finitef(direction.z);
}

__device__ __forceinline__ bool central_reflection_content(
    const native::PackedDenoiserPathGuide &guide, float3 incident, float3 primary_normal,
    float3 &central, float &distance) {
    central=reflect_direction(normalize3(incident),normalize3(primary_normal));
    if((guide.metadata>>22u)&1u){distance=0.f;return true;}
    const float3 secondary_normal=unpack_oct8(guide.payload1);
    const float numerator=dot3(secondary_normal,unpack_normal(guide.payload0));
    const float denominator=dot3(secondary_normal,central);
    // Conservative oct8 direction bound: 2*sqrt(18)/255.
    constexpr float oct8_error=2.f*4.242640687119285f/255.f;
    if(!(fabsf(numerator)>oct8_error && fabsf(denominator)>oct8_error &&
         numerator*denominator>0.f))return false;
    distance=path_guide_hit_t(guide)*numerator/denominator;
    return finitef(distance)&&distance>0.f;
}

template<int Radius, bool StaticEntry, bool EncodedAddress, bool MeasurementGuide=false, bool GlossyMeasurementGuide=false,bool CalibratedAngles=false>
__device__ __forceinline__ QueryState<Radius, StaticEntry> make_query_state(
    const native::Phase3Params &params,
    std::uint32_t index,
    const native::CurrentTemporalInput &current,
    bool suppress_response_motion = false,
    int deterministic_response_branch_override = -1) {
    QueryState<Radius, StaticEntry> state{};
    const std::uint32_t sample_width = params.output_width * 3u;
    state.index = index;
    state.encoded_x = index % sample_width;
    state.encoded_y = index / sample_width;
    state.channel = state.encoded_x % 3u;
    state.current = current;
    state.current_valid = surface_valid(state.current.metadata);
    state.current_domain = signal_domain(state.current.metadata);
    state.current_class = surface_class(state.current.metadata);
    // The RGB-history dispatcher rejects any secondary-guide binding.
    // Specialize the read here, leaving the kernel parameter block read-only.
    if constexpr (MeasurementGuide) state.path_guide = {};
    else state.path_guide = load_denoiser_path_guide(params,index);
    state.responsive_guide = state.current_valid != 0u &&
        responsive_path_guide(state.path_guide) &&
        ((state.current_class &
         (kSurfaceNearSpecular | kSurfaceTransmission)) != 0u ||
         (GlossyMeasurementGuide && state.current_class==kSurfaceGlossy));
    state.responsive_path = !suppress_response_motion &&
        MeasurementGuide && path_guide_class(state.path_guide)==0u &&
        state.current_valid != 0u &&
        state.responsive_guide;
    state.responsive_hit_log = state.responsive_path &&
        ((state.path_guide.metadata >> 22u) & 1u) == 0u
        ? path_guide_hit_log(state.path_guide) : 0.0f;
    state.responsive_previous_hit_log = state.responsive_hit_log;
    float embedded_path_risk = 0.0f;
    {
        const denoise_native::PackedSurfaceObservationV1 observation =
            pointer<const denoise_native::PackedSurfaceObservationV1>(
                params.current_packed_observation)[index];
        state.current_roughness = static_cast<float>(
            (observation.metadata >>
             denoise_native::kPackedSurfaceRoughnessShift) & 0xffu) *
            (1.0f / 255.0f);
        if (params.current_path_risk_packed != 0u) {
            const std::uint32_t auxiliary =
                (observation.metadata >>
                 denoise_native::kPackedSurfacePathRiskShift) &
                denoise_native::kPackedSurfacePathRiskMask;
            embedded_path_risk = static_cast<float>(auxiliary) *
                (1.0f / 63.0f);
            if (0u != 0u &&
                auxiliary != 0u) {
                state.responsive_direction_eta =
                    packed_response_eta(auxiliary);
            }
        }
    }
    state.deterministic_response_path = !suppress_response_motion &&
        0u != 0u &&
        params.current_path_risk_packed != 0u &&
        state.current_valid != 0u &&
        deterministic_response_class_allowed(params, state.current_class) &&
        finitef(state.responsive_direction_eta) &&
        state.responsive_direction_eta >= 0.25f &&
        state.responsive_direction_eta <= 4.0f;
    const bool guide_risk = diffuse_risk_path_guide(state.path_guide);
    state.diffuse_risk_path =
        (guide_risk || params.current_path_risk_packed != 0u) &&
        (state.current_class & kSurfaceDiffuse) != 0u &&
        (state.current_class & kResponsiveSurfaceMask) == 0u;
    state.contribution_dominance = state.diffuse_risk_path
        ? (guide_risk
               ? static_cast<float>(state.path_guide.metadata & 0xffu) *
                     (1.0f / 255.0f)
               : embedded_path_risk)
        : 0.0f;
    state.spatial_prior_physical = half_to_float(state.current.signal_albedo.x);
    state.current_albedo = half_to_float(state.current.signal_albedo.y);
    state.spatial_prior_signal = state.current_domain != 0u
        ? state.spatial_prior_physical / fmaxf(state.current_albedo, 0.05f)
        : state.spatial_prior_physical;
    {
        // Spatial-current production paths never consume the raw carrier in
        // their posterior. Avoid the otherwise dead full-frame radiance read;
        state.current_physical = state.spatial_prior_physical;
        state.current_signal = state.spatial_prior_signal;
    }
    if (!finitef(state.spatial_prior_physical) ||
        !finitef(state.spatial_prior_signal) ||
        !finitef(state.current_physical) || !finitef(state.current_signal) ||
        state.spatial_prior_physical < 0.0f || state.current_physical < 0.0f) {
        mark_frame_status(params, native::kPhase3FrameStatusNonfinite);
    }
    {
        const native::Half4 phase2 =
            pointer<const native::Half4>(params.current_phase2)[index];
        float stored_variance = half_to_float(phase2.w);
        // Private M inputs use the variance sign as availability, not negative energy.
        if constexpr(MeasurementGuide) stored_variance=fabsf(stored_variance);

        const std::uint32_t scalar_index = index * 4u + state.channel;
        const float source_albedo = 1u != 0u
            ? state.current_albedo
            : pointer<const float>(params.current_albedo)[scalar_index];
        const float variance_scale_squared =
            native_current::encoded_atrous_variance_scale_squared(
                state.current_valid,
                state.current_domain != 0u,
                source_albedo);
        const float converted_variance =
            stored_variance * variance_scale_squared;
        if (!half_is_finite(phase2.w) || stored_variance < 0.0f ||
            !finitef(source_albedo) ||
            !finitef(variance_scale_squared) ||
            !finitef(converted_variance)) {
            mark_frame_status(
                params, native::kPhase3FrameStatusInvalidReconstruction);
        }
        state.current_estimator_variance = fmaxf(converted_variance, 0.0f);
        state.current_effective_count = 1.0f;
        state.current_phase2_confidence = 1.0f;
    }
    if (params.history_valid == 0u) return state;

    const LocalRay current_local = analytical_local_ray(
        params, state.encoded_x, state.encoded_y);
    state.current_view = current_local.view_id;

    // StaticEntry is selected only after the host proves that the calibrated
    // LF layout and camera transform are unchanged.  A surface additionally
    // has to prove identity rigid motion here.  Under those conditions the
    // exact physical reprojection is the current encoded identity, so avoid
    // reconstructing the same result through two world transforms and two
    // focal-plane intersections.  This is an exact-address fast path, not a
    // same-index fallback: any unproven surface fails closed and dynamic
    // frames use the general physical projection below.
    if constexpr (StaticEntry) {
        {
            if (params.static_layout == 0u) return state;
            if (state.current_valid != 0u) {
                const std::uint32_t instance_id = static_cast<std::uint32_t>(
                    state.current.surface_key >> 32u);
                const native::RigidInstanceMotionRecord *motion =
                    rigid_motion_record(params, instance_id);
                if (motion == nullptr ||
                    (motion->metadata.x & kRigidMotionValid) == 0u ||
                    (motion->metadata.x & kRigidMotionIdentity) == 0u ||
                    !finitef(state.current.depth) ||
                    state.current.depth < 0.0f) {
                    return state;
                }
                state.predicted_previous_position_local = add(
                    current_local.origin,
                    multiply(current_local.direction, state.current.depth));
            } else {
                state.predicted_previous_position_local = current_local.origin;
            }
            state.query_lattice_x = static_cast<float>(state.encoded_x);
            state.query_lattice_y = static_cast<float>(state.encoded_y);
            state.query_u_x = current_local.origin.x;
            state.continuous_view = static_cast<float>(current_local.view_id);
            state.inverse_u_spacing = inverse_view_spacing(
                params, min(current_local.view_id,
                            max(params.view_count, 1u) - 1u));
            state.query_direction_previous_local = current_local.direction;
            state.predicted_previous_normal_local = transform_vector(
                params.previous_world_to_local,
                unpack_normal(state.current.packed_normal));
            state.exact_identity_query = true;
            state.query_valid =
                finitef(state.query_u_x) &&
                finitef(state.query_direction_previous_local.x) &&
                finitef(state.query_direction_previous_local.y) &&
                finitef(state.query_direction_previous_local.z) &&
                finitef(state.predicted_previous_position_local.x) &&
                finitef(state.predicted_previous_position_local.y) &&
                finitef(state.predicted_previous_position_local.z) &&
                finitef(state.predicted_previous_normal_local.x) &&
                finitef(state.predicted_previous_normal_local.y) &&
                finitef(state.predicted_previous_normal_local.z);
            return state;
        }
    }

    const float3 current_world_origin = transform_point(
        params.current_local_to_world, current_local.origin);
    const float3 current_world_direction = transform_vector(
        params.current_local_to_world, current_local.direction);
    float3 previous_world_point = current_world_origin;
    float3 previous_world_direction = current_world_direction;
    const float3 current_world_primary_normal = normalize3(
        unpack_normal(state.current.packed_normal));
    float3 previous_world_normal = current_world_primary_normal;
    float3 current_world_primary_point = current_world_origin;
    bool identity_transport = state.current_valid == 0u;

    if (state.current_valid != 0u) {
        const std::uint32_t instance_id = static_cast<std::uint32_t>(
            state.current.surface_key >> 32u);
        const native::RigidInstanceMotionRecord *motion =
            rigid_motion_record(params, instance_id);
        if (motion == nullptr ||
            (motion->metadata.x & kRigidMotionValid) == 0u ||
            (StaticEntry &&
             (motion->metadata.x & kRigidMotionIdentity) == 0u) ||
            !finitef(state.current.depth) || state.current.depth < 0.0f) {
            return state;
        }
        identity_transport =
            (motion->metadata.x & kRigidMotionIdentity) != 0u;
        current_world_primary_point = add(
            current_world_origin,
            multiply(current_world_direction, state.current.depth));
        previous_world_point = transform_motion_point(
            *motion, current_world_primary_point);
        previous_world_direction = normalize3(transform_motion_vector(
            *motion, current_world_direction));
        previous_world_normal = transform_motion_vector(
            *motion, previous_world_normal);
        if (state.response_direction_valid) {
            state.response_direction_previous_world = normalize3(
                transform_motion_vector(
                    *motion, state.response_direction_current_world));
        }
    }

    float3 query_origin_previous_local = transform_point(
        params.previous_world_to_local, previous_world_point);
    // Surface validation follows the transported primary material point.
    // A responsive virtual ray may start at the previous aperture (and point
    // at a virtual/ refracted witness), so its ray origin must never replace
    // this geometric validation point.
    float3 validation_position_previous_local =
        query_origin_previous_local;
    float3 query_direction_previous_local = transform_vector(
        params.previous_world_to_local, previous_world_direction);
    if constexpr(MeasurementGuide) {
        // Same camera and rigid identity imply the identical physical LF target.
        if(params.static_layout!=0u && identity_transport)state.responsive_path=false;
    }
    if (state.deterministic_response_path) {
        state.response_vmb_attempted = true;
        const float3 previous_primary_local = transform_point(
            params.previous_world_to_local, previous_world_point);
        const float3 previous_primary_normal = normalize3(transform_vector(
            params.previous_world_to_local, previous_world_normal));
        const float3 incident_current_world = normalize3(
            current_world_direction);
        float3 desired_outgoing_world{};
        float3 desired_incident_previous_local{};
        bool response_query_valid = false;
        bool transmission_response =
            deterministic_response_branch_override >= 0
            ? deterministic_response_branch_override == 1
            : (state.current_class & kSurfaceTransmission) != 0u;

        if (transmission_response) {
            response_query_valid = central_transmission_direction(
                incident_current_world, current_world_primary_normal,
                state.responsive_direction_eta, desired_outgoing_world);
            if (response_query_valid) {
                const float3 desired_outgoing_previous_local = normalize3(
                    transform_vector(
                        params.previous_world_to_local,
                        desired_outgoing_world));
                response_query_valid =
                    inverse_central_transmission_direction(
                        desired_outgoing_previous_local,
                        previous_primary_normal,
                        state.responsive_direction_eta,
                        desired_incident_previous_local);
            }
        } else {
            desired_outgoing_world = reflect_direction(
                incident_current_world, current_world_primary_normal);
            const float3 desired_outgoing_previous_local = normalize3(
                transform_vector(
                    params.previous_world_to_local, desired_outgoing_world));
            desired_incident_previous_local = reflect_direction(
                desired_outgoing_previous_local, previous_primary_normal);
            response_query_valid =
                finitef(desired_incident_previous_local.x) &&
                finitef(desired_incident_previous_local.y) &&
                finitef(desired_incident_previous_local.z);
        }
        bool stable_hit_solved = false;

        if (!stable_hit_solved && response_query_valid) {
            response_query_valid = primary_incident_to_lf_ray(
                previous_primary_local,
                desired_incident_previous_local,
                query_origin_previous_local,
                query_direction_previous_local);
            if (response_query_valid) {
                validation_position_previous_local = previous_primary_local;
            }
        } else if (stable_hit_solved) {
            response_query_valid = true;
        }
        state.response_stable_hit_solved = stable_hit_solved;
        state.response_vmb_solved = response_query_valid;
        // Fail back to the existing physical primary-surface address.  This
        // is not same-index history: the ordinary dynamic SMB transform above
        // remains fully active whenever the central response inversion is not
        // representable by the one-dimensional LF aperture.
    } else if (state.responsive_path) {
        state.response_vmb_attempted = true;
        const std::uint32_t branch = path_guide_class(state.path_guide);
        const bool environment =
            ((state.path_guide.metadata >> 22u) & 1u) != 0u;
        state.response_environment = environment;
        float3 continuation_world = unpack_normal(
            state.path_guide.payload0);
        float content_hit_t = path_guide_hit_t(state.path_guide);
        if constexpr(GlossyMeasurementGuide) {
            if(state.current_class==kSurfaceGlossy) {
                if(!central_reflection_content(state.path_guide,current_world_direction,
                    unpack_normal(state.current.packed_normal),continuation_world,content_hit_t))return state;
            }
        }
        const float3 current_primary_normal = normalize3(
            unpack_normal(state.current.packed_normal));
        const float3 previous_primary_normal = normalize3(
            transform_vector(
                params.previous_world_to_local, previous_world_normal));
        const float3 previous_primary_local = transform_point(
            params.previous_world_to_local, previous_world_point);
        const float3 aperture_previous_local = make_float3(
            view_position(params, current_local.view_id), 0.0f, 0.0f);
        bool response_query_valid = false;
        if (environment) {
            const float3 continuation_previous_local = normalize3(
                transform_vector(
                    params.previous_world_to_local, continuation_world));
            if (branch == 0u) {
                const float3 virtual_direction = normalize3(subtract(
                    continuation_previous_local,
                    multiply(previous_primary_normal,
                        2.0f * dot3(continuation_previous_local,
                                    previous_primary_normal))));
                query_origin_previous_local = aperture_previous_local;
                query_direction_previous_local = virtual_direction;
                response_query_valid = finitef(virtual_direction.x) &&
                    finitef(virtual_direction.y) &&
                    finitef(virtual_direction.z) &&
                    intersect_forward_plane(
                        aperture_previous_local, virtual_direction,
                        previous_primary_local, previous_primary_normal,
                        validation_position_previous_local);
            } else {
                const float eta = path_guide_eta(state.path_guide);
                const float normal_projection = dot3(
                    continuation_previous_local,
                    previous_primary_normal);
                const float3 tangent = subtract(
                    continuation_previous_local,
                    multiply(previous_primary_normal, normal_projection));
                const float3 incident_tangent = multiply(tangent, eta);
                const float tangent_squared = dot3(
                    incident_tangent, incident_tangent);
                if (finitef(tangent_squared) && tangent_squared < 1.0f) {
                    const float3 current_direction_previous_local =
                        normalize3(transform_vector(
                            params.previous_world_to_local,
                            current_world_direction));
                    const float normal_sign = dot3(
                        current_direction_previous_local,
                        previous_primary_normal) >= 0.0f ? 1.0f : -1.0f;
                    query_origin_previous_local = aperture_previous_local;
                    query_direction_previous_local = normalize3(add(
                        incident_tangent,
                        multiply(previous_primary_normal,
                            normal_sign * sqrt_rn(
                                fmaxf(1.0f - tangent_squared, 0.0f)))));
                    response_query_valid =
                        finitef(query_direction_previous_local.x) &&
                        finitef(query_direction_previous_local.y) &&
                        finitef(query_direction_previous_local.z) &&
                        intersect_forward_plane(
                            aperture_previous_local,
                            query_direction_previous_local,
                            previous_primary_local,
                            previous_primary_normal,
                            validation_position_previous_local);
                }
            }
        } else {
            const float hit_t = content_hit_t;
            if (finitef(hit_t) && hit_t > 0.0f) {
                const float3 secondary_current_world = add(
                    current_world_primary_point,
                    multiply(continuation_world, hit_t));
                const std::uint32_t secondary_instance =
                    state.path_guide.metadata & 0x1fffu;
                const native::RigidInstanceMotionRecord *secondary_motion =
                    rigid_motion_record(params, secondary_instance);
                if (secondary_motion != nullptr &&
                    (secondary_motion->metadata.x & kRigidMotionValid) != 0u) {
                    const float3 secondary_previous_world =
                        transform_motion_point(
                            *secondary_motion, secondary_current_world);
                    const float3 secondary_previous_local = transform_point(
                        params.previous_world_to_local,
                        secondary_previous_world);
                    if (branch == 0u) {
                        const float3 virtual_previous = subtract(
                            secondary_previous_local,
                            multiply(previous_primary_normal,
                                2.0f * dot3(
                                    subtract(secondary_previous_local,
                                             previous_primary_local),
                                    previous_primary_normal)));
                        // A virtual point is the endpoint used to construct
                        // the previous camera ray.  It is not the ray origin:
                        // using it as both origin and endpoint shifts the
                        // focal/aperture intersections onto a different line
                        // and produces a systematically wrong LF address.
                        query_origin_previous_local = aperture_previous_local;
                        query_direction_previous_local = normalize3(subtract(
                            virtual_previous, aperture_previous_local));
                        response_query_valid =
                            finitef(query_direction_previous_local.x) &&
                            finitef(query_direction_previous_local.y) &&
                            finitef(query_direction_previous_local.z) &&
                            intersect_forward_plane(
                                aperture_previous_local,
                                query_direction_previous_local,
                                previous_primary_local,
                                previous_primary_normal,
                                validation_position_previous_local);
                    } else {
                        float3 interface_point{};
                        response_query_valid = solve_planar_snell_point(
                            aperture_previous_local,
                            previous_primary_local,
                            previous_primary_normal,
                            secondary_previous_local,
                            path_guide_eta(state.path_guide),
                            interface_point);
                        if (response_query_valid) {
                            // The solved interface point constrains the
                            // refracted camera ray; the physical origin stays
                            // at the previous aperture.
                            query_origin_previous_local =
                                aperture_previous_local;
                            query_direction_previous_local = normalize3(
                                subtract(interface_point,
                                         aperture_previous_local));
                            validation_position_previous_local =
                                interface_point;
                        }
                    }
                    if (response_query_valid) {
                        const float3 previous_transport = subtract(
                            secondary_previous_local,
                            validation_position_previous_local);
                        const float previous_hit_t = sqrt_rn(fmaxf(
                            dot3(previous_transport, previous_transport),
                            0.0f));
                        response_query_valid = finitef(previous_hit_t) &&
                            previous_hit_t > 0.0f;
                        if (response_query_valid) {
                            state.responsive_previous_hit_log =
                                log2f(1.0f + previous_hit_t);
                        }
                    }
                }
            }
        }
        state.response_vmb_solved = response_query_valid;
        if (!response_query_valid) return state;
        state.exact_identity_query = false;
    }
    if (fabsf(query_direction_previous_local.z) <= 1.0e-6f) {
        return state;
    }
    const float reciprocal_z = 1.0f / query_direction_previous_local.z;
    const float3 query_u = add(
        query_origin_previous_local,
        multiply(query_direction_previous_local,
                 -query_origin_previous_local.z * reciprocal_z));
    const float3 query_x = add(
        query_origin_previous_local,
        multiply(query_direction_previous_local,
                 (params.focal_distance - query_origin_previous_local.z) *
                     reciprocal_z));
    if constexpr (EncodedAddress) {
        if (params.current_motion == 0u ||
            params.current_motion_validity == 0u ||
            pointer<const std::uint32_t>(
                params.current_motion_validity)[index] == 0u) {
            return state;
        }
        const denoise_native::Float2 motion =
            pointer<const denoise_native::Float2>(params.current_motion)[index];
        if (!finitef(motion.x) || !finitef(motion.y)) {
            return state;
        }
        state.query_lattice_x = static_cast<float>(state.encoded_x) - motion.x;
        state.query_lattice_y = static_cast<float>(state.encoded_y) - motion.y;
        state.query_u_x = query_u.x;
        const AngularCoordinate angular = angular_coordinate<CalibratedAngles>(
            params, state.query_u_x);
        state.continuous_view = angular.continuous_view;
        state.inverse_u_spacing = angular.inverse_spacing;
        state.query_direction_previous_local = query_direction_previous_local;
        state.predicted_previous_position_local =
            validation_position_previous_local;
        state.predicted_previous_normal_local = transform_vector(
            params.previous_world_to_local, previous_world_normal);
        state.lf_displacement = clampf(
            0.02f * sqrt_rn(
                motion.x * motion.x * (1.0f / 9.0f) +
                motion.y * motion.y),
            0.0f, 0.25f);
        state.query_valid = finitef(state.query_lattice_x) &&
            finitef(state.query_lattice_y) && finitef(state.query_u_x) &&
            state.query_lattice_x >= 0.0f && state.query_lattice_y >= 0.0f &&
            state.query_lattice_x <=
                static_cast<float>(params.output_width * 3u - 1u) &&
            state.query_lattice_y <=
                static_cast<float>(params.output_height - 1u);
        return state;
    }
    state.query_lattice_x =
        (query_x.x + params.focal_width * 0.5f) *
            params.inverse_focal_lattice_width - 0.5f;
    state.query_lattice_y =
        (params.focal_height * 0.5f - query_x.y) *
            params.inverse_focal_lattice_height - 0.5f;
    state.query_u_x = query_u.x;
    const float exact_x_tolerance = 32.0f * FLT_EPSILON * fmaxf(
        static_cast<float>(state.encoded_x), 1.0f);
    const float exact_y_tolerance = 32.0f * FLT_EPSILON * fmaxf(
        static_cast<float>(state.encoded_y), 1.0f);
    const float exact_view_tolerance = 32.0f * FLT_EPSILON *
        static_cast<float>(max(params.view_count, 1u));
    const std::uint32_t spacing_index = min(
        current_local.view_id, max(params.view_count, 1u) - 1u);
    const float exact_u_error_views = fabsf(
        state.query_u_x - current_local.origin.x) *
        inverse_view_spacing(params, spacing_index);
    const bool exact_sample_query =
        fabsf(state.query_lattice_x - static_cast<float>(state.encoded_x)) <=
            exact_x_tolerance &&
        fabsf(state.query_lattice_y - static_cast<float>(state.encoded_y)) <=
            exact_y_tolerance &&
        exact_u_error_views <= exact_view_tolerance;
    if (identity_transport &&
        (params.static_layout != 0u || exact_sample_query)) {
        state.query_lattice_x = static_cast<float>(state.encoded_x);
        state.query_lattice_y = static_cast<float>(state.encoded_y);
        state.query_u_x = current_local.origin.x;
        state.exact_identity_query = true;
    }
    const AngularCoordinate angular = angular_coordinate<CalibratedAngles>(
        params, state.query_u_x);
    state.continuous_view = angular.continuous_view;
    state.inverse_u_spacing = angular.inverse_spacing;
    if constexpr (!StaticEntry) {
        const float x_motion_pixels =
            (state.query_lattice_x - static_cast<float>(state.encoded_x)) /
            3.0f;
        const float y_motion_pixels =
            state.query_lattice_y - static_cast<float>(state.encoded_y);
        const float u_motion_views =
            (state.query_u_x - current_local.origin.x) *
            state.inverse_u_spacing;
        state.lf_displacement = clampf(
            0.02f * sqrt_rn(
                x_motion_pixels * x_motion_pixels +
                y_motion_pixels * y_motion_pixels +
                u_motion_views * u_motion_views),
            0.0f,
            0.25f);
    }
    state.query_direction_previous_local = query_direction_previous_local;
    state.predicted_previous_position_local =
        validation_position_previous_local;
    state.predicted_previous_normal_local = transform_vector(
        params.previous_world_to_local, previous_world_normal);
    state.query_valid = finitef(state.query_lattice_x) &&
        finitef(state.query_lattice_y) && finitef(state.query_u_x);
    return state;
}

struct PhaseTap {
    std::uint32_t index{0xffffffffu};
    std::uint32_t site{};
    float weight{};
    float distance_squared{FLT_MAX};
};

constexpr std::uint32_t kPhaseSiteCoordinateMask = (1u << 13u) - 1u;
constexpr std::uint32_t kPhaseSiteRowShift = 13u;
constexpr std::uint32_t kPhaseSiteViewShift = 26u;

__device__ __forceinline__ std::uint32_t pack_phase_site(
    std::uint32_t pixel,
    std::uint32_t row,
    std::uint32_t view) {
    return pixel | (row << kPhaseSiteRowShift) |
        (view << kPhaseSiteViewShift);
}

struct FixedPhaseTaps {
    PhaseTap tap0{};
    PhaseTap tap1{};
    PhaseTap tap2{};
    PhaseTap tap3{};
    PhaseTap tap4{};
    PhaseTap tap5{};
    PhaseTap tap6{};
    PhaseTap tap7{};
    std::uint32_t dominant_index{0xffffffffu};
    float dominant_weight{-1.0f};
};

#include "measurement_reconstruction_surface_query.h"

struct ContinuousPoint {
    float lattice_x{};
    float lattice_y{};
    float u_x{};
    float du_dp{};
    float du_dy{};
    float wrapped_phase{};
    bool valid{};
};

struct ContinuousMetric {
    float a00{};
    float a01{};
    float a11{};
    float determinant{};
    bool valid{};
};

__device__ __forceinline__ int nearest_branch(float value) {
    const float lower = floorf(value);
    return static_cast<int>(value - lower > 0.5f ? lower + 1.0f : lower);
}

__device__ __forceinline__ int nearest_lattice_integer(float value) {
    const float lower = floorf(value);
    return static_cast<int>(value - lower > 0.5f ? lower + 1.0f : lower);
}

struct ReducedLatticeBasis {
    int b0x{1};
    int b0y{};
    int b1x{};
    int b1y{1};
    bool valid{};
};

__device__ __forceinline__ float metric_inner_product(
    const ContinuousMetric &metric,
    int ax,
    int ay,
    int bx,
    int by) {
    return static_cast<float>(ax) *
            (metric.a00 * static_cast<float>(bx) +
             metric.a01 * static_cast<float>(by)) +
        static_cast<float>(ay) *
            (metric.a01 * static_cast<float>(bx) +
             metric.a11 * static_cast<float>(by));
}

template<bool StopAfterZero = false>
__device__ __forceinline__ ReducedLatticeBasis reduce_lattice_basis(
    const ContinuousMetric &metric) {
    ReducedLatticeBasis result{};
    if (!metric.valid) return result;
    #pragma unroll
    for (int iteration = 0; iteration < 4; ++iteration) {
        float norm0 = metric_inner_product(
            metric, result.b0x, result.b0y, result.b0x, result.b0y);
        float norm1 = metric_inner_product(
            metric, result.b1x, result.b1y, result.b1x, result.b1y);
        if (!finitef(norm0) || !finitef(norm1) ||
            !(norm0 > 1.0e-12f) || !(norm1 > 1.0e-12f)) {
            return {};
        }
        if (norm1 < norm0) {
            const int temporary_x = result.b0x;
            const int temporary_y = result.b0y;
            result.b0x = result.b1x;
            result.b0y = result.b1y;
            result.b1x = temporary_x;
            result.b1y = temporary_y;
            const float temporary_norm = norm0;
            norm0 = norm1;
            norm1 = temporary_norm;
        }
        const float projection = metric_inner_product(
            metric, result.b0x, result.b0y, result.b1x, result.b1y) / norm0;
        if (!finitef(projection) || fabsf(projection) > 4096.0f) return {};
        const int multiple = nearest_lattice_integer(projection);
        if constexpr (StopAfterZero) {
            if (multiple == 0) break;
        }
        const long long next_x = static_cast<long long>(result.b1x) -
            static_cast<long long>(multiple) * result.b0x;
        const long long next_y = static_cast<long long>(result.b1y) -
            static_cast<long long>(multiple) * result.b0y;
        if (next_x < INT_MIN || next_x > INT_MAX ||
            next_y < INT_MIN || next_y > INT_MAX) {
            return {};
        }
        result.b1x = static_cast<int>(next_x);
        result.b1y = static_cast<int>(next_y);
    }
    const long long determinant =
        static_cast<long long>(result.b0x) * result.b1y -
        static_cast<long long>(result.b0y) * result.b1x;
    result.valid = determinant == 1ll || determinant == -1ll;
    return result;
}

struct InterfaceFootprint {
    float3 reflected{},transmitted{};
    float reflection{},front{},width2{};
};
__device__ __forceinline__ InterfaceFootprint interface_footprint(float3 incident,float3 ns,float3 ng,float front,float width2) {
    InterfaceFootprint f{};f.front=front;f.width2=width2;
    const float eta=dot3(incident,ng)<0.f?front:1.f/front,ratio=1.f/eta;
    const float3 face=dot3(incident,ns)<=0.f?ns:multiply(ns,-1.f);
    const float ci=fminf(1.f,fmaxf(0.f,-dot3(incident,face)));
    const float ct2=1.f-ratio*ratio*fmaxf(0.f,1.f-ci*ci);
    f.reflected=reflect_direction(incident,ns);f.reflection=1.f;
    if(ct2>0.f) {
        const float ct=sqrtf(ct2);
        const float s=(ci-eta*ct)/fmaxf(ci+eta*ct,1.e-20f);
        const float p=(eta*ci-ct)/fmaxf(eta*ci+ct,1.e-20f);
        f.reflection=fminf(1.f,.5f*(s*s+p*p));
        f.transmitted=normalize3(add(multiply(incident,ratio),multiply(face,fmaf(ratio,ci,-ct))));
    }
    return f;
}
__device__ __forceinline__ float angular_affinity(float3 a,float3 b,float w2) {
    const float c=clampf(dot3(a,b),-1.f,1.f);
    return w2/fmaxf(w2+(1.f-c)/fmaxf(1.f+c,1.e-8f),1.e-20f);
}
struct CandidateEvaluation {
    float dielectric_overlap{-1.f};
    float incident_cosine{1.f};
    float reflected_cosine{1.f};
    float diffuse_footprint{1.f};
    float previous_innovation{};
    bool accepted{};
    std::uint32_t rejection_mask{};
    float correspondence_confidence{};
};

// Current MR writes confidence zero. Its complete metadata is held in bits
// 0..14 and 30. The resident storage uses the other sixteen bits for an
// independent half-precision angular correction and removes them on every
// read of the original TemporalSample32.
constexpr unsigned kCurrentMrMetadataMask=0x40007fffu;
__device__ __forceinline__ native::TemporalSample prior_temporal_sample(
    const native::Phase3Params &params, unsigned index) {
    if (!params.history_geometry_read)
        return pointer<const native::TemporalSample>(params.history_read)[index];
    const uint4 head=params.native_first_history_head && index==0u
        ? *pointer<const uint4>(params.native_first_history_head) : pointer<const uint4>(params.history_geometry_read)[index];
    uint4 tail=pointer<const uint4>(params.history_read)[index];
    if(params.native_first_history_head)tail.x &= kCurrentMrMetadataMask;
    native::TemporalSample result;
    memcpy(&result,&head,sizeof(head));
    memcpy(reinterpret_cast<unsigned char*>(&result)+sizeof(head),&tail,sizeof(tail));
    return result;
}

template<int Radius, bool StaticEntry, bool MeasurementGuide=false, bool AcceptanceOnly=false, bool ContiguousHistory=false, bool Optical=false, bool Dielectric=false>
__device__ __forceinline__ CandidateEvaluation evaluate_candidate(
    const native::Phase3Params &params,
    const QueryState<Radius, StaticEntry> &state,
    std::uint32_t candidate_index,
    std::uint32_t candidate_site,
    native::TemporalSample &previous,
    std::uint64_t previous_material_normals=0u,
    float3 predicted_material_normal=make_float3(0.f,0.f,0.f), InterfaceFootprint receiver={}) {
    CandidateEvaluation result{};
    if constexpr (ContiguousHistory) {
        previous = pointer<const native::TemporalSample>(params.history_read)[candidate_index];
        if constexpr(Optical) {
            const unsigned bits=previous.metadata;
            result.previous_innovation=half_to_float(static_cast<unsigned short>(((bits>>15u)&0x7fffu)|((bits>>16u)&0x8000u)));
            previous.metadata&=kCurrentMrMetadataMask;

        }
    } else {
        previous = prior_temporal_sample(params, candidate_index);
    }
    const std::uint32_t previous_valid = surface_valid(previous.metadata);
    const std::uint32_t previous_domain = signal_domain(previous.metadata);
    const std::uint32_t previous_class = surface_class(previous.metadata);

    const bool response_log_conditioned =
        (0u != 0u) &&
        state.deterministic_response_path;
    const bool response_log_stored =
        (previous.metadata & (1u << kResponseLogHistoryBit)) != 0u;

    const std::uint32_t current_response_auxiliary =
        0u;

    const bool history_finite =
        half_is_finite(previous.signal_physical_mean_rms.x) &&
        half_is_finite(previous.signal_physical_mean_rms.z) &&
        half_is_finite(previous.signal_physical_mean_rms.w) &&
        ((half_is_finite(previous.sigma_albedo.x) &&
          half_is_finite(previous.sigma_albedo.y)));
    const bool validity_ok = state.current_valid == previous_valid;
    const bool domain_ok = state.current_domain == previous_domain;
    const bool class_ok = state.current_class == previous_class;
    const bool identity_ok = state.current.surface_key == previous.surface_key;

    const bool response_log_layout_ok =
        MeasurementGuide || response_log_conditioned == response_log_stored;

    bool accepted = history_finite && validity_ok && domain_ok && class_ok &&
        identity_ok && response_log_layout_ok;
    if (!history_finite) result.rejection_mask |= 1u << 0u;
    if (!validity_ok) result.rejection_mask |= 1u << 1u;
    if (!identity_ok) result.rejection_mask |= 1u << 3u;
    if (!class_ok) result.rejection_mask |= 1u << 7u;
    if (!domain_ok) result.rejection_mask |= 1u << 8u;

    if (!response_log_layout_ok) result.rejection_mask |= 1u << 13u;

    if constexpr (AcceptanceOnly) {
        if (!accepted) return result;
    }
    float response_direction_risk = 0.0f;

    if (state.responsive_path &&
        1u != 0u) {
        const bool environment =
            ((state.path_guide.metadata >> 22u) & 1u) != 0u;
        const float previous_hit_log = half_to_float(MeasurementGuide ? previous.signal_physical_mean_rms.w : previous.sigma_albedo.x);
        if constexpr(MeasurementGuide) {
            const bool stored_valid=(previous.metadata&(1u<<29u))!=0u;
            const bool stored_environment=(previous.metadata&(1u<<28u))!=0u;
            const bool identity=environment || ((previous.metadata>>15u)&0x1fffu)==(state.path_guide.metadata&0x1fffu);
            accepted=accepted && stored_valid && stored_environment==environment && identity;
        }
        const float tolerance = fmaxf(
            state.current_roughness, 1.0f / 255.0f);
        const bool hit_ok = environment ||
            (finitef(previous_hit_log) &&
             fabsf(state.responsive_previous_hit_log - previous_hit_log) <=
                 tolerance);
        if (!hit_ok) result.rejection_mask |= 1u << 10u;
        accepted = accepted && hit_ok;
    }

    if constexpr (StaticEntry) {
        if (3u != 0u && state.exact_identity_query &&
            candidate_index == state.index) {
            float maximum_risk = response_direction_risk;
            if (state.current_valid != 0u && previous_valid != 0u) {
                const float previous_albedo =
                    half_to_float(
                          previous.sigma_albedo.y);
                const float albedo_scale = fmaxf(
                    fmaxf(fabsf(state.current_albedo),
                          fabsf(previous_albedo)),
                    0.05f);
                const float albedo_risk =
                    fabsf(state.current_albedo - previous_albedo) /
                        (albedo_scale * params.maximum_albedo_relative_error);
                const bool albedo_ok = albedo_risk <= 1.0f;
                const bool depth_ok = previous.depth >= 0.0f &&
                    finitef(previous.depth);
                const float3 previous_normal = transform_vector(
                    params.previous_world_to_local,
                    unpack_normal(previous.packed_normal));
                const float normal_similarity = dot3(
                    state.predicted_previous_normal_local, previous_normal);
                const float tangent_denominator = dot3(
                    state.query_direction_previous_local, previous_normal);
                const float plane_residual = fabsf(
                    (state.current.depth - previous.depth) *
                    tangent_denominator);
                const float depth_footprint = fmaxf(
                    previous.depth * params.inverse_focal_distance, 1.0f);
                const float position_tolerance = fmaxf(
                    params.base_position_tolerance * depth_footprint,
                    1.0e-6f * params.position_tolerance_lattice);
                const float depth_tolerance = position_tolerance * fminf(
                    fabsf(tangent_denominator), 1.0f);
                const bool position_ok = depth_ok &&
                    finitef(plane_residual) &&
                    fabsf(tangent_denominator) > 1.0e-6f &&
                    plane_residual <= depth_tolerance;
                const bool normal_ok = normal_similarity >=
                    params.minimum_normal_similarity;
                if (!position_ok) result.rejection_mask |= 1u << 4u;
                if (!normal_ok) result.rejection_mask |= 1u << 5u;
                if (!albedo_ok) result.rejection_mask |= 1u << 6u;
                accepted = accepted && position_ok && normal_ok && albedo_ok;
                const float position_risk = clampf(
                    plane_residual / fmaxf(depth_tolerance, 1.0e-12f),
                    0.0f, 1.0f);
                const float normal_risk = clampf(
                    (1.0f - normal_similarity) *
                        params.inverse_normal_error_range,
                    0.0f, 1.0f);
                maximum_risk = fmaxf(
                    position_risk,
                    fmaxf(normal_risk, clampf(albedo_risk, 0.0f, 1.0f)));
            }
            result.accepted = accepted;
            result.correspondence_confidence = accepted
                ? 1.0f - maximum_risk
                : 0.0f;
            return result;
        }
    }

    const std::uint32_t candidate_pixel =
        candidate_site & kPhaseSiteCoordinateMask;
    const std::uint32_t candidate_y =
        (candidate_site >> kPhaseSiteRowShift) & kPhaseSiteCoordinateMask;
    const std::uint32_t candidate_view = MeasurementGuide && params.view_count>64u
        ? producer_view_id(params,candidate_pixel*3u+state.channel,candidate_y)
        : candidate_site >> kPhaseSiteViewShift;
    const std::uint32_t candidate_x = candidate_pixel * 3u + state.channel;
    const LocalRay previous_ray = analytical_local_ray(
        params, candidate_x, candidate_y, candidate_view);
    const float ray_error = fmaxf(
        1.0f - dot3(state.query_direction_previous_local,
                    previous_ray.direction), 0.0f);
    const bool ray_ok = ray_error <= params.maximum_ray_angular_error
        ;
    if constexpr(Dielectric)result.incident_cosine=1.f-ray_error;
    if (!ray_ok) result.rejection_mask |= 1u << 9u;
    accepted = accepted && ray_ok;
    float maximum_risk = fmaxf(
        response_direction_risk,
        clampf(
            ray_error * params.inverse_maximum_ray_angular_error,
            0.0f, 1.0f));
    if (state.current_valid != 0u && previous_valid != 0u) {
        const float previous_albedo =
            half_to_float(
                  previous.sigma_albedo.y);
        const float albedo_scale = fmaxf(
            fmaxf(fabsf(state.current_albedo), fabsf(previous_albedo)), 0.05f);
        const float albedo_risk =
            fabsf(state.current_albedo - previous_albedo) /
                (albedo_scale * params.maximum_albedo_relative_error);
        const bool albedo_ok = albedo_risk <= 1.0f;
        const bool depth_ok = previous.depth >= 0.0f && finitef(previous.depth);
        const float3 previous_normal = transform_vector(
            params.previous_world_to_local,
            unpack_normal(previous.packed_normal));
        const float depth_footprint = fmaxf(
            previous.depth * params.inverse_focal_distance, 1.0f);
        const float normal_similarity = dot3(
            state.predicted_previous_normal_local, previous_normal);
        if constexpr(Optical) {
            if(state.current_class==kSurfaceGlossy || state.current_class==kSurfaceNearSpecular) {
                const auto desired=reflect_direction(state.query_direction_previous_local,
                                                    state.predicted_previous_normal_local);
                const auto sampled=reflect_direction(previous_ray.direction,previous_normal);
                result.reflected_cosine=clampf(dot3(desired,sampled),-1.f,1.f);
            } else if(state.current_class==kSurfaceDiffuse) {
                const auto point=state.predicted_previous_position_local;
                if(point.z>0.f) {
                    const float native_u=previous_ray.origin.x;
                    const float fx=fmaf(point.x-native_u,params.focal_distance/point.z,native_u);
                    const float dx=fmaf(fx+.5f*params.focal_width,params.inverse_focal_lattice_width,-.5f)-float(candidate_x);
                    const float dy=state.query_lattice_y-float(candidate_y);
                    // The initial filtered observation composes the step-one
                    // B-spline with a three-point prefilter in encoded coordinates.
                    const float variance=1.f+(1.f-.786986f);
                    result.diffuse_footprint=__expf(-fmaf(dx,dx,dy*dy)/(4.f*variance));
                }
            }
        }
        const float tangent_denominator = dot3(
            previous_ray.direction, previous_normal);
        const float plane_numerator = dot3(
            subtract(state.predicted_previous_position_local,
                     previous_ray.origin),
            previous_normal);
        const float plane_residual = fabsf(
            plane_numerator - previous.depth * tangent_denominator);
        const float position_tolerance = fmaxf(
            params.base_position_tolerance * depth_footprint,
            1.0e-6f * params.position_tolerance_lattice);
        const float depth_tolerance = position_tolerance * fminf(
            fabsf(tangent_denominator), 1.0f);
        const bool position_ok = depth_ok && finitef(plane_numerator) &&
            finitef(plane_residual) && fabsf(tangent_denominator) > 1.0e-6f &&
            plane_numerator * tangent_denominator >= 0.0f &&
            plane_residual <= depth_tolerance;
        const bool normal_ok = normal_similarity >=
            params.minimum_normal_similarity;
        if (!position_ok) result.rejection_mask |= 1u << 4u;
        if (!normal_ok) result.rejection_mask |= 1u << 5u;
        if (!albedo_ok) result.rejection_mask |= 1u << 6u;
        accepted = accepted && position_ok && normal_ok && albedo_ok;
        if constexpr(Dielectric) {
            if(accepted && receiver.front>0.f && previous_material_normals) {
                const auto *ns=pointer<const unsigned>(previous_material_normals);
                const unsigned code=ns[candidate_index];
                const float front=__uint_as_float(ns[size_t(params.output_width)*3u*params.output_height+
                                                     unsigned(previous.surface_key>>32u)]);
                if(code!=0xffffffffu && front>0.f) {
                    const auto n=normalize3(transform_vector(params.previous_world_to_local,unpack_normal(code)));
                    const auto donor=interface_footprint(previous_ray.direction,n,previous_normal,front,receiver.width2);
                    const float r=sqrtf(receiver.reflection*donor.reflection);
                    const float t=sqrtf((1.f-receiver.reflection)*(1.f-donor.reflection));
                    const float affinity=r*angular_affinity(receiver.reflected,donor.reflected,receiver.width2)+
                        (t>0.f?t*angular_affinity(receiver.transmitted,donor.transmitted,receiver.width2):0.f);
                    result.dielectric_overlap=clampf(affinity,0.f,1.f);
                }
            }
        }
        const float position_risk = clampf(
            plane_residual / fmaxf(depth_tolerance, 1.0e-12f),
            0.0f, 1.0f);
        const float normal_risk = clampf(
            (1.0f - normal_similarity) * params.inverse_normal_error_range,
            0.0f, 1.0f);
        const float bounded_albedo_risk = clampf(albedo_risk, 0.0f, 1.0f);
        maximum_risk = fmaxf(
            maximum_risk,
            fmaxf(position_risk, fmaxf(normal_risk, bounded_albedo_risk)));
    }
    result.accepted = accepted;
    result.correspondence_confidence = accepted
        ? 1.0f - maximum_risk
        : 0.0f;
    return result;
}

} // namespace
