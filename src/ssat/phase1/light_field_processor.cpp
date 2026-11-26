#include "ssat/phase1/light_field_processor.h"
#include "ssat/phase1/phase_mapping_gpu.h"
#include "ssat/resident_storage.h"

#include "dsl/dsl.h"
#include "math/base.h"
#include "math/basic_types.h"
#include "math/transform.h"
#include "rhi/resources/shader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lfs::lightfield {
using namespace ocarina;

namespace {

} // namespace

struct RayGenerationParam {
    BufferDesc<float4> origin;
    BufferDesc<float4> direction;
    BufferDesc<float2> motion;
    BufferDesc<uint32_t> view_id;
    BufferDesc<float> inverse_ray_length;
    BufferDesc<float> view_x_lut;
    BufferDesc<float> focal_x_lut;
    BufferDesc<float> focal_y_lut;
    float4x4 local_to_world{};
    uint32_t output_width{};
    uint32_t output_height{};
    float focal_distance{};
    float focal_width{};
    float focal_height{};
    uint32_t motion_enabled{};
};

struct StaticRayTableParam {
    BufferDesc<uint32_t> view_id;
    BufferDesc<float> inverse_ray_length;
    BufferDesc<float> view_x_lut;
    uint32_t output_width{};
    uint32_t output_height{};
    uint32_t view_count{};
    float pitch{};
    float slant_tangent{};
    float offset{};
    float focal_distance{};
    float focal_width{};
    float focal_height{};
};

struct MotionParam {
    BufferDesc<float> depth;
    BufferDesc<uint32_t> validity;
    BufferDesc<float4> ray_origin;
    BufferDesc<float4> ray_direction;
    BufferDesc<float2> motion;
    BufferDesc<float4> world_velocity;
    BufferDesc<uint32_t> motion_validity;
    float4x4 previous_local_to_world{};
    uint32_t history_valid{};
    uint32_t output_width{};
    uint32_t output_height{};
    uint32_t view_count{};
    float pitch{};
    float slant{};
    float offset{};
    float focal_distance{};
    float focal_width{};
    float focal_height{};
    float array_angle_degrees{};
};

struct InterlaceParam {
    BufferDesc<float4> subpixel_radiance;
    BufferDesc<float4> canonical;
    uint32_t output_width{};
};

} // namespace lfs::lightfield

OC_PARAM_STRUCT(lfs::lightfield, RayGenerationParam,
                origin, direction, motion, view_id, inverse_ray_length,
                view_x_lut, focal_x_lut, focal_y_lut,
                local_to_world,
                output_width, output_height,
                focal_distance, focal_width, focal_height,
                motion_enabled) {};
OC_PARAM_STRUCT(lfs::lightfield, StaticRayTableParam,
                view_id, inverse_ray_length, view_x_lut,
                output_width, output_height, view_count,
                pitch, slant_tangent, offset, focal_distance,
                focal_width, focal_height) {};
OC_PARAM_STRUCT(lfs::lightfield, MotionParam,
                depth, validity, ray_origin, ray_direction, motion,
                world_velocity, motion_validity,
                previous_local_to_world, history_valid,
                output_width, output_height, view_count,
                pitch, slant, offset, focal_distance,
                focal_width, focal_height, array_angle_degrees) {};
OC_PARAM_STRUCT(lfs::lightfield, InterlaceParam,
                subpixel_radiance, canonical, output_width) {};

namespace lfs::lightfield {
namespace {

[[nodiscard]] float4x4 camera_to_world(
    const backend::CameraState &camera) noexcept {
    const float3 position = make_float3(
        camera.position[0], camera.position[1], camera.position[2]);
    const float3 target = make_float3(
        camera.target[0], camera.target[1], camera.target[2]);
    const float3 up = make_float3(
        camera.up[0], camera.up[1], camera.up[2]);
    return look_at<EPort::H>(position, target, up);
}

[[nodiscard]] PhaseTransform camera_to_phase_transform(
    const backend::CameraState &camera) noexcept {
    return look_at_phase_transform(
        {camera.position[0], camera.position[1], camera.position[2]},
        {camera.target[0], camera.target[1], camera.target[2]},
        {camera.up[0], camera.up[1], camera.up[2]});
}

[[nodiscard]] backend::Error missing(std::string message) {
    return {backend::ErrorCode::ResourceMissing, std::move(message)};
}

template<typename T>
[[nodiscard]] uint64_t native_address(const BufferView<T> &view) noexcept {
    return static_cast<uint64_t>(view.handle()) +
        static_cast<uint64_t>(view.offset_in_byte());
}

[[nodiscard]] std::array<float, 16u> matrix_to_array(
    const float4x4 &value) noexcept {
    return {
        value[0].x, value[0].y, value[0].z, value[0].w,
        value[1].x, value[1].y, value[1].z, value[1].w,
        value[2].x, value[2].y, value[2].z, value[2].w,
        value[3].x, value[3].y, value[3].z, value[3].w};
}

} // namespace

struct LightFieldProcessor::Impl {
    LenticularParams lenticular{};
    LightFieldGeometryParams geometry{};
    backend::FrameProcessor *denoiser{};
    PhaseFrameStateConsumer *phase_consumer{};
    LightFieldTelemetry telemetry{};
    Device *device{};
    backend::Extent2D sample_extent{};
    backend::Extent2D output_extent{};
    uint64_t layout_generation{};
    uint64_t layout_hash{};
    float4x4 current_local_to_world{make_float4x4(1.f)};
    float4x4 previous_local_to_world{make_float4x4(1.f)};
    PhaseTransform current_phase_transform{};
    PhaseTransform previous_phase_transform{};
    bool history_valid{};
    bool motion_required{};
    bool current_rays_required{};
    bool previous_rays_required{};
    bool materialized_ray_vectors_required{};
    bool motion_bound{};
    bool frame_completed{};

    uint32_t write_bank{};
    backend::ResourceRequirementSet child_requirements{};

    std::array<Buffer<float4>, 2u> ray_origin{};
    std::array<Buffer<float4>, 2u> ray_direction{};
    Buffer<float2> motion{};
    Buffer<uint32_t> view_id{};
    Buffer<float> inverse_ray_length{};
    Buffer<float> view_x_lut{};
    Buffer<float> focal_x_lut{};
    Buffer<float> focal_y_lut{};
    Buffer<float4> canonical{};
    Shader<void(RayGenerationParam)> ray_shader{};
    Shader<void(StaticRayTableParam)> static_ray_table_shader{};
    Shader<void(MotionParam)> motion_shader{};
    Shader<void(InterlaceParam)> interlace_shader{};
    std::vector<float> view_x_host{};
    std::vector<float> focal_x_host{};
    std::vector<float> focal_y_host{};
    bool static_ray_table_pending{};
    uint64_t static_ray_table_generation{};
    uint64_t static_ray_table_hash{};

    Impl(
        LenticularParams input_lenticular,
        LightFieldGeometryParams input_geometry,
        backend::FrameProcessor *input_denoiser) noexcept
        : lenticular(std::move(input_lenticular)),
          geometry(std::move(input_geometry)),
          denoiser(input_denoiser),
          phase_consumer(dynamic_cast<PhaseFrameStateConsumer *>(input_denoiser)) {}

    void invalidate_history() noexcept {
        history_valid = false;
        motion_bound = false;
        frame_completed = false;
        if (phase_consumer) {
            phase_consumer->invalidate_phase_frame_state();
        }
    }

    std::shared_ptr<const CalibratedLattice> host_lattice{};
    std::shared_ptr<const CalibratedLattice> calibrated_lattice(Stream &stream) {
        if(host_lattice)return host_lattice;
        // This is an explicit cold preparation transfer, before any frame rays.
        // It consumes the existing StaticRay5 shader, never a second ViewId formula.
        if(!static_ray_table_pending || !telemetry.prepared)
            throw std::runtime_error("Calibrated lattice was not requested during cold preparation.");
        std::vector<uint32_t> packed(view_id.size());
        stream << view_x_lut.upload(view_x_host.data())
               << focal_x_lut.upload(focal_x_host.data())
               << focal_y_lut.upload(focal_y_host.data())
               << static_ray_table_shader(static_ray_table_parameters()).dispatch(uint32_t(view_id.size()))
               << view_id.download(packed.data()) << synchronize() << commit();
        static_ray_table_pending=false;static_ray_table_generation=layout_generation;static_ray_table_hash=layout_hash;
        auto lattice=std::make_shared<CalibratedLattice>();
        lattice->width=sample_extent.width;lattice->height=sample_extent.height;
        lattice->views=uint32_t(view_x_host.size());lattice->x=focal_x_host;lattice->y=focal_y_host;
        lattice->view.resize(size_t(sample_extent.area()));
        for(size_t i=0;i<lattice->view.size();++i)lattice->view[i]=uint8_t(packed[i/4u]>>((i%4u)*8u));
        telemetry.calibration_readback_bytes+=uint64_t(packed.size())*4u;
        ++telemetry.calibration_synchronizations;
        host_lattice=lattice;return host_lattice;
    }

    [[nodiscard]] backend::Error bind_phase_state(
        const backend::FrameContext &context) noexcept {
        if (!phase_consumer) return {};
        PhaseFrameState state{};
        state.frame_index = context.frame_index;
        state.layout_generation = context.layout_generation;
        state.layout_hash = context.layout_hash;
        state.current = make_phase_layout(
            lenticular, geometry, current_phase_transform);
        state.previous = make_phase_layout(
            lenticular, geometry, previous_phase_transform);
        state.static_view_id_address = native_address(view_id.view());
        state.static_inverse_ray_length_address =
            native_address(inverse_ray_length.view());
        state.static_view_x_address = native_address(view_x_lut.view());
        state.static_focal_x_address = native_address(focal_x_lut.view());
        state.static_focal_y_address = native_address(focal_y_lut.view());
        state.static_ray_count = inverse_ray_length.size();
        state.static_focal_x_count = focal_x_lut.size();
        state.static_focal_y_count = focal_y_lut.size();
        state.static_view_count = static_cast<uint32_t>(view_x_lut.size());
        state.previous_valid = history_valid;
        state.calibrated_lattice=[this,stream=context.stream]{
            if(!stream)throw std::runtime_error("Calibrated lattice requires a preparation stream.");
            return calibrated_lattice(*stream);
        };
        if (!state.valid() || !phase_consumer->set_phase_frame_state(state)) {
            return {backend::ErrorCode::ResourceMismatch,
                    "Light-field phase state is invalid or was rejected by the processor."};
        }
        return {};
    }

    [[nodiscard]] backend::Error allocate_and_compile(
        const backend::FrameContext &context) noexcept {
        if (context.device == nullptr ||
            context.sample_domain != backend::SampleDomain::EncodedSubpixelGrid ||
            !context.sample_extent.valid() || !context.output_extent.valid() ||
            context.sample_extent.width != context.output_extent.width * 3u ||
            context.sample_extent.height != context.output_extent.height ||
            lenticular.width != context.output_extent.width ||
            lenticular.height != context.output_extent.height ||
            lenticular.view_count == 0u || lenticular.view_count > 256u ||
            lenticular.pitch <= 0.0 || geometry.focal_distance <= 0.0) {
            return {backend::ErrorCode::ResourceMismatch,
                    "Light-field processing requires a valid 3W by H StaticRay5 layout."};
        }
        const bool changed = device != context.device ||
            sample_extent != context.sample_extent ||
            output_extent != context.output_extent ||
            layout_generation != context.layout_generation ||
            layout_hash != context.layout_hash ||
            motion_required != child_requirements.contains(
                backend::Semantic::Motion,
                backend::TemporalSlot::Current,
                0u) ||
            current_rays_required !=
                (child_requirements.contains(
                     backend::Semantic::RayOrigin,
                     backend::TemporalSlot::Current) ||
                 child_requirements.contains(
                     backend::Semantic::RayDirection,
                     backend::TemporalSlot::Current)) ||
            previous_rays_required !=
                (child_requirements.retains_for_previous(
                     backend::Semantic::RayOrigin) ||
                 child_requirements.retains_for_previous(
                     backend::Semantic::RayDirection) ||
                 child_requirements.contains(
                     backend::Semantic::RayOrigin,
                     backend::TemporalSlot::Previous) ||
                 child_requirements.contains(
                     backend::Semantic::RayDirection,
                     backend::TemporalSlot::Previous));
        if (!changed && telemetry.prepared) {
            return {};
        }

        device = context.device;
        sample_extent = context.sample_extent;
        output_extent = context.output_extent;
        layout_generation = context.layout_generation;
        layout_hash = context.layout_hash;
        // A newly allocated StaticRay5 table must never be consumed as if it
        // belonged to the previous calibrated layout. The first matching
        // primary-ray submission emits the table build on the same stream.
        static_ray_table_generation = 0u;
        static_ray_table_hash = 0u;
        motion_required = child_requirements.contains(
            backend::Semantic::Motion,
            backend::TemporalSlot::Current,
            0u);
        current_rays_required =
            child_requirements.contains(
                backend::Semantic::RayOrigin,
                backend::TemporalSlot::Current) ||
            child_requirements.contains(
                backend::Semantic::RayDirection,
                backend::TemporalSlot::Current);
        previous_rays_required =
            child_requirements.retains_for_previous(
                backend::Semantic::RayOrigin) ||
            child_requirements.retains_for_previous(
                backend::Semantic::RayDirection) ||
            child_requirements.contains(
                backend::Semantic::RayOrigin,
                backend::TemporalSlot::Previous) ||
            child_requirements.contains(
                backend::Semantic::RayDirection,
                backend::TemporalSlot::Previous);
        materialized_ray_vectors_required = current_rays_required ||
            motion_required || previous_rays_required;
        invalidate_history();
        write_bank = 0u;

        try {
            const size_t sample_count = static_cast<size_t>(sample_extent.area());
            const size_t output_count = static_cast<size_t>(output_extent.area());
            const size_t submitted_ray_count = sample_count;
            const size_t materialized_ray_count =
                materialized_ray_vectors_required ? submitted_ray_count : 0u;
            if (materialized_ray_vectors_required) {
                ray_origin[0] = device->create_buffer<float4>(
                    submitted_ray_count, "lfs.lightfield.ray_origin.current");
                ray_direction[0] = device->create_buffer<float4>(
                    submitted_ray_count, "lfs.lightfield.ray_direction.current");
            } else {
                ray_origin[0] = {};
                ray_direction[0] = {};
            }
            if (previous_rays_required) {
                ray_origin[1] = device->create_buffer<float4>(
                    submitted_ray_count, "lfs.lightfield.ray_origin.previous");
                ray_direction[1] = device->create_buffer<float4>(
                    submitted_ray_count, "lfs.lightfield.ray_direction.previous");
            } else {
                ray_origin[1] = {};
                ray_direction[1] = {};
            }
            motion = device->create_buffer<float2>(
                motion_required ? submitted_ray_count : 1u,
                motion_required
                    ? "lfs.lightfield.motion"
                    : "lfs.lightfield.motion.unused");
            if(denoiser && denoise::resident::dedicated(device))canonical={};
            else canonical = device->create_buffer<float4>(output_count,"lfs.lightfield.canonical");
            view_id = device->create_buffer<uint32_t>(
                (sample_count + 3u) / 4u,
                "lfs.lightfield.static_view_id_r8_packed");
            inverse_ray_length = device->create_buffer<float>(
                sample_count, "lfs.lightfield.static_inverse_ray_length_r32");
            view_x_lut = device->create_buffer<float>(
                std::max<size_t>(lenticular.view_count, 1u),
                "lfs.lightfield.static_view_x_lut");
            focal_x_lut = device->create_buffer<float>(
                static_cast<size_t>(sample_extent.width),
                "lfs.lightfield.static_focal_x_lut");
            focal_y_lut = device->create_buffer<float>(
                static_cast<size_t>(sample_extent.height),
                "lfs.lightfield.static_focal_y_lut");
            view_x_host.resize(std::max<uint32_t>(lenticular.view_count, 1u));
            focal_x_host.resize(static_cast<size_t>(sample_extent.width));
            focal_y_host.resize(static_cast<size_t>(sample_extent.height));
            constexpr double kPi = 3.14159265358979323846;
            for (uint32_t view = 0u; view < view_x_host.size(); ++view) {
                const double unit = view_x_host.size() > 1u
                    ? static_cast<double>(view) /
                        static_cast<double>(view_x_host.size() - 1u)
                    : 0.5;
                const double angle = (unit - 0.5) *
                    geometry.array_angle_degrees * (kPi / 180.0);
                view_x_host[view] = static_cast<float>(
                    geometry.focal_distance * std::tan(angle));
            }
            const float focal_width = static_cast<float>(geometry.focal_width);
            const float focal_height = static_cast<float>(geometry.focal_height);
            const float lattice_width = focal_width /
                static_cast<float>(sample_extent.width);
            const float lattice_height = focal_height /
                static_cast<float>(sample_extent.height);
            for (uint32_t x = 0u; x < sample_extent.width; ++x) {
                focal_x_host[x] = -focal_width * 0.5f +
                    (static_cast<float>(x) + 0.5f) * lattice_width;
            }
            for (uint32_t y = 0u; y < sample_extent.height; ++y) {
                focal_y_host[y] = focal_height * 0.5f -
                    (static_cast<float>(y) + 0.5f) * lattice_height;
            }
            static_ray_table_pending = true;
            host_lattice.reset();
            telemetry.calibration_readback_bytes=0u;
            telemetry.calibration_synchronizations=0u;

            telemetry.ray_bank_device_bytes =
                static_cast<uint64_t>(materialized_ray_count) * sizeof(float4) *
                (previous_rays_required ? 4u :
                 materialized_ray_vectors_required ? 2u : 0u);
            telemetry.motion_device_bytes =
                static_cast<uint64_t>(motion.size()) * sizeof(float2);
            telemetry.canonical_present_device_bytes =
                static_cast<uint64_t>(canonical.size()) * sizeof(float4);
            telemetry.static_lf_table_device_bytes =
                static_cast<uint64_t>(view_id.size()) * sizeof(uint32_t) +
                static_cast<uint64_t>(sample_count) * sizeof(float) +
                static_cast<uint64_t>(view_x_lut.size()) * sizeof(float) +
                static_cast<uint64_t>(focal_x_lut.size()) * sizeof(float) +
                static_cast<uint64_t>(focal_y_lut.size()) * sizeof(float);
            telemetry.total_owned_device_bytes =
                telemetry.ray_bank_device_bytes +
                telemetry.motion_device_bytes +
                telemetry.canonical_present_device_bytes +
                telemetry.static_lf_table_device_bytes;
            telemetry.dense_target_count = sample_count;
            telemetry.submitted_ray_count = submitted_ray_count;
            telemetry.radiance_write_count = submitted_ray_count;
            telemetry.factorized_primary_ray_source =
                !materialized_ray_vectors_required;
            ++telemetry.allocation_generation;
            telemetry.allocation_count =
                (materialized_ray_vectors_required ? 2u : 0u) +
                (previous_rays_required ? 2u : 0u) +
                1u + 1u + 5u;
        } catch (const std::exception &exception) {
            return {backend::ErrorCode::OutOfMemory, exception.what()};
        } catch (...) {
            return {backend::ErrorCode::OutOfMemory,
                    "Light-field GPU buffer allocation failed."};
        }

        try {
            Kernel static_ray_table_kernel = [](
                    Var<StaticRayTableParam> param) noexcept {
                const Uint word_index = dispatch_id();
                const Uint dense_width = param.output_width * 3u;
                const Uint sample_count = dense_width * param.output_height;
                Uint packed_view_ids = 0u;
                $for(lane, 4u) {
                    const Uint index = word_index * 4u + lane;
                    $if(index < sample_count) {
                        const Uint2 encoded = make_uint2(
                            index % dense_width, index / dense_width);
                        const Float phase = cast<float>(encoded.x) +
                            3.f * cast<float>(encoded.y) *
                                param.slant_tangent +
                            param.offset;
                        const Float period = max(param.pitch, 0.001f);
                        const Float wrapped = phase -
                            floor(phase / period) * period;
                        const Float view_width = period /
                            max(cast<float>(param.view_count), 1.f);
                        Uint view = min(
                            cast<uint>(floor(wrapped / view_width)),
                            max(param.view_count, 1u) - 1u);
                        view = max(param.view_count, 1u) - 1u - view;
                        const Float view_x = param.view_x_lut.read(view);
                        const Float lattice_width = param.focal_width /
                            cast<float>(dense_width);
                        const Float lattice_height = param.focal_height /
                            cast<float>(param.output_height);
                        const Float focal_x = -param.focal_width * 0.5f +
                            (cast<float>(encoded.x) + 0.5f) * lattice_width;
                        const Float focal_y = param.focal_height * 0.5f -
                            (cast<float>(encoded.y) + 0.5f) * lattice_height;
                        const Float3 delta = make_float3(
                            focal_x - view_x, focal_y, param.focal_distance);
                        const Float inverse_length = rsqrt(max(
                            dot(delta, delta), 0x1p-60f));
                        packed_view_ids |= (view & 0xffu) << (lane * 8u);
                        param.inverse_ray_length.write(index, inverse_length);
                    };
                };
                param.view_id.write(word_index, packed_view_ids);
            };

            Kernel ray_kernel = [](Var<RayGenerationParam> param) noexcept {
                const Uint2 encoded = dispatch_idx().xy();
                const Uint dense_index = dispatch_id();
                const Uint packed_view_ids =
                    param.view_id.read(dense_index >> 2u);
                const Uint view = (packed_view_ids >>
                    ((dense_index & 3u) * 8u)) & 0xffu;
                const Float view_x = param.view_x_lut.read(view);
                const Float focal_x = param.focal_x_lut.read(encoded.x);
                const Float focal_y = param.focal_y_lut.read(encoded.y);
                const Float3 origin_local = make_float3(view_x, 0.f, 0.f);
                const Float3 direction_local = make_float3(
                    focal_x - view_x, focal_y, param.focal_distance) *
                    param.inverse_ray_length.read(dense_index);
                const Float3 origin = transform_point(
                    param.local_to_world, origin_local);
                const Float3 direction = transform_vector(
                    param.local_to_world, direction_local);
                param.origin.write(
                    dense_index, make_float4(origin, 1.f));
                param.direction.write(
                    dense_index, make_float4(direction, 0.f));
                $if(param.motion_enabled != 0u) {
                    param.motion.write(dense_index, make_float2(0.f));
                };
            };

            Kernel motion_kernel = [](Var<MotionParam> param) noexcept {
                const Uint2 dispatch = dispatch_idx().xy();
                const Uint index = dispatch_id();
                Float2 value = make_float2(0.f);
                $if(param.history_valid != 0u &&
                    param.validity.read(index) != 0u &&
                    param.motion_validity.read(index) != 0u) {
                    const gpu::LocalPhaseCoordinate phase =
                        gpu::local_phi(param, dispatch);

                    const Float3 origin = param.ray_origin.read(index).xyz();
                    const Float3 direction = normalize(
                        param.ray_direction.read(index).xyz());
                    const Float3 world_position = origin +
                        direction * param.depth.read(index);
                    const Float3 previous_world_position = world_position -
                        param.world_velocity.read(index).xyz();
                    const Float3 previous_local = transform_point(
                        inverse(param.previous_local_to_world),
                        previous_world_position);
                    Float denominator = previous_local.z - phase.u.z;
                    denominator = select(
                        abs(denominator) < 0.001f,
                        select(denominator >= 0.f, 0.001f, -0.001f),
                        denominator);
                    const Float focal_t =
                        (param.focal_distance - phase.u.z) / denominator;
                    const Float3 previous_focal = phase.u +
                        (previous_local - phase.u) * focal_t;
                    const Float pixel_width = param.focal_width /
                        cast<float>(param.output_width);
                    const Float pixel_height = param.focal_height /
                        cast<float>(param.output_height);
                    const Float2 previous_center = make_float2(
                        ((previous_focal.x + param.focal_width * 0.5f) /
                            pixel_width) * 3.f,
                        (param.focal_height * 0.5f - previous_focal.y) /
                            pixel_height);
                    const Float2 current_center = make_float2(
                        cast<float>(dispatch.x) + 0.5f,
                        cast<float>(dispatch.y) + 0.5f);
                    value = current_center - previous_center;
                };
                param.motion.write(index, value);
            };

            Kernel interlace_kernel = [](Var<InterlaceParam> param) noexcept {
                const Uint2 pixel = dispatch_idx().xy();
                const Uint output_index = dispatch_id();
                const Uint base = pixel.y * param.output_width * 3u +
                    pixel.x * 3u;
                const Float red = param.subpixel_radiance.read(base).x;
                const Float green = param.subpixel_radiance.read(base + 1u).y;
                const Float blue = param.subpixel_radiance.read(base + 2u).z;
                const Float3 exposed = 1.f - exp(-make_float3(red, green, blue));
                const Float3 srgb = select(
                    exposed <= make_float3(0.0031308f),
                    exposed * 12.92f,
                    1.055f * pow(exposed, make_float3(1.0f / 2.4f)) - 0.055f);
                param.canonical.write(
                    output_index, make_float4(srgb, 1.f));
            };

            static_ray_table_shader = device->compile(
                static_ray_table_kernel,
                "lfs_lightfield_build_static_ray_table");
            ray_shader = device->compile(
                ray_kernel, "lfs_lightfield_generate_primary_rays");
            motion_shader = device->compile(
                motion_kernel, "lfs_lightfield_motion");
            interlace_shader = device->compile(
                interlace_kernel, "lfs_lightfield_interlace_present");
        } catch (const std::exception &exception) {
            return {backend::ErrorCode::BackendFailure, exception.what()};
        } catch (...) {
            return {backend::ErrorCode::BackendFailure,
                    "Light-field GPU shader compilation failed."};
        }

        telemetry.sample_extent = sample_extent;
        telemetry.output_extent = output_extent;
        telemetry.layout_generation = layout_generation;
        telemetry.layout_hash = layout_hash;
        telemetry.prepared = true;
        telemetry.rays_generated = false;
        telemetry.motion_generated = false;
        telemetry.canonical_present_ready = false;
        telemetry.srgb_transfer_applied = true;
        return {};
    }

    [[nodiscard]] StaticRayTableParam static_ray_table_parameters() const noexcept {
        StaticRayTableParam param{};
        param.view_id = view_id.view().descriptor();
        param.inverse_ray_length = inverse_ray_length.view().descriptor();
        param.view_x_lut = view_x_lut.view().descriptor();
        param.output_width = output_extent.width;
        param.output_height = output_extent.height;
        param.view_count = std::max(lenticular.view_count, 1u);
        param.pitch = static_cast<float>(lenticular.pitch);
        param.slant_tangent = static_cast<float>(std::tan(lenticular.slant));
        param.offset = static_cast<float>(lenticular.offset);
        param.focal_distance = static_cast<float>(geometry.focal_distance);
        param.focal_width = static_cast<float>(geometry.focal_width);
        param.focal_height = static_cast<float>(geometry.focal_height);
        return param;
    }

    [[nodiscard]] RayGenerationParam ray_parameters() const noexcept {
        RayGenerationParam param{};
        param.origin = ray_origin[write_bank].view().descriptor();
        param.direction = ray_direction[write_bank].view().descriptor();
        param.motion = motion.view().descriptor();
        param.view_id = view_id.view().descriptor();
        param.inverse_ray_length = inverse_ray_length.view().descriptor();
        param.view_x_lut = view_x_lut.view().descriptor();
        param.focal_x_lut = focal_x_lut.view().descriptor();
        param.focal_y_lut = focal_y_lut.view().descriptor();
        param.local_to_world = current_local_to_world;
        param.output_width = output_extent.width;
        param.output_height = output_extent.height;
        param.focal_distance = static_cast<float>(geometry.focal_distance);
        param.focal_width = static_cast<float>(geometry.focal_width);
        param.focal_height = static_cast<float>(geometry.focal_height);
        param.motion_enabled = motion_required ? 1u : 0u;
        return param;
    }

};

LightFieldProcessor::LightFieldProcessor(
    LenticularParams lenticular,
    LightFieldGeometryParams geometry,
    backend::FrameProcessor *denoiser)
    : impl_(std::make_unique<Impl>(
          std::move(lenticular), std::move(geometry), denoiser)) {}

LightFieldProcessor::LightFieldProcessor(LightFieldProcessor &&) noexcept = default;
LightFieldProcessor &LightFieldProcessor::operator=(LightFieldProcessor &&) noexcept = default;
LightFieldProcessor::~LightFieldProcessor() noexcept = default;

void LightFieldProcessor::set_camera(
    const backend::CameraState &camera) noexcept {
    impl_->current_local_to_world = camera_to_world(camera);
    impl_->current_phase_transform = camera_to_phase_transform(camera);
}

backend::Error LightFieldProcessor::reset(
    backend::ResetReason reason) noexcept {
    if (reason == backend::ResetReason::None) {
        return {backend::ErrorCode::InvalidArgument,
                "Light-field reset reason must not be None."};
    }
    impl_->invalidate_history();

    ++impl_->telemetry.reset_count;
    impl_->telemetry.history_valid_before = false;
    impl_->telemetry.rays_generated = false;
    impl_->telemetry.motion_generated = false;
    impl_->telemetry.canonical_present_ready = false;
    return impl_->denoiser ? impl_->denoiser->reset(reason) : backend::Error{};
}

backend::Error LightFieldProcessor::prepare(
    const backend::FrameContext &context) noexcept {
    if (impl_->denoiser) {
        backend::Error error = impl_->denoiser->prepare(context);
        if (error) {
            return error;
        }
        backend::Result<backend::ResourceRequirementSet> child =
            impl_->denoiser->resource_requirements(context);
        if (!child.ok()) {
            return child.error();
        }
        impl_->child_requirements = std::move(child).value();
    } else {
        impl_->child_requirements = {};
    }
    if(auto error=impl_->allocate_and_compile(context))return error;
    // Give spatial consumers a cold preparation point before any queued trace.
    return impl_->bind_phase_state(context);
}

backend::Result<backend::ResourceRequirementSet>
LightFieldProcessor::resource_requirements(
    const backend::FrameContext &context) noexcept {
    backend::ResourceRequirementSet upstream;
    if (!impl_->denoiser) {
        backend::Result<void> added = upstream.add({
            backend::Semantic::Radiance,
            backend::TemporalSlot::Current,
            0u,
            backend::ValueType::Float4});
        if (!added.ok()) {
            return backend::Result<backend::ResourceRequirementSet>::failure(
                added.error());
        }
        return backend::Result<backend::ResourceRequirementSet>::success(
            std::move(upstream));
    }

    backend::Result<backend::ResourceRequirementSet> child =
        impl_->denoiser->resource_requirements(context);
    if (!child.ok()) {
        return child;
    }
    const bool requested_motion = child.value().contains(
        backend::Semantic::Motion,
        backend::TemporalSlot::Current,
        0u);
    const bool requested_current_rays =
        child.value().contains(
            backend::Semantic::RayOrigin,
            backend::TemporalSlot::Current) ||
        child.value().contains(
            backend::Semantic::RayDirection,
            backend::TemporalSlot::Current);
    const bool requested_previous_rays =
        child.value().retains_for_previous(backend::Semantic::RayOrigin) ||
        child.value().retains_for_previous(backend::Semantic::RayDirection) ||
        child.value().contains(
            backend::Semantic::RayOrigin,
            backend::TemporalSlot::Previous) ||
        child.value().contains(
            backend::Semantic::RayDirection,
            backend::TemporalSlot::Previous);
    if (requested_motion != impl_->motion_required ||
        requested_current_rays != impl_->current_rays_required ||
        requested_previous_rays != impl_->previous_rays_required) {
        return backend::Result<backend::ResourceRequirementSet>::failure({
            backend::ErrorCode::InvalidState,
            "Light-field owned resource requirements changed after prepare."});
    }
    impl_->child_requirements = child.value();
    for (const backend::ResourceRequirement &requirement :
         impl_->child_requirements.items()) {
        if (requirement.semantic == backend::Semantic::RayOrigin ||
            requirement.semantic == backend::Semantic::RayDirection ||
            (requirement.temporal_slot == backend::TemporalSlot::Current &&
             requirement.semantic == backend::Semantic::Motion)) {
            continue;
        }
        backend::Result<void> added = upstream.add(requirement);
        if (!added.ok()) {
            return backend::Result<backend::ResourceRequirementSet>::failure(
                added.error());
        }
    }
    if (!upstream.contains(backend::Semantic::Radiance)) {
        backend::Result<void> added = upstream.add({
            backend::Semantic::Radiance,
            backend::TemporalSlot::Current,
            0u,
            backend::ValueType::Float4});
        if (!added.ok()) {
            return backend::Result<backend::ResourceRequirementSet>::failure(
                added.error());
        }
    }
    if (impl_->child_requirements.requires_rigid_instance_motion()) {
        upstream.require_rigid_instance_motion();
    }
    if (impl_->child_requirements.contains(backend::Semantic::Motion)) {
        backend::Result<void> added = upstream.add({
            backend::Semantic::Depth,
            backend::TemporalSlot::Current,
            0u,
            backend::ValueType::Float});
        if (added.ok()) {
            added = upstream.add({
                backend::Semantic::Validity,
                backend::TemporalSlot::Current,
                0u,
                backend::ValueType::UInt});
        }
        if (added.ok()) {
            added = upstream.add({
                backend::Semantic::WorldVelocity,
                backend::TemporalSlot::Current,
                0u,
                backend::ValueType::Float4});
        }
        if (added.ok()) {
            added = upstream.add({
                backend::Semantic::MotionValidity,
                backend::TemporalSlot::Current,
                0u,
                backend::ValueType::UInt});
        }
        if (!added.ok()) {
            return backend::Result<backend::ResourceRequirementSet>::failure(
                added.error());
        }
    }
    return backend::Result<backend::ResourceRequirementSet>::success(
        std::move(upstream));
}

backend::Result<backend::PrimaryRayInput> LightFieldProcessor::primary_rays(
    const backend::FrameContext &context,
    CommandBatch &commands) noexcept {
    if (!impl_->telemetry.prepared || context.device != impl_->device ||
        context.sample_extent != impl_->sample_extent ||
        context.output_extent != impl_->output_extent ||
        context.layout_generation != impl_->layout_generation ||
        context.layout_hash != impl_->layout_hash) {
        return backend::Result<backend::PrimaryRayInput>::failure({
            backend::ErrorCode::InvalidState,
            "Light-field primary rays were requested before matching prepare."});
    }
    if (impl_->static_ray_table_pending) {
        commands << impl_->view_x_lut.upload(impl_->view_x_host.data())
                 << impl_->focal_x_lut.upload(impl_->focal_x_host.data())
                 << impl_->focal_y_lut.upload(impl_->focal_y_host.data());
        if (impl_->static_ray_table_pending) {
            commands << impl_->static_ray_table_shader(
                impl_->static_ray_table_parameters()).dispatch(
                static_cast<uint32_t>(impl_->view_id.size()));
            impl_->static_ray_table_pending = false;
            impl_->static_ray_table_generation = context.layout_generation;
            impl_->static_ray_table_hash = context.layout_hash;
        }
    }
    if (impl_->static_ray_table_generation != context.layout_generation ||
        impl_->static_ray_table_hash != context.layout_hash) {
        return backend::Result<backend::PrimaryRayInput>::failure({
            backend::ErrorCode::InvalidState,
            "Light-field StaticRay5 table does not match this layout generation."});
    }

    if (impl_->materialized_ray_vectors_required) {
        commands << impl_->ray_shader(
            impl_->ray_parameters()).dispatch(
            make_uint2(impl_->sample_extent.width, impl_->sample_extent.height));
    }

    impl_->telemetry.rays_generated = true;
    impl_->telemetry.frame_index = context.frame_index;
    // Encoded motion is generated after the primary G-buffer is available.
    const uint32_t submitted_ray_count = static_cast<uint32_t>(
        impl_->sample_extent.area());
    backend::PrimaryRayInput input{};
    input.motion = impl_->motion.view(0u,
        impl_->motion_required ? submitted_ray_count : 1u);
    input.ray_count = submitted_ray_count;
    input.dense_target_count = static_cast<uint32_t>(
        impl_->sample_extent.area());
    if (impl_->materialized_ray_vectors_required) {
        input.origin = impl_->ray_origin[impl_->write_bank].view();
        input.direction = impl_->ray_direction[impl_->write_bank].view();
    } else {
        input.factorized_view_id = impl_->view_id.view();
        input.factorized_inverse_ray_length =
            impl_->inverse_ray_length.view();
        input.factorized_view_x = impl_->view_x_lut.view();
        input.factorized_focal_x = impl_->focal_x_lut.view();
        input.factorized_focal_y = impl_->focal_y_lut.view();
        input.factorized_local_to_world = matrix_to_array(
            impl_->current_local_to_world);
        input.factorized_focal_distance = static_cast<float>(
            impl_->geometry.focal_distance);
        input.factorized_ray_source = true;
    }
    return backend::Result<backend::PrimaryRayInput>::success(std::move(input));
}

backend::Result<backend::ProcessOutput> LightFieldProcessor::process(
    const backend::FrameContext &context,
    const backend::FrameResources &resources,
    CommandBatch &commands) noexcept {
    if (!impl_->telemetry.prepared || !impl_->telemetry.rays_generated ||
        context.device != impl_->device ||
        context.sample_extent != impl_->sample_extent ||
        context.output_extent != impl_->output_extent ||
        context.layout_generation != impl_->layout_generation ||
        context.layout_hash != impl_->layout_hash ||
        impl_->static_ray_table_pending ||
        impl_->static_ray_table_generation != context.layout_generation ||
        impl_->static_ray_table_hash != context.layout_hash) {
        return backend::Result<backend::ProcessOutput>::failure({
            backend::ErrorCode::InvalidState,
            "Light-field process context does not match prepared rays."});
    }
    const auto *radiance = resources.view<float4>(backend::Semantic::Radiance);
    if (!radiance) {
        return backend::Result<backend::ProcessOutput>::failure(
            missing("Light-field processing requires current radiance."));
    }

    backend::FrameResources augmented = resources;
    impl_->frame_completed = false;
    const backend::Error phase_error = impl_->bind_phase_state(context);
    if (phase_error) {
        return backend::Result<backend::ProcessOutput>::failure(phase_error);
    }
    const auto add_owned = [&](backend::ResourceBinding binding)
        -> backend::Result<void> {
        if (augmented.find(
                binding.semantic,
                binding.temporal_slot,
                binding.layer) != nullptr) {
            return backend::Result<void>::success();
        }
        return augmented.add(std::move(binding));
    };
    if (impl_->denoiser &&
        impl_->child_requirements.contains(backend::Semantic::RayOrigin)) {
        backend::Result<void> added = add_owned({
            backend::Semantic::RayOrigin,
            backend::SampleDomain::EncodedSubpixelGrid,
            backend::TemporalSlot::Current,
            0u,
            backend::ValueType::Float4,
            impl_->ray_origin[impl_->write_bank].view()});
        if (!added.ok()) {
            return backend::Result<backend::ProcessOutput>::failure(
                added.error());
        }
    }
    if (impl_->denoiser &&
        impl_->child_requirements.contains(backend::Semantic::RayDirection)) {
        backend::Result<void> added = add_owned({
            backend::Semantic::RayDirection,
            backend::SampleDomain::EncodedSubpixelGrid,
            backend::TemporalSlot::Current,
            0u,
            backend::ValueType::Float4,
            impl_->ray_direction[impl_->write_bank].view()});
        if (!added.ok()) {
            return backend::Result<backend::ProcessOutput>::failure(
                added.error());
        }
    }
    if (impl_->denoiser && impl_->history_valid &&
        impl_->child_requirements.contains(
            backend::Semantic::RayOrigin,
            backend::TemporalSlot::Previous)) {
        backend::Result<void> added = add_owned({
            backend::Semantic::RayOrigin,
            backend::SampleDomain::EncodedSubpixelGrid,
            backend::TemporalSlot::Previous,
            0u,
            backend::ValueType::Float4,
            impl_->ray_origin[impl_->write_bank ^ 1u].view()});
        if (!added.ok()) {
            return backend::Result<backend::ProcessOutput>::failure(
                added.error());
        }
    }
    if (impl_->denoiser && impl_->history_valid &&
        impl_->child_requirements.contains(
            backend::Semantic::RayDirection,
            backend::TemporalSlot::Previous)) {
        backend::Result<void> added = add_owned({
            backend::Semantic::RayDirection,
            backend::SampleDomain::EncodedSubpixelGrid,
            backend::TemporalSlot::Previous,
            0u,
            backend::ValueType::Float4,
            impl_->ray_direction[impl_->write_bank ^ 1u].view()});
        if (!added.ok()) {
            return backend::Result<backend::ProcessOutput>::failure(
                added.error());
        }
    }

    const bool need_motion = impl_->denoiser &&
        impl_->child_requirements.contains(backend::Semantic::Motion);
    if (need_motion) {
        const auto *depth = resources.view<float>(backend::Semantic::Depth);
        const auto *validity = resources.view<uint32_t>(backend::Semantic::Validity);
        const auto *world_velocity = resources.view<float4>(
            backend::Semantic::WorldVelocity);
        const auto *motion_validity = resources.view<uint32_t>(
            backend::Semantic::MotionValidity);
        if (!depth || !validity || !world_velocity || !motion_validity) {
            return backend::Result<backend::ProcessOutput>::failure(
                missing("Light-field motion requires depth, validity, world velocity, and motion validity."));
        }
        MotionParam motion_param{};
        motion_param.depth = depth->descriptor();
        motion_param.validity = validity->descriptor();
        motion_param.ray_origin = impl_->ray_origin[
            impl_->write_bank].view().descriptor();
        motion_param.ray_direction = impl_->ray_direction[
            impl_->write_bank].view().descriptor();
        motion_param.motion = impl_->motion.view().descriptor();
        motion_param.world_velocity = world_velocity->descriptor();
        motion_param.motion_validity = motion_validity->descriptor();
        motion_param.previous_local_to_world = impl_->previous_local_to_world;
        motion_param.history_valid = impl_->history_valid ? 1u : 0u;
        motion_param.output_width = impl_->output_extent.width;
        motion_param.output_height = impl_->output_extent.height;
        motion_param.view_count = std::max(impl_->lenticular.view_count, 1u);
        motion_param.pitch = static_cast<float>(impl_->lenticular.pitch);
        motion_param.slant = static_cast<float>(impl_->lenticular.slant);
        motion_param.offset = static_cast<float>(impl_->lenticular.offset);
        motion_param.focal_distance =
            static_cast<float>(impl_->geometry.focal_distance);
        motion_param.focal_width = static_cast<float>(impl_->geometry.focal_width);
        motion_param.focal_height = static_cast<float>(impl_->geometry.focal_height);
        motion_param.array_angle_degrees =
            static_cast<float>(impl_->geometry.array_angle_degrees);
        commands << impl_->motion_shader(motion_param).dispatch(
            make_uint2(impl_->sample_extent.width, impl_->sample_extent.height));
        backend::Result<void> added = add_owned({
            backend::Semantic::Motion,
            backend::SampleDomain::EncodedSubpixelGrid,
            backend::TemporalSlot::Current,
            0u,
            backend::ValueType::Float2,
            impl_->motion.view()});
        if (!added.ok()) {
            return backend::Result<backend::ProcessOutput>::failure(
                added.error());
        }
    }

    impl_->telemetry.frame_index = context.frame_index;
    impl_->telemetry.history_valid_before = impl_->history_valid;
    impl_->telemetry.motion_generated = need_motion;
    impl_->motion_bound = need_motion;

    if (impl_->denoiser) {
        return impl_->denoiser->process(context, augmented, commands);
    }
    return backend::Result<backend::ProcessOutput>::success({
        *radiance, false});
}

backend::Result<backend::PresentOutput> LightFieldProcessor::present(
    const backend::FrameContext &context,
    const backend::FrameResources &resources,
    const backend::ProcessOutput &output,
    CommandBatch &commands) noexcept {
    if (!output.presentation_only &&
        output.radiance.size() != impl_->sample_extent.area()) {
        return backend::Result<backend::PresentOutput>::failure({
            backend::ErrorCode::InvalidState,
            "Light-field canonical present has no complete subpixel output."});
    }
    if (impl_->denoiser) {
        backend::Result<backend::PresentOutput> child = impl_->denoiser->present(
            context, resources, output, commands);
        if (!child.ok()) {
            return child;
        }
        if (child.value().complete) {
            impl_->telemetry.canonical_present_ready = true;
            impl_->frame_completed = true;
            return child;
        }
    }
    if(impl_->canonical.size()!=impl_->output_extent.area()){
        try {
            impl_->canonical=impl_->device->create_buffer<float4>(impl_->output_extent.area(),"lfs.lightfield.canonical");
            impl_->telemetry.canonical_present_device_bytes=impl_->canonical.size()*sizeof(float4);
            impl_->telemetry.total_owned_device_bytes+=impl_->telemetry.canonical_present_device_bytes;
            ++impl_->telemetry.allocation_count;
        } catch (const std::exception& error) {return backend::Result<backend::PresentOutput>::failure({backend::ErrorCode::OutOfMemory,error.what()});}
    }
    InterlaceParam param{};
    param.subpixel_radiance = output.radiance.descriptor();
    param.canonical = impl_->canonical.view().descriptor();
    param.output_width = impl_->output_extent.width;
    commands << impl_->interlace_shader(param).dispatch(
        make_uint2(impl_->output_extent.width, impl_->output_extent.height));
    impl_->telemetry.canonical_present_ready = true;
    impl_->frame_completed = true;
    return backend::Result<backend::PresentOutput>::success({
        true,
        impl_->denoiser
            ? "lfs_lightfield_denoised_interlace_exposure_linear_srgb_present"
            : "lfs_lightfield_raw_interlace_exposure_linear_srgb_present",
        impl_->canonical.view()});
}

backend::Error LightFieldProcessor::enqueue_commit(
    const backend::FrameContext &context,
    CommandBatch &commands) noexcept {
    if (!impl_->frame_completed) {
        return {backend::ErrorCode::InvalidState,
                "Light-field presentation is not pending commit."};
    }
    if (impl_->denoiser) {
        const backend::Error error =
            impl_->denoiser->enqueue_commit(context, commands);
        if (error) {
            return error;
        }
    }
    return {};
}

backend::Error LightFieldProcessor::validate_submission(
    const backend::FrameContext &context) noexcept {

    return impl_->denoiser
        ? impl_->denoiser->validate_submission(context)
        : backend::Error{};
}

void LightFieldProcessor::finalize_success(
    const backend::FrameContext &context) noexcept {
    if (!impl_->frame_completed) {
        return;
    }
    if (impl_->denoiser) {
        impl_->denoiser->finalize_success(context);
    }
    impl_->previous_local_to_world = impl_->current_local_to_world;
    impl_->previous_phase_transform = impl_->current_phase_transform;
    impl_->history_valid = true;
    if (impl_->previous_rays_required) {
        impl_->write_bank ^= 1u;
    }
    impl_->frame_completed = false;
}

void LightFieldProcessor::abort_submission(
    const backend::FrameContext &context) noexcept {
    if (impl_->denoiser) {
        impl_->denoiser->abort_submission(context);
    }

    impl_->frame_completed = false;
    impl_->telemetry.canonical_present_ready = false;
    // A failed submission may have failed while the queued table build was
    // executing. Re-emit it instead of trusting partially produced data.
    impl_->static_ray_table_pending = true;
    impl_->static_ray_table_generation = 0u;
    impl_->static_ray_table_hash = 0u;
}

const LightFieldTelemetry &LightFieldProcessor::telemetry() const noexcept {
    return impl_->telemetry;
}

} // namespace lfs::lightfield
