#include "ssat/phase2/encoded_filter_core.h"

#include "ssat/phase2/native/encoded_atrous_native_runtime.h"

#include "dsl/dsl.h"
#include "math/base.h"
#include "math/basic_types.h"
#include "rhi/resources/shader.h"

#include "ssat/runtime_paths.h"
#include "ssat/phase2/calibrated_view_chart.h"
#include "ssat/phase2/resident_view_chart.h"
#include "ssat/phase2/resident_aperture_cells.h"
#include "ssat/resident_storage.h"
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lfs::denoise {
using namespace ocarina;

struct EncodedPrefilterParam {
    BufferDesc<half4> source;

    BufferDesc<half4> destination;

    BufferDesc<float> depth;
    BufferDesc<float4> normal;
    BufferDesc<float4> albedo;
    BufferDesc<float4> ray_origin;
    BufferDesc<float4> ray_direction;
    BufferDesc<uint4> packed_observation;
    BufferDesc<uint> static_view_id;
    BufferDesc<float> static_inverse_ray_length;
    BufferDesc<float> static_view_x;
    BufferDesc<float> static_focal_x;
    BufferDesc<float> static_focal_y;
    BufferDesc<uint> validity;
    BufferDesc<uint> instance_id;

    float4 world_to_local_row0{};
    float4 world_to_local_row1{};
    float4 world_to_local_row2{};
    float focal_distance{};
};

struct EncodedAtrousParam {
    BufferDesc<half4> source;
    BufferDesc<half4> destination;

    BufferDesc<float> depth;
    BufferDesc<float4> normal;
    BufferDesc<float4> albedo;
    BufferDesc<float4> ray_origin;
    BufferDesc<float4> ray_direction;
    BufferDesc<uint4> packed_observation;
    BufferDesc<uint> static_view_id;
    BufferDesc<float> static_inverse_ray_length;
    BufferDesc<float> static_view_x;
    BufferDesc<float> static_focal_x;
    BufferDesc<float> static_focal_y;
    BufferDesc<uint> validity;
    BufferDesc<uint> instance_id;

    uint step_width{};

    float sigma_luminance{};
    float sigma_normal{};
    float sigma_depth{};
    float sigma_albedo{};
    float4 world_to_local_row0{};
    float4 world_to_local_row1{};
    float4 world_to_local_row2{};
    float focal_distance{};
};

struct EncodedFinalParam {
    BufferDesc<half4> source;

    BufferDesc<half4> temporal;
    BufferDesc<float4> albedo;
    BufferDesc<uint> validity;
    BufferDesc<uint4> packed_observation;

    BufferDesc<half4> resolved_signal;

    float energy_compensation{};

    BufferDesc<float> static_inverse_ray_length;
    BufferDesc<float> static_view_x;
    BufferDesc<float> static_focal_x;
    BufferDesc<float> static_focal_y;
    BufferDesc<uint> static_view_id;
    float4 world_to_local_row0{};
    float4 world_to_local_row1{};
    float4 world_to_local_row2{};
    float focal_distance{};    BufferDesc<uint> view_chart_ids,view_chart_offsets,frame_status;
    uint32_t view_chart_templates{}, template_id_bits{};

};

} // namespace lfs::denoise

OC_PARAM_STRUCT(lfs::denoise, EncodedPrefilterParam,
    source, destination, depth, normal, albedo, ray_origin, ray_direction, packed_observation, static_view_id, static_inverse_ray_length, static_view_x, static_focal_x, static_focal_y, validity, instance_id, world_to_local_row0, world_to_local_row1, world_to_local_row2, focal_distance) {};

OC_PARAM_STRUCT(lfs::denoise, EncodedAtrousParam,
    source, destination, depth, normal, albedo, ray_origin, ray_direction, packed_observation, static_view_id, static_inverse_ray_length, static_view_x, static_focal_x, static_focal_y, validity, instance_id, step_width, sigma_luminance, sigma_normal, sigma_depth, sigma_albedo, world_to_local_row0, world_to_local_row1, world_to_local_row2, focal_distance) {};

OC_PARAM_STRUCT(lfs::denoise, EncodedFinalParam,
    source, temporal, albedo, validity, packed_observation, resolved_signal, energy_compensation, static_inverse_ray_length, static_view_x, static_focal_x, static_focal_y, static_view_id, world_to_local_row0, world_to_local_row1, world_to_local_row2, focal_distance, view_chart_ids, view_chart_offsets, frame_status, view_chart_templates, template_id_bits) {};

namespace lfs::denoise {
namespace {

[[nodiscard]] Float luminance(const Float3 &value) noexcept {
    return dot(value, make_float3(0.2126f, 0.7152f, 0.0722f));
}

[[nodiscard]] Float3 safe_albedo(const Float3 &value) noexcept {
    return sqrt(value * value + make_float3(0.01f));
}

struct PackedSpatialGuide {
    Float hit_distance;
    Float3 normal;
    Float3 albedo;
    Uint validity;
    Uint instance_id;
    Uint material_id;
    Uint domain;
    Uint surface_flags;
    Float roughness;
};

[[nodiscard]] Float3 unpack_oct16_normal(Uint packed) noexcept {
    const Float x = cast<float>(packed & 0xffffu) *
        (2.0f / 65535.0f) - 1.0f;
    const Float y = cast<float>((packed >> 16u) & 0xffffu) *
        (2.0f / 65535.0f) - 1.0f;
    const Float z = 1.0f - abs(x) - abs(y);
    const Float fold = clamp(-z, 0.0f, 1.0f);
    return normalize(make_float3(
        x + select(x >= 0.0f, -fold, fold),
        y + select(y >= 0.0f, -fold, fold), z));
}

[[nodiscard]] Float2 unpack_half2(Uint bits) noexcept {
    return cast<float2>(as<half2>(bits));
}

template<typename Param>
[[nodiscard]] PackedSpatialGuide packed_spatial_guide(
    const Var<Param> &param,
    Uint index) noexcept {
    const Uint4 first = param.packed_observation.read(index * 2u);
    const Uint4 second = param.packed_observation.read(index * 2u + 1u);
    const Uint domain = second.y & backend::PackedSurfaceDomainMask;
    const Bool surface = domain ==
        static_cast<uint>(backend::PackedSurfaceDomain::Surface);
    const Float2 albedo_xy = unpack_half2(second.z);
    const Float2 albedo_z_diffuse = unpack_half2(second.w);
    PackedSpatialGuide guide{};
    guide.hit_distance = as<float>(first.z);
    guide.normal = select(
        surface, unpack_oct16_normal(first.w), make_float3(0.0f, 0.0f, 1.0f));
    guide.albedo = make_float3(
        albedo_xy.x, albedo_xy.y, albedo_z_diffuse.x);
    guide.validity = select(surface, 1u, 0u);
    guide.instance_id = select(surface, first.y, ~0u);
    guide.material_id = select(surface, first.x, ~0u);
    guide.domain = domain;
    guide.surface_flags =
        (second.y >> backend::PackedSurfaceFlagsShift) &
        backend::PackedSurfaceFlagsMask;
    guide.roughness = cast<float>(
        (second.y >> backend::PackedSurfaceRoughnessShift) & 0xffu) *
        (1.0f / 255.0f);
    return guide;
}

template<typename Param>
[[nodiscard]] Float3 local_normal(
    const Var<Param> &param,
    const Float3 &world_normal) noexcept {
    return normalize(make_float3(
        dot(world_normal, param.world_to_local_row0.xyz()),
        dot(world_normal, param.world_to_local_row1.xyz()),
        dot(world_normal, param.world_to_local_row2.xyz())));
}

template<typename Param>
[[nodiscard]] Float3 factorized_position(
    const Var<Param> &param,
    Uint index,
    const Int2 &pixel,
    Float hit_distance) noexcept {
    const Uint packed = param.static_view_id.read(index >> 2u);
    const Uint view = (packed >> ((index & 3u) * 8u)) & 0xffu;
    const Float view_x = param.static_view_x.read(view);
    const Float inverse = param.static_inverse_ray_length.read(index);
    const Float3 direction = make_float3(
        (param.static_focal_x.read(cast<uint>(pixel.x)) - view_x) * inverse,
        param.static_focal_y.read(cast<uint>(pixel.y)) * inverse,
        param.focal_distance * inverse);
    return make_float3(view_x, 0.0f, 0.0f) + direction * hit_distance;
}

template<typename Param>
[[nodiscard]] Float3 spatial_normal(
    const Var<Param> &param,
    Uint index,
    bool factorized) noexcept {
    const Float3 world = param.normal.read(index).xyz();
    if (factorized) return local_normal(param, world);
    return normalize(world);
}

template<typename Param>
[[nodiscard]] Float3 spatial_position(
    const Var<Param> &param,
    Uint index,
    const Int2 &pixel,
    Float hit_distance,
    bool factorized) noexcept {
    if (factorized) {
        return factorized_position(param, index, pixel, hit_distance);
    }
    return param.ray_origin.read(index).xyz() + hit_distance *
        normalize(param.ray_direction.read(index).xyz());
}

struct StencilTap {
    Int2 pixel{};
    Bool valid{false};
};

[[nodiscard]] backend::Error resource_error(std::string message) {
    return {backend::ErrorCode::ResourceMissing, std::move(message)};
}

template<typename T>
[[nodiscard]] const BufferView<T> *required_view(
    const backend::FrameResources &resources,
    backend::Semantic semantic,
    backend::TemporalSlot slot = backend::TemporalSlot::Current,
    uint32_t layer = 0u) noexcept {
    return resources.view<T>(semantic, slot, layer);
}

template<typename T>
[[nodiscard]] uint64_t native_address(const BufferView<T> &view) noexcept {
    return static_cast<uint64_t>(view.handle()) +
        static_cast<uint64_t>(view.offset_in_byte());
}

} // namespace

struct EncodedFilterCore::Impl {
    InterstageProcessor interstage_processor{};
    EncodedConfig config{};
    EncodedTelemetry telemetry{};
    Device *device{};
    Stream *device_stream{};
    backend::Extent2D extent{};
    backend::SampleDomain domain{backend::SampleDomain::ImagePixelGrid};
    uint64_t layout_generation{};
    uint64_t layout_hash{};
    bool history_valid{};
    bool history_pending_commit{};
    uint32_t history_write_index{};
    BufferView<half4> native_demodulated_current{};
    BufferView<uint> native_current_frame_status{};
    BufferView<half4> demodulated_output{};
    Buffer<float4> canonical_output{};
    uint64_t native_current_frame_index{};
    uint64_t native_current_layout_generation{};
    uint64_t native_current_layout_hash{};
    lightfield::PhaseFrameState phase_state{};
    bool motion_committed{}, motion_current{};
    using HeldRawStorage = encoded_atrous::native::HeldRawStorage;
    HeldRawStorage held_raw_committed{HeldRawStorage::Empty}, held_raw_pending{HeldRawStorage::Empty};

    bool phase_state_bound{};

    std::array<Buffer<half4>, 2u> history_radiance{};
    std::array<Buffer<half4>, 2u> spatial{};
    bool resident_scratch{};
    std::array<BufferView<half4>,2> borrowed_spatial{};
    BufferView<float4> borrowed_canonical{};
    BufferView<std::uint16_t> angular_residuals{}, angular_mirror{};
    bool angular_history_valid{}, angular_commit_pending{};
    bool native_trace_scratch{};
    [[nodiscard]] bool stationary_path() const noexcept { return !resident_scratch && !motion_current && phase_state.repeated_camera(); }
    [[nodiscard]] BufferView<half4> spatial_bank(std::size_t bank) const noexcept {
        return resident_scratch?borrowed_spatial[bank]:spatial[bank].view();
    }
    [[nodiscard]] BufferView<float4> canonical_view() const noexcept {
        return resident_scratch?borrowed_canonical:canonical_output.view();
    }
    Buffer<uint> spatial_native_status{};
    Buffer<half4> hold_state{};
    Buffer<uint> view_chart_ids{},view_chart_offsets{},stationary_atlas{},registered_cells{};
    uint32_t template_id_bits{32}, aperture_cell_bits{}, aperture_cell_count{};
    uint32_t atlas_side{};
    Buffer<uint> motion_cell_sites{};
    Buffer<uint4> motion_cell_values{};
    Buffer<uint4> stationary_weights{}, stationary_moments{};
    Buffer<uint> weights_refresh{}, cell_coefficients{};
    bool stationary_cache_valid{};
    float atlas_inverse_dx{}, atlas_inverse_span{};

    Buffer<uint> view_chart_centers{},view_chart_blocks{},view_chart_sources{};
    Buffer<uint> view_chart_slots{};
    uint32_t view_chart_block_count{},view_chart_source_count{};
    uint32_t view_chart_templates{},view_chart_views{},view_chart_count{};
    float4 view_chart_focal_bounds{};

    encoded_atrous::native::EncodedAtrousNativeRuntime
        spatial_native_runtime{};
    bool spatial_native_submission{};

    Shader<void(EncodedAtrousParam)> atrous_shader{};
    Shader<void(EncodedFinalParam)> final_shader{};

    Shader<void(Buffer<uint>)> clear_spatial_native_status_shader{};

    explicit Impl(EncodedConfig input_config) : config(input_config) {

        config.atrous_iterations = std::clamp(config.atrous_iterations, 1u, 5u);
        config.sigma_luminance = std::max(config.sigma_luminance, 0.01f);
        config.sigma_normal = std::max(config.sigma_normal, 1.0f);
        config.sigma_depth = std::max(config.sigma_depth, 0.01f);
        config.sigma_albedo = std::max(config.sigma_albedo, 0.0f);
        config.energy_compensation = std::clamp(
            config.energy_compensation, 0.9f, 1.1f);
        telemetry.atrous_iterations = config.atrous_iterations;
        telemetry.internal_radiance_fp16 = true;
    }

    void invalidate_history() noexcept {
        motion_committed=motion_current=false;
        held_raw_committed=held_raw_pending=HeldRawStorage::Empty;
        stationary_cache_valid = false;
        angular_history_valid = false;
        angular_commit_pending = false;
        angular_residuals = {};
        angular_mirror = {};
        history_valid = false;
        history_pending_commit = false;
        history_write_index = 0u;
    }

    void invalidate_native_current() noexcept {
        native_demodulated_current = {};
        native_current_frame_status = {};
        native_current_frame_index = 0u;
        native_current_layout_generation = 0u;
        native_current_layout_hash = 0u;
    }

    [[nodiscard]] bool native_tiled_atrous() const noexcept { return true; }

    bool prepare_view_chart(const lightfield::PhaseFrameState &state) noexcept {
        if(view_chart_count==extent.area()) {
            if(!resident_scratch || config.atrous_iterations!=3u || !state.previous_valid ||
               state.repeated_camera() || registered_cells.size())return true;
            try {
                const auto lattice=state.calibrated_lattice();
                const auto lookup=resident_chart::aperture_cells(lattice->view,lattice->width,lattice->height,lattice->views);
                registered_cells=device->create_buffer<uint>(lookup.words.size(),"lfs.registered_compact_view_cells");
                *device_stream << registered_cells.upload(lookup.words.data()) << synchronize() << commit();
                atlas_side=lookup.side;aperture_cell_bits=lookup.bits;aperture_cell_count=lookup.count;
                atlas_inverse_dx=float(lattice->width-1u)/(lattice->x.back()-lattice->x.front());
                atlas_inverse_span=float(57.29577951308232/std::max(state.current.array_angle_degrees,1.e-6));
                telemetry.internal_history_bytes+=registered_cells.size()*sizeof(uint);++telemetry.allocation_count;
                return true;
            }catch(const std::exception& e) {
                registered_cells={};
                std::fprintf(stderr,"Moving aperture preparation failed: %s\n",e.what());return false;
            }
        }
        try{
            if(!state.calibrated_lattice||!device_stream)throw std::runtime_error("Missing calibrated lattice provider or preparation stream.");
            const auto lattice=state.calibrated_lattice();
            if(!lattice||lattice->width!=extent.width||lattice->height!=extent.height||lattice->views!=state.static_view_count)
                throw std::runtime_error("Calibrated lattice does not match this layout.");
            const bool direct=resident::dedicated(device);
            const unsigned footprint=config.atrous_iterations==2u?9u:25u;
            const auto chart=direct?resident_chart::cached(*lattice,footprint):view_chart::cached(*lattice,footprint);
            auto packed=direct?resident_chart::pack_ids(chart->ids,chart->offsets.size()/34u):resident_chart::TemplateIds{};
            template_id_bits=packed.bits;
            const auto& addresses=packed.bits==16u?packed.words:chart->ids;
            view_chart_ids=device->create_buffer<uint>(addresses.size(),"lfs.view_chart.ids");
            if(direct) {
                view_chart_offsets=device->create_buffer<uint>(chart->offsets.size(),"lfs.view_chart.offsets");
                *device_stream << view_chart_ids.upload(addresses.data()) << view_chart_offsets.upload(chart->offsets.data()) << synchronize() << commit();
            } else {
            auto offsets=chart->offsets;
            if(config.atrous_iterations==3u) {
                // The first nine cached Q25 slots are in ascending native
                // address order. Append that order to each original 34-site
                // template, leaving all existing consumers and offsets intact.
                const size_t templates=chart->offsets.size()/34u;
                offsets.reserve(chart->offsets.size()+9u*templates);
                for(size_t t=0;t<templates;++t) {
                    std::array<uint32_t,9> inner;
                    std::copy_n(chart->offsets.data()+34u*t+25u,9u,inner.begin());
                    const auto address=[&](uint32_t bits) {
                        return int32_t(int16_t(bits))+int32_t(int16_t(bits>>16u))*int32_t(lattice->width);
                    };
                    std::sort(inner.begin(),inner.end(),[&](uint32_t a,uint32_t b){return address(a)<address(b);});
                    offsets.insert(offsets.end(),inner.begin(),inner.end());
                }
            }
            view_chart_offsets=device->create_buffer<uint>(offsets.size(),"lfs.view_chart.offsets");
            view_chart_centers=device->create_buffer<uint>(chart->centers.size(),"lfs.view_chart.centers");
            view_chart_blocks=device->create_buffer<uint>(chart->blocks.size()*8u,"lfs.view_chart.blocks");
            view_chart_sources=device->create_buffer<uint>(chart->sources.size(),"lfs.view_chart.sources");
            view_chart_slots=device->create_buffer<uint>(chart->slots.size(),"lfs.view_chart.direct_slots");
            // Store the nearest actual calibrated ray in each focal-plane cell of each view.
            // Empty cells remain invalid and are skipped during the projected gather.
            atlas_side = uint32_t(std::ceil(std::sqrt(float(lattice->views) / 3.f)));
            // One temporary Half4 per cell fits the existing full-width scratch.
            auto cell_count = [&](uint32_t side) {
                return size_t((lattice->width+3u*side-1u)/(3u*side)) *
                       ((lattice->height+side-1u)/side) * lattice->views;
            };
            while (cell_count(atlas_side)>extent.area()) ++atlas_side;
            const uint32_t cell_width = 3u * atlas_side;
            const uint32_t cols = (lattice->width + cell_width - 1u) / cell_width;
            const uint32_t rows = (lattice->height + atlas_side - 1u) / atlas_side;
            std::vector<uint32_t> representatives(size_t(cols) * rows * lattice->views, UINT32_MAX);
            std::vector<uint32_t> nearest(representatives.size(), UINT32_MAX);
            const uint32_t coarse=2u*atlas_side, coarse_width=3u*coarse;
            const uint32_t coarse_cols=(lattice->width+coarse_width-1)/coarse_width;
            const uint32_t coarse_rows=(lattice->height+coarse-1)/coarse;
            std::vector<uint32_t> sites(size_t(coarse_cols)*coarse_rows*lattice->views,UINT32_MAX);
            std::vector<uint32_t> distance_to_center(sites.size(),UINT32_MAX);
            for (uint32_t y=0; y<lattice->height; ++y) {
                for (uint32_t x=0; x<lattice->width; ++x) {
                    const uint32_t i=y*lattice->width+x, v=lattice->view[i];
                    const uint32_t tx=x/cell_width, ty=y/atlas_side;
                    const size_t cell=(size_t(v)*rows+ty)*cols+tx;
                    const int dx=int(2u*(x%cell_width)+1u)-int(cell_width);
                    const int dy=int(2u*(y%atlas_side)+1u)-int(atlas_side);
                    const uint32_t distance=uint32_t(dx*dx+9*dy*dy);
                    if (distance<nearest[cell]) {nearest[cell]=distance;representatives[cell]=i;}
                    const size_t bucket=(size_t(y/coarse)*coarse_cols+x/coarse_width)*lattice->views+v;
                    const int rx=int(2*(x%coarse_width)+1)-int(coarse_width);
                    const int ry=int(2*(y%coarse)+1)-int(coarse);
                    const uint32_t r2=uint32_t(rx*rx+9*ry*ry);
                    if(r2<distance_to_center[bucket]) {distance_to_center[bucket]=r2;sites[bucket]=i;}
                }
            }
            atlas_inverse_dx=float(lattice->width-1u)/(lattice->x.back()-lattice->x.front());
            atlas_inverse_span=float(57.29577951308232/std::max(state.current.array_angle_degrees,1.e-6));
            {
                const size_t words_per_ray=config.atrous_iterations==2u?4u:7u;
                stationary_weights=device->create_buffer<uint4>((extent.area()*words_per_ray+3u)/4u,"lfs.stationary_weights");
                weights_refresh=device->create_buffer<uint>(1,"lfs.stationary_weights_refresh");
                telemetry.internal_history_bytes+=stationary_weights.size()*sizeof(uint4)+sizeof(uint);
                telemetry.allocation_count+=2;
            }
            if(config.atrous_iterations==3u) {
                stationary_moments=device->create_buffer<uint4>(representatives.size(),"lfs.stationary_radiance_moments");
                telemetry.internal_history_bytes+=stationary_moments.size()*sizeof(uint4);
                cell_coefficients=device->create_buffer<uint>(7ull*representatives.size(),"lfs.stationary_cell_coefficients");
                telemetry.internal_history_bytes+=cell_coefficients.size()*sizeof(uint32_t);
                telemetry.allocation_count+=2u;
            }
            stationary_atlas=device->create_buffer<uint>(representatives.size(),"lfs.stationary_view_atlas");
            motion_cell_sites=device->create_buffer<uint>(sites.size(),"lfs.aperture_field_sites");
            const size_t diffuse_cells=2ull*cols*rows;
            motion_cell_values=device->create_buffer<uint4>(2ull*(sites.size()+diffuse_cells),"lfs.aperture_field_values");
            telemetry.internal_history_bytes+=motion_cell_sites.size()*sizeof(uint)+motion_cell_values.size()*sizeof(uint4);
            telemetry.allocation_count+=2;
            *device_stream << motion_cell_sites.upload(sites.data()) << stationary_atlas.upload(representatives.data()) << view_chart_ids.upload(chart->ids.data()) << view_chart_offsets.upload(offsets.data())
                << view_chart_centers.upload(chart->centers.data()) << view_chart_blocks.upload(reinterpret_cast<const uint*>(chart->blocks.data()))
                << view_chart_sources.upload(chart->sources.data()) << view_chart_slots.upload(chart->slots.data()) << synchronize() << commit();
            }
            view_chart_count=uint32_t(chart->ids.size());view_chart_templates=uint32_t(chart->offsets.size()/34u);view_chart_views=chart->views;
            view_chart_block_count=direct?(view_chart_count+255u)/256u:uint32_t(chart->blocks.size());
            view_chart_source_count=uint32_t(chart->sources.size());
            view_chart_focal_bounds={chart->bounds[0],chart->bounds[1],chart->bounds[2],chart->bounds[3]};
            telemetry.internal_history_bytes+=uint64_t(view_chart_ids.size()+view_chart_offsets.size()+chart->centers.size()+chart->blocks.size()*8u+chart->sources.size()+chart->slots.size())*4u;
            telemetry.internal_history_bytes+=stationary_atlas.size()*sizeof(uint32_t);
            telemetry.allocation_count+=direct?2u:7u;
            return true;
        }catch(const std::exception &e){std::fprintf(stderr,"Calibrated LF reconstruction preparation failed: %s\n",e.what());return false;}
        catch(...){std::fputs("Calibrated LF reconstruction preparation failed.\n",stderr);return false;}
    }

    [[nodiscard]] bool phase_state_matches(
        const backend::FrameContext &context) const noexcept {
        return phase_state_bound && phase_state.valid() &&
            phase_state.frame_index == context.frame_index &&
            phase_state.layout_generation == context.layout_generation &&
            phase_state.layout_hash == context.layout_hash &&
            phase_state.static_ray_count == context.sample_extent.area() &&
            phase_state.static_focal_x_count == context.sample_extent.width &&
            phase_state.static_focal_y_count == context.sample_extent.height;
    }

    [[nodiscard]] backend::Error allocate_and_compile(
        const backend::FrameContext &context) noexcept {
        if (context.device == nullptr || !context.sample_extent.valid()) {
            return {backend::ErrorCode::InvalidState,
                    "encoded filter requires a valid device and sample extent."};
        }
        const bool layout_changed =
            device != context.device ||
            extent != context.sample_extent ||
            domain != context.sample_domain ||
            layout_generation != context.layout_generation ||
            layout_hash != context.layout_hash;
        if (!layout_changed && telemetry.prepared) {
            return {};
        }

        device = context.device;
        device_stream = context.stream;
        extent = context.sample_extent;
        domain = context.sample_domain;
        layout_generation = context.layout_generation;
        layout_hash = context.layout_hash;
        invalidate_history();
        invalidate_native_current();
        phase_state_bound = false;
        demodulated_output = {};

        spatial_native_runtime.clear_submission();
        spatial_native_submission = false;

        try {
            const size_t count = static_cast<size_t>(extent.area());
            hold_state={};view_chart_ids={};view_chart_offsets={};view_chart_centers={};view_chart_blocks={};view_chart_sources={};view_chart_slots={};
            view_chart_block_count=view_chart_source_count=view_chart_templates=view_chart_count=view_chart_views=0u;
            registered_cells={};aperture_cell_count=aperture_cell_bits=0u;template_id_bits=32u;
            stationary_atlas={};stationary_weights={};stationary_moments={};
            weights_refresh={};cell_coefficients={};stationary_cache_valid=false;


            const bool fused_output=(config.atrous_iterations==3u&&config.responsive_angular_radiometry)||(config.atrous_iterations==2u&&!config.responsive_angular_radiometry);
            resident_scratch=resident::dedicated(device);
            if(fused_output && !resident_scratch)canonical_output=device->create_buffer<float4>(count/3u,"lfs.ssat.canonical");else canonical_output={};
            const size_t history_bank_count = config.atrous_iterations == 2u || fused_output ? 0u : 1u;
            for (size_t index = 0u; index < history_bank_count; ++index) {
                history_radiance[index] = device->create_buffer<half4>(
                    count, "lfs.encoded.history_radiance." + std::to_string(index));
            }
            {
                for (size_t index = 0u; index < spatial.size(); ++index) {
                    if(resident_scratch){spatial[index]={};continue;}
                    spatial[index] = device->create_buffer<half4>(
                        count, "lfs.encoded.spatial." + std::to_string(index));
                }
                {
                    const bool quality_coverage=config.atrous_iterations==3u && config.responsive_angular_radiometry;
                    const bool plain_coverage=!resident_scratch && config.atrous_iterations==2u && !config.responsive_angular_radiometry;
                    spatial_native_status = device->create_buffer<uint>(
                        1u+((quality_coverage || plain_coverage)?reconstruction_coverage_words(extent.width,extent.height):0u), "lfs.encoded.status_and_reconstruction_coverage");
                }
            }

        } catch (const std::exception &exception) {
            return {backend::ErrorCode::OutOfMemory, exception.what()};
        } catch (...) {
            return {backend::ErrorCode::OutOfMemory,
                    "encoded filter failed to allocate its GPU history buffers."};
        }

        try {
            const bool packed_spatial_geometry = true;
            const bool plain_current = !config.responsive_angular_radiometry &&
                    ((!config.angular_compatibility_spatial &&
                      !config.angular_compatibility_aperture_only &&
                      !config.responsive_radiometry_glossy_only) ||
                     (config.angular_compatibility_spatial &&
                      config.angular_compatibility_aperture_only &&
                      config.responsive_radiometry_glossy_only));

            {
                if (const backend::Error initialize_error =
                        spatial_native_runtime.initialize(*device,
                            plain_current
                                ? encoded_atrous::native::ExecutionProfile::PlainCurrent
                                : (config.responsive_angular_radiometry &&
                                   config.angular_compatibility_spatial && config.angular_compatibility_aperture_only &&
                                   config.responsive_radiometry_glossy_only
                                    ? encoded_atrous::native::ExecutionProfile::GlossyMeasurement
                                    : encoded_atrous::native::ExecutionProfile::General))) {
                    return initialize_error;
                }
                Kernel clear_spatial_native_status_kernel = [](
                    BufferVar<uint> status) noexcept {
                    status.write(dispatch_id(), 0u);
                };
                clear_spatial_native_status_shader = device->compile(
                    clear_spatial_native_status_kernel,
                    "lfs_encoded_spatial_native_status_clear");
            }

            auto atrous_body = [
                                    packed_spatial_geometry,
                                    statistical_reconstruction = true,
                                    combine_exponents =
                                        true ](
                Var<EncodedAtrousParam> &param) noexcept {
                const Int2 size = make_int2(dispatch_dim().xy());
                const Int2 pixel = make_int2(dispatch_idx().xy());
                const Uint index = dispatch_id();

                Float4 center = cast<float4>(param.source.read(index));
                const Bool center_known = center.w >= 0.0f;
                Bool variance_known = center_known;
                if (statistical_reconstruction) center.w = abs(center.w);

                const PackedSpatialGuide center_packed =
                    packed_spatial_geometry
                    ? packed_spatial_guide(param, index)
                    : PackedSpatialGuide{};
                const Uint center_valid = packed_spatial_geometry
                    ? center_packed.validity : param.validity.read(index);
                const Uint center_instance = packed_spatial_geometry
                    ? center_packed.instance_id : param.instance_id.read(index);
                const Uint center_material = packed_spatial_geometry
                    ? center_packed.material_id : ~0u;
                const Float center_depth = packed_spatial_geometry
                    ? center_packed.hit_distance : param.depth.read(index);
                const Float3 center_normal = packed_spatial_geometry
                    ? local_normal(param, center_packed.normal)
                    : spatial_normal(param, index, false);
                const Float3 center_albedo = packed_spatial_geometry
                    ? center_packed.albedo : param.albedo.read(index).xyz();
                const Float3 center_position = spatial_position(
                    param, index, pixel, center_depth,
                    packed_spatial_geometry);
                const Float center_luminance = luminance(center.xyz());
                const Uint destination_carrier=index%3u;
                const Float center_component=select(destination_carrier==0u,center.x,select(destination_carrier==1u,center.y,center.z));
                const Float center_component_scale=center_component/max(center_luminance,1.e-20f);
                const Float phi_luminance =
                    param.sigma_luminance * sqrt(max(center.w, 1.0e-4f)) +
                    1.0e-3f;
                const Bool center_surface = center_packed.domain ==
                        static_cast<uint>(backend::PackedSurfaceDomain::Surface) &&
                    center_packed.validity != 0u;

                Uint phase_class = 0u;

                Float3 color_sum = make_float3(0.0f);
                Float variance_sum = 0.0f;
                Float weight_sum = 0.0f;
                for (int y = -2; y <= 2; ++y) {
                    for (int x = -2; x <= 2; ++x) {
                        const Int2 square_pixel = pixel +
                            make_int2(x, y) * cast<int>(param.step_width);
                        StencilTap mapped{};
                        {
                            mapped.pixel = square_pixel;
                            mapped.valid = all(square_pixel >= make_int2(0)) &&
                                all(square_pixel < size);
                        }
                        const Int2 sample_pixel = mapped.pixel;
                        $if(mapped.valid) {
                            const Uint sample_index =
                                cast<uint>(sample_pixel.y) * cast<uint>(size.x) +
                                cast<uint>(sample_pixel.x);
                            Float4 sample = cast<float4>(param.source.read(sample_index));
                            const Bool sample_known = sample.w >= 0.0f;
                            if (statistical_reconstruction) sample.w = abs(sample.w);

                            const PackedSpatialGuide sample_packed =
                                packed_spatial_geometry
                                ? packed_spatial_guide(param, sample_index)
                                : PackedSpatialGuide{};
                            const Uint sample_valid = packed_spatial_geometry
                                ? sample_packed.validity
                                : param.validity.read(sample_index);
                            Bool geometry_consistent = (center_valid == 0u && sample_valid == 0u);
                            Float geometry_weight = select(
                                geometry_consistent, 1.0f, 0.0f);
                            Float geometry_exponent = 0.0f;
                            const Uint sample_instance = packed_spatial_geometry
                                ? sample_packed.instance_id
                                : param.instance_id.read(sample_index);

                            const Bool instance_compatible =
                                center_instance == sample_instance;
                            $if(center_valid != 0u && sample_valid != 0u &&
                                instance_compatible) {
                                const Float sample_depth = packed_spatial_geometry
                                    ? sample_packed.hit_distance
                                    : param.depth.read(sample_index);
                                const Float3 sample_normal = packed_spatial_geometry
                                    ? local_normal(param, sample_packed.normal)
                                    : spatial_normal(param, sample_index, false);
                                const Float3 sample_albedo = packed_spatial_geometry
                                    ? sample_packed.albedo
                                    : param.albedo.read(sample_index).xyz();
                                const Float3 sample_position = spatial_position(
                                    param, sample_index, sample_pixel,
                                    sample_depth,
                                    packed_spatial_geometry);
                                const Float3 position_delta =
                                    sample_position - center_position;
                                Float normal_weight = max(dot(center_normal, sample_normal), 0.0f);
                                if (combine_exponents) {
                                    for (int power = 0; power < 5; ++power) normal_weight *= normal_weight;
                                } else {
                                    normal_weight = pow(normal_weight, param.sigma_normal);
                                }
                                const Float plane_exponent = -abs(dot(position_delta, center_normal)) /
                                    max(length(position_delta) *
                                            param.sigma_depth,
                                        1.0e-3f);
                                const Float albedo_exponent = -length(center_albedo - sample_albedo) *
                                    param.sigma_albedo;
                                Float depth_weight = 1.0f, albedo_weight = 1.0f;
                                if (combine_exponents) {
                                    geometry_exponent = plane_exponent + albedo_exponent;
                                } else {
                                    depth_weight = exp(plane_exponent);
                                    albedo_weight = exp(albedo_exponent);
                                }
                                if (0u ==
                                    3u) {
                                    normal_weight = select(
                                        param.step_width == 4u, 1.0f,
                                        normal_weight);
                                }
                                if (0u ==
                                    4u) {
                                    depth_weight = select(
                                        param.step_width == 4u, 1.0f,
                                        depth_weight);
                                }
                                if (0u ==
                                    2u) {
                                    albedo_weight = select(
                                        param.step_width == 4u, 1.0f,
                                        albedo_weight);
                                }
                                geometry_weight = normal_weight * depth_weight *
                                    albedo_weight;

                                geometry_consistent = true;
                            };

                            $if(geometry_consistent) {
                                const int abs_x = x < 0 ? -x : x;
                                const int abs_y = y < 0 ? -y : y;
                                const float kernel_x =
                                    abs_x == 0 ? 0.375f :
                                    (abs_x == 1 ? 0.25f : 0.0625f);
                                const float kernel_y =
                                    abs_y == 0 ? 0.375f :
                                    (abs_y == 1 ? 0.25f : 0.0625f);
                                const Float sample_luminance=luminance(sample.xyz());
                                const Float difference = center_luminance - sample_luminance;
                                Float luminance_residual = abs(difference) / phi_luminance;
                                if (statistical_reconstruction) {
                                    Float squared_difference=difference*difference;
                                    Float difference_variance=center.w+sample.w;
                                    // Only the final Quality refinement changes its metric.
                                    // Earlier levels reconstruct full RGB using luminance.
                                    $if(param.step_width==4u) {
                                        const Float component=select(destination_carrier==0u,sample.x,select(destination_carrier==1u,sample.y,sample.z));
                                        const Float scale=component/max(sample_luminance,1.e-20f);
                                        const Float delta=center_component-component;
                                        squared_difference=delta*delta;
                                        difference_variance=center.w*center_component_scale*center_component_scale+sample.w*scale*scale;
                                    };
                                    luminance_residual = select(center_known && sample_known,
                                        0.5f * squared_difference / max(difference_variance, 1.0e-20f),
                                        luminance_residual);
                                }
                                Float luminance_weight = combine_exponents
                                    ? exp(geometry_exponent - luminance_residual)
                                    : exp(-luminance_residual);
                                if (0u ==
                                    1u) {
                                    luminance_weight = select(
                                        param.step_width == 4u, 1.0f,
                                        luminance_weight);
                                }
                                if (0u ==
                                    5u) {
                                    const Float cauchy = 1.0f /
                                        (1.0f + luminance_residual *
                                                    luminance_residual);
                                    luminance_weight = select(
                                        param.step_width == 4u, cauchy,
                                        luminance_weight);
                                }
                                const Float weight =
                                    kernel_x * kernel_y *
                                    geometry_weight * luminance_weight;
                                if (statistical_reconstruction) {
                                    variance_known = variance_known && (sample_known || weight == 0.0f);
                                }
                                Float3 sample_signal = sample.xyz();
                                color_sum += sample_signal * weight;
                                variance_sum += sample.w * weight * weight;
                                weight_sum += weight;
                            };
                        };
                    }
                }

                const Float inverse_weight = 1.0f / max(weight_sum, 1.0e-4f);
                const Float3 filtered = color_sum * inverse_weight;
                const Float filtered_variance = max(
                    variance_sum /
                        max(weight_sum * weight_sum, 1.0e-4f),
                    1.0e-4f);
                {
                param.destination.write(
                    index,
                    cast<half4>(make_float4(filtered, statistical_reconstruction ? select(variance_known, filtered_variance, -filtered_variance) : filtered_variance)));
                }
            };

            Kernel atrous_kernel = [&](Var<EncodedAtrousParam> param) noexcept {
                atrous_body(param);
            };

            const bool native_composition_only =
                true;
            const bool diffuse_detail_preservation =
                true;
            Kernel final_kernel = [
                                   packed_spatial_geometry,
                                   native_composition_only,
                                   diffuse_detail_preservation,
                                   statistical_reconstruction = true,
                                   glossy_single_frame_prior = config.responsive_angular_radiometry](
                Var<EncodedFinalParam> param) noexcept {
                const Int2 size = make_int2(dispatch_dim().xy());
                const Int2 pixel = make_int2(dispatch_idx().xy());
                const Uint index = dispatch_id();

                Float4 filtered = cast<float4>(param.source.read(index));
                const Bool temporal_variance_known=filtered.w>=0.f;
                if (statistical_reconstruction) filtered.w = abs(filtered.w);

                const PackedSpatialGuide packed = packed_spatial_geometry
                    ? packed_spatial_guide(param, index)
                    : PackedSpatialGuide{};
                const Uint valid = packed_spatial_geometry
                    ? packed.validity : param.validity.read(index);
                const Float3 albedo = safe_albedo(
                    packed_spatial_geometry
                        ? packed.albedo : param.albedo.read(index).xyz());
                const Float3 center_normal =
                    make_float3(0.0f, 0.0f, 1.0f);
                const Float3 center_position =
                    make_float3(0.0f);

                Bool preserve_filtered = statistical_reconstruction;
                if(glossy_single_frame_prior) {
                    const Uint cls=packed.surface_flags;
                    const Bool pure_glossy=valid==1u && (cls&static_cast<uint32_t>(backend::SurfaceFlagGlossy))!=0u &&
                        (cls&(static_cast<uint32_t>(backend::SurfaceFlagDiffuse)|static_cast<uint32_t>(backend::SurfaceFlagNearSpecular)|static_cast<uint32_t>(backend::SurfaceFlagTransmission)))==0u;
                    // Presentation reconstruction is the same for all pure-glossy observations.
                    // The variance validity bit still controls statistical ES weights,
                    // but does not select a different final display operator.
                    preserve_filtered=!pure_glossy; // Same-view final reconstruction, preserving the original G-only final stage.
                }
                if (diffuse_detail_preservation && !statistical_reconstruction) {
                    const Uint non_diffuse = packed.surface_flags &
                        (static_cast<uint32_t>(backend::SurfaceFlagGlossy) |
                         static_cast<uint32_t>(
                             backend::SurfaceFlagNearSpecular) |
                         static_cast<uint32_t>(
                             backend::SurfaceFlagTransmission) |
                         static_cast<uint32_t>(
                             backend::SurfaceFlagEmissive));
                    preserve_filtered = valid != 0u &&
                        (packed.surface_flags &
                         static_cast<uint32_t>(
                             backend::SurfaceFlagDiffuse)) != 0u &&
                         non_diffuse == 0u;
                }
                Float3 smooth_sum = make_float3(0.0f);
                Float smooth_weight = 0.0f;
                $if(!preserve_filtered) {
                    const Bool narrow=param.template_id_bits==16u;
                    const Uint word=param.view_chart_ids.read(select(narrow,index>>1u,index));
                    const Uint tid=select(narrow,(word>>((index&1u)*16u))&65535u,word);
                    $if(tid>=param.view_chart_templates){param.frame_status.write(0u,2u);}
                    $else {
                        const Uint center_view=(param.static_view_id.read(index>>2u)>>((index&3u)*8u))&255u;
                        const Float3 cn=local_normal(param,packed.normal);
                        const Float3 cp=factorized_position(param,index,pixel,packed.hit_distance);
                        constexpr float center_weight=.786986f*.786986f;
                        constexpr float neighbor_weight=(1.f-center_weight)/8.f;
                        for(uint k=0u;k<9u;++k){
                            const Uint bits=param.view_chart_offsets.read(tid*34u+25u+k);
                            const Int dx=cast<int>(bits<<16u)>>16,dy=cast<int>(bits)>>16;
                            const Int2 sp=pixel+make_int2(dx,dy);
                            $if(all(sp>=0)&&all(sp<size)){
                                const Uint j=cast<uint>(sp.y)*cast<uint>(size.x)+cast<uint>(sp.x);
                                const Uint view=(param.static_view_id.read(j>>2u)>>((j&3u)*8u))&255u;
                                $if(view!=center_view){param.frame_status.write(0u,2u);}
                                $else {
                                    const PackedSpatialGuide sg=packed_spatial_guide(param,j);
                                    const Bool center=j==index;
                                    $if(center || (sg.validity==1u&&sg.instance_id==packed.instance_id&&sg.surface_flags==packed.surface_flags)){
                                        Float weight=select(center,center_weight,neighbor_weight);
                                        $if(!center){
                                            const Float3 n=local_normal(param,sg.normal),p=factorized_position(param,j,sp,sg.hit_distance),dp=p-cp;
                                            Float nw=max(dot(cn,n),0.f);for(int a=0;a<5;++a)nw*=nw;
                                            weight*=nw*exp(-abs(dot(dp,cn))/max(length(dp),1.e-3f)-4.f*length(packed.albedo-sg.albedo));
                                        };
                                        smooth_sum+=cast<float4>(param.source.read(j)).xyz()*weight;smooth_weight+=weight;
                                    };
                                };
                            }
                            $else {param.frame_status.write(0u,2u);};
                        }
                    };
                };
                const Float3 smoothed = select(
                    preserve_filtered,
                    filtered.xyz(),
                    smooth_sum / max(smooth_weight, 1.0e-4f));
                // Zero detail restoration must not read the old/aliased
                // temporal allocation or evaluate an unused division/sqrt.
                Float3 combined = smoothed;

                const Float3 resolved =
                    combined * param.energy_compensation;
                const Float resolved_variance = max(
                    filtered.w * param.energy_compensation *
                        param.energy_compensation,
                    1.0e-4f);
                {
                param.resolved_signal.write(
                    index,
                    cast<half4>(make_float4(resolved, resolved_variance)));
                }

            };

            {

                if (config.atrous_iterations > 2u) {
                    atrous_shader = device->compile(
                        atrous_kernel,
                        "lfs_encoded_atrous");
                }
                final_shader = device->compile(
                    final_kernel, "lfs_encoded_final_combine_fp32");
            }
        } catch (const std::exception &exception) {
            return {backend::ErrorCode::BackendFailure, exception.what()};
        } catch (...) {
            return {backend::ErrorCode::BackendFailure,
                    "encoded filter shader compilation failed."};
        }

        telemetry.domain = domain;
        telemetry.extent = extent;
        telemetry.layout_generation = layout_generation;
        telemetry.layout_hash = layout_hash;
        const uint64_t half_buffers = std::count_if(history_radiance.begin(),history_radiance.end(),[](const auto& b){return b.size()!=0;}) +
            std::count_if(spatial.begin(),spatial.end(),[](const auto& b){return b.size()!=0;});
        telemetry.allocation_count = static_cast<uint32_t>(half_buffers);
        telemetry.internal_history_bytes = extent.area() *
            (half_buffers * sizeof(half4));
        {
            ++telemetry.allocation_count;
            telemetry.internal_history_bytes += sizeof(uint32_t);
        }
        telemetry.internal_history_bytes+=(spatial_native_status.size()-1u)*sizeof(uint);
        telemetry.internal_history_bytes+=view_chart_ids.size()*sizeof(uint)+view_chart_offsets.size()*sizeof(uint);
        telemetry.allocation_count+=view_chart_ids.size()!=0u?2u:0u;
        telemetry.internal_history_bytes+=(view_chart_centers.size()+view_chart_blocks.size()+view_chart_sources.size())*sizeof(uint);
        telemetry.allocation_count+=view_chart_centers.size()!=0u?3u:0u;
        telemetry.internal_history_bytes+=view_chart_slots.size()*sizeof(uint);telemetry.allocation_count+=view_chart_slots.size()!=0u?1u:0u;
        telemetry.internal_history_bytes+=canonical_output.size()*sizeof(float4);telemetry.allocation_count+=canonical_output.size()!=0u?1u:0u;
        telemetry.prepared = true;
        telemetry.output_ready = false;
        return {};
    }
};

EncodedFilterCore::EncodedFilterCore(EncodedConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

EncodedFilterCore::EncodedFilterCore(EncodedFilterCore &&) noexcept = default;
EncodedFilterCore &EncodedFilterCore::operator=(EncodedFilterCore &&) noexcept = default;
EncodedFilterCore::~EncodedFilterCore() noexcept = default;

backend::Error EncodedFilterCore::reset(backend::ResetReason reason) noexcept {
    if (reason == backend::ResetReason::None) {
        return {backend::ErrorCode::InvalidArgument,
                "encoded filter reset reason must not be None."};
    }
    impl_->invalidate_history();
    impl_->invalidate_native_current();
    impl_->demodulated_output = {};

    impl_->spatial_native_runtime.clear_submission();
    impl_->spatial_native_submission = false;

    ++impl_->telemetry.reset_count;
    impl_->telemetry.history_valid_before = false;
    impl_->telemetry.temporal_used = false;
    impl_->telemetry.output_ready = false;
    return {};
}

backend::Error EncodedFilterCore::prepare(
    const backend::FrameContext &context) noexcept {
    return impl_->allocate_and_compile(context);
}

backend::Error EncodedFilterCore::enqueue_commit(
    const backend::FrameContext &,
    CommandBatch &commands) noexcept {
    if (impl_->angular_commit_pending && !impl_->native_trace_scratch) {
        // Both hit banks are dead after presentation. The next original GBuffer
        // will overwrite its current bank, leaving the other copy readable.
        commands << impl_->angular_residuals.copy_to(impl_->angular_mirror);
    }
    return {};
}

void EncodedFilterCore::finalize_success(const backend::FrameContext &) noexcept {
    impl_->motion_committed=impl_->motion_current;
    impl_->held_raw_committed=impl_->held_raw_pending;
    impl_->held_raw_pending=Impl::HeldRawStorage::Empty;
    impl_->stationary_cache_valid=!impl_->resident_scratch && impl_->phase_state.repeated_camera();
    impl_->angular_history_valid = impl_->angular_commit_pending;
    impl_->angular_commit_pending = false;
    impl_->spatial_native_runtime.clear_submission();impl_->spatial_native_submission=false;
}
void EncodedFilterCore::abort_submission(const backend::FrameContext &) noexcept {
    impl_->invalidate_history();impl_->invalidate_native_current();
    impl_->spatial_native_runtime.clear_submission();impl_->spatial_native_submission=false;
    impl_->demodulated_output={};
}

backend::Error EncodedFilterCore::validate_submission(const backend::FrameContext &) noexcept {
    return impl_->spatial_native_submission?impl_->spatial_native_runtime.validate_submission():backend::Error{};
}

backend::Result<backend::ResourceRequirementSet>
EncodedFilterCore::resource_requirements(
    const backend::FrameContext &) noexcept {
    backend::ResourceRequirementSet requirements;
    for (auto [semantic, type] : {std::pair{backend::Semantic::Radiance, backend::ValueType::Float4},
            std::pair{backend::Semantic::PackedSurfaceObservation, backend::ValueType::PackedSurfaceObservation32}}) {
        auto added = requirements.add({semantic, backend::TemporalSlot::Current, 0u, type, false});
        if (!added.ok()) return backend::Result<backend::ResourceRequirementSet>::failure(added.error());
    }
    return backend::Result<backend::ResourceRequirementSet>::success(
        std::move(requirements));
}

backend::Result<backend::ProcessOutput> EncodedFilterCore::process(
    const backend::FrameContext &context,
    const backend::FrameResources &resources,
    CommandBatch &commands) noexcept {
    impl_->history_pending_commit = false;
    impl_->telemetry.spatial_init_step1_fused = false;
    impl_->demodulated_output = {};

    impl_->spatial_native_runtime.clear_submission();
    impl_->spatial_native_submission = false;
    const bool use_history = false;
    backend::Result<backend::ResourceRequirementSet> required =
        resource_requirements(context);
    if (!required.ok()) {
        return backend::Result<backend::ProcessOutput>::failure(
            required.error());
    }
    const backend::Result<void> validation = resources.validate(required.value());
    if (!validation.ok()) {
        return backend::Result<backend::ProcessOutput>::failure(validation.error());
    }
    if (!impl_->telemetry.prepared || context.device != impl_->device ||
        context.sample_extent != impl_->extent ||
        context.sample_domain != impl_->domain ||
        context.layout_generation != impl_->layout_generation ||
        context.layout_hash != impl_->layout_hash) {
        return backend::Result<backend::ProcessOutput>::failure({
            backend::ErrorCode::InvalidState,
            "encoded filter process context does not match its prepared layout."});
    }
    if (!impl_->phase_state_matches(context)) {
        return backend::Result<backend::ProcessOutput>::failure({
            backend::ErrorCode::ResourceMismatch,
            "Encoded Spatial static LF table does not match this frame."});
    }

    const auto *radiance = required_view<float4>(resources, backend::Semantic::Radiance);
    const auto *packed_observation = required_view<backend::PackedSurfaceObservationV1>(
        resources, backend::Semantic::PackedSurfaceObservation);
    if (!radiance || !packed_observation || packed_observation->size() != context.sample_extent.area()) {
        return backend::Result<backend::ProcessOutput>::failure(resource_error("Spatial resources are incomplete."));
    }
    const uint32_t write_index = 0u;
    {
        EncodedPrefilterParam prefilter{};
        if(impl_->resident_scratch){
            const auto count=impl_->extent.area();
            impl_->native_trace_scratch=resident::native_packed_owner(context.device);
            const auto native=resident::trace_workspaces(context.device,count,native_address(*packed_observation));
            const auto workspace=impl_->native_trace_scratch?native.spatial:resident::geometry_bank(context.device,0,count);
            const auto canonical=impl_->native_trace_scratch?native.canonical:resident::geometry_bank(context.device,1,count);
            if(!workspace||!canonical||workspace==canonical)
                return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::ResourceMismatch,"Cannot acquire post-trace workspace."});
            BufferView<half4> scratch{workspace,size_t(2*count)};
            impl_->borrowed_spatial={scratch.subview(0,size_t(count)),scratch.subview(size_t(count),size_t(count))};
            impl_->borrowed_canonical=BufferView<float4>{canonical,size_t(count/3)};
        }
        prefilter.destination = impl_->spatial_bank(0).descriptor();

        {
            prefilter.packed_observation =
                packed_observation->view_as<uint4>().descriptor();
            prefilter.static_view_id = BufferView<uint>{
                impl_->phase_state.static_view_id_address,
                (context.sample_extent.area() + 3u) / 4u}.descriptor();
            prefilter.static_inverse_ray_length = BufferView<float>{
                impl_->phase_state.static_inverse_ray_length_address,
                context.sample_extent.area()}.descriptor();
            prefilter.static_view_x = BufferView<float>{
                impl_->phase_state.static_view_x_address,
                impl_->phase_state.static_view_count}.descriptor();
            prefilter.static_focal_x = BufferView<float>{
                impl_->phase_state.static_focal_x_address,
                context.sample_extent.width}.descriptor();
            prefilter.static_focal_y = BufferView<float>{
                impl_->phase_state.static_focal_y_address,
                context.sample_extent.height}.descriptor();
            const auto &transform = impl_->phase_state.current.local_to_world;
            prefilter.world_to_local_row0 = make_float4(
                static_cast<float>(transform.axis_x.x),
                static_cast<float>(transform.axis_x.y),
                static_cast<float>(transform.axis_x.z), 0.0f);
            prefilter.world_to_local_row1 = make_float4(
                static_cast<float>(transform.axis_y.x),
                static_cast<float>(transform.axis_y.y),
                static_cast<float>(transform.axis_y.z), 0.0f);
            prefilter.world_to_local_row2 = make_float4(
                static_cast<float>(transform.axis_z.x),
                static_cast<float>(transform.axis_z.y),
                static_cast<float>(transform.axis_z.z), 0.0f);
            prefilter.focal_distance = static_cast<float>(
                impl_->phase_state.current.focal_distance);
        }
        encoded_atrous::native::EncodedAtrousNativeParams native_atrous{};
        bool captured_exposure_in_es1=false;
        const bool native_tiled_atrous = true;
        const bool fused_first = native_tiled_atrous &&
            ((impl_->config.atrous_iterations == 2u && !impl_->config.responsive_angular_radiometry) ||
             (impl_->config.atrous_iterations == 3u && impl_->config.responsive_angular_radiometry));
        native_atrous.resident_storage=impl_->resident_scratch?1u:0u;
        native_atrous.template_id_bits=impl_->template_id_bits;
        const bool angular_active=impl_->resident_scratch?
            impl_->config.atrous_iterations==3u && !impl_->phase_state.repeated_camera():!impl_->stationary_path();
        if (!impl_->resident_scratch) {
            native_atrous.canonical = native_address(impl_->canonical_view());
            native_atrous.final_energy = impl_->config.energy_compensation;
            native_atrous.deterministic_environment = context.deterministic_environment ? 1u : 0u;
        }
        native_atrous.register_motion=fused_first && impl_->phase_state.previous_valid && angular_active;
        impl_->angular_commit_pending = false;
        impl_->held_raw_pending=Impl::HeldRawStorage::Empty;
        const bool resident_quality = impl_->resident_scratch && impl_->config.atrous_iterations == 3u;
        const bool hold_registration = resident_quality && impl_->angular_history_valid &&
            impl_->phase_state.previous_valid && impl_->phase_state.repeated_camera();
        if (resident_quality && (native_atrous.register_motion || hold_registration)) {
            const auto count = context.sample_extent.area();
            const auto offset = context.output_extent.area() * sizeof(float4) + count * sizeof(uint2);
            const auto capacity = count * sizeof(uint4);
            if (count != 3ull * context.output_extent.area() || offset > capacity ||
                count * sizeof(std::uint16_t) > capacity - offset) {
                return backend::Result<backend::ProcessOutput>::failure(resource_error("The previous angular estimate would overlap live resident storage."));
            }
            const auto canonical_bank = native_address(impl_->borrowed_canonical);
            const auto spatial_bank = native_address(impl_->borrowed_spatial[0]);
            impl_->angular_residuals = BufferView<std::uint16_t>{canonical_bank + offset, size_t(count)};
            impl_->angular_mirror = BufferView<std::uint16_t>{spatial_bank + offset, size_t(count)};
            if (hold_registration && !impl_->native_trace_scratch) {
                const auto preserved = resident::inactive_geometry_bank(context.device, count);
                if (preserved == spatial_bank) {
                    commands << impl_->angular_mirror.copy_to(impl_->angular_residuals);
                } else if (preserved != canonical_bank) {
                    return backend::Result<backend::ProcessOutput>::failure(resource_error("The original source did not retain its inactive hit bank."));
                }
            }
            if (hold_registration) native_atrous.register_motion = 2u;
            native_atrous.resident_angular_residuals = native_address(impl_->angular_residuals);
            impl_->angular_commit_pending = true;
        }
        native_atrous.hold_same=(!impl_->resident_scratch && fused_first && impl_->motion_current && impl_->phase_state.repeated_camera() &&
            context.geometry_motion_model!=backend::TraceRequest::GeometryMotionModel::Deforming)?
            (impl_->stationary_cache_valid?2u:1u):0u;
        if(native_atrous.hold_same && (impl_->extent.width==4608u || impl_->extent.width==11520u)) {
            const size_t required=4ull*impl_->extent.area();
            if(impl_->hold_state.size()!=required) {
                impl_->hold_state=impl_->device->create_buffer<half4>(required,"lfs.held_optical_observations");
                impl_->telemetry.internal_history_bytes+=required*sizeof(half4);
                ++impl_->telemetry.allocation_count;
            }
            native_atrous.hold_state=native_address(impl_->hold_state.view());
        }
        impl_->telemetry.spatial_init_step1_fused = fused_first;
        {
            commands << impl_->clear_spatial_native_status_shader(
                impl_->spatial_native_status).dispatch(uint32_t(impl_->spatial_native_status.size()));
            native_atrous.radiance = native_address(*radiance);
            native_atrous.packed_observation =
                native_address(*packed_observation);
            native_atrous.static_view_id =
                impl_->phase_state.static_view_id_address;
            native_atrous.static_inverse_ray_length =
                impl_->phase_state.static_inverse_ray_length_address;
            native_atrous.static_view_x =
                impl_->phase_state.static_view_x_address;
            native_atrous.static_focal_x =
                impl_->phase_state.static_focal_x_address;
            native_atrous.static_focal_y =
                impl_->phase_state.static_focal_y_address;

            native_atrous.scratch_a = native_address(impl_->spatial_bank(0));
            native_atrous.scratch_b = native_address(impl_->spatial_bank(1));
            native_atrous.frame_status =
                native_address(impl_->spatial_native_status.view());
            native_atrous.world_to_local_row0 = {
                prefilter.world_to_local_row0.x,
                prefilter.world_to_local_row0.y,
                prefilter.world_to_local_row0.z,
                prefilter.world_to_local_row0.w};
            native_atrous.world_to_local_row1 = {
                prefilter.world_to_local_row1.x,
                prefilter.world_to_local_row1.y,
                prefilter.world_to_local_row1.z,
                prefilter.world_to_local_row1.w};
            native_atrous.world_to_local_row2 = {
                prefilter.world_to_local_row2.x,
                prefilter.world_to_local_row2.y,
                prefilter.world_to_local_row2.z,
                prefilter.world_to_local_row2.w};
            native_atrous.focal_distance = prefilter.focal_distance;
            native_atrous.static_view_count =
                impl_->phase_state.static_view_count;
            native_atrous.count = static_cast<uint32_t>(
                context.sample_extent.area());

            native_atrous.angular_compatibility_spatial =
                impl_->config.angular_compatibility_spatial
                    ? 1u : 0u;
            native_atrous.responsive_angular_radiometry =
                impl_->config.responsive_angular_radiometry ? 1u : 0u;
            native_atrous.angular_compatibility_aperture_only =
                impl_->config.angular_compatibility_aperture_only ? 1u : 0u;

            native_atrous.responsive_radiometry_glossy_only =
                impl_->config.responsive_radiometry_glossy_only ? 1u : 0u;
            native_atrous.angular_compatibility_responsive_only =
                0u;

            native_atrous.statistical_reconstruction = 1u;
            native_atrous.temporal_history_available =
                impl_->config.responsive_angular_radiometry &&
                impl_->phase_state.previous_valid ? 1u : 0u;

            if (!impl_->stationary_path()) {
                auto acquire=native_atrous;
                if(!impl_->resident_scratch && impl_->config.atrous_iterations==3u && fused_first &&
                   acquire.register_motion==1u && !acquire.hold_same && acquire.deterministic_environment &&
                   (impl_->extent.width==4608u || impl_->extent.width==11520u) &&
                   impl_->stationary_weights.size()*sizeof(uint4)>=size_t(7)*impl_->extent.area()*sizeof(uint) &&
                   impl_->cell_coefficients.size()*sizeof(uint)>=impl_->extent.area()*sizeof(float4) &&
                   impl_->stationary_moments.size()==impl_->stationary_atlas.size() &&
                   impl_->stationary_moments.size()*sizeof(uint4)>=impl_->extent.area()*sizeof(half4)) {
                    acquire.atlas_moments=native_address(impl_->stationary_moments.view());
                    captured_exposure_in_es1=true;
                }
                if (const backend::Error dispatch_error =
                        impl_->spatial_native_runtime.dispatch_init_prefilter(
                            encoded_atrous::native::extent_profile(impl_->extent),
                            acquire, commands, fused_first)) {
                    return backend::Result<backend::ProcessOutput>::failure(
                        dispatch_error);
                }
                impl_->spatial_native_submission = true;
            }

            native_atrous.angular_compatibility_spatial =
                impl_->config.responsive_radiometry_glossy_only ||
                (impl_->config.angular_compatibility_spatial &&
                        !impl_->config.angular_compatibility_prefilter_only)
                    ? 1u : 0u;
            if (impl_->config.responsive_radiometry_glossy_only) {
                // Device-side glossy scope applies to both init and recursive
                // stages. Recursive stages additionally require a responsive
                // guide, so diffuse/mixed current remains the base ES result.
                native_atrous.angular_compatibility_responsive_only = 1u;
            }
            native_atrous.responsive_angular_radiometry =
                impl_->config.responsive_angular_radiometry
                    ? 1u : 0u;
        }
        BufferView<half4> spatial_source = impl_->spatial_bank(0);

        for (uint32_t iteration = 0u;
             iteration < impl_->config.atrous_iterations;
             ++iteration) {
            BufferView<half4> destination =
                impl_->spatial_bank((iteration + 1u) & 1u);
                    EncodedAtrousParam atrous{};
            atrous.source = spatial_source.descriptor();
            atrous.destination = destination.descriptor();

            {
                atrous.packed_observation =
                    packed_observation->view_as<uint4>().descriptor();
                atrous.static_view_id = prefilter.static_view_id;
                atrous.static_inverse_ray_length =
                    prefilter.static_inverse_ray_length;
                atrous.static_view_x = prefilter.static_view_x;
                atrous.static_focal_x = prefilter.static_focal_x;
                atrous.static_focal_y = prefilter.static_focal_y;
                atrous.world_to_local_row0 = prefilter.world_to_local_row0;
                atrous.world_to_local_row1 = prefilter.world_to_local_row1;
                atrous.world_to_local_row2 = prefilter.world_to_local_row2;
                atrous.focal_distance = prefilter.focal_distance;

            }
            atrous.step_width = 1u << iteration;
            atrous.sigma_luminance = impl_->config.sigma_luminance;
            atrous.sigma_normal = impl_->config.sigma_normal;
            atrous.sigma_depth = impl_->config.sigma_depth;
            atrous.sigma_albedo = impl_->config.sigma_albedo;

            const bool stationary_final = iteration==1u && impl_->stationary_path();
            const bool terminal_iteration = stationary_final || iteration + 1u == impl_->config.atrous_iterations;
            if (terminal_iteration) {
                native_atrous.canonical=native_address(impl_->canonical_view());native_atrous.final_energy=impl_->config.energy_compensation;native_atrous.deterministic_environment=context.deterministic_environment?1u:0u;
                if(impl_->view_chart_count!=impl_->extent.area()||impl_->view_chart_views!=impl_->phase_state.static_view_count||impl_->view_chart_ids.size()!=(impl_->template_id_bits==16u?(impl_->extent.area()+1u)/2u:impl_->extent.area())||impl_->view_chart_offsets.size()!=size_t(impl_->view_chart_templates)*(!impl_->resident_scratch && impl_->config.atrous_iterations==3u?43u:34u))
                    return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::InvalidState,"View-chart layout binding mismatch; no fallback."});
                native_atrous.view_chart_ids=native_address(impl_->view_chart_ids.view());native_atrous.view_chart_offsets=native_address(impl_->view_chart_offsets.view());
                native_atrous.view_chart_templates=impl_->view_chart_templates;native_atrous.view_chart_count=impl_->view_chart_count;native_atrous.view_chart_views=impl_->view_chart_views;
                native_atrous.view_chart_centers=native_address(impl_->view_chart_centers.view());native_atrous.view_chart_blocks=native_address(impl_->view_chart_blocks.view());native_atrous.view_chart_sources=native_address(impl_->view_chart_sources.view());
                native_atrous.view_chart_slots=native_address(impl_->view_chart_slots.view());
                native_atrous.view_chart_block_count=impl_->view_chart_block_count;native_atrous.view_chart_source_count=impl_->view_chart_source_count;native_atrous.view_chart_cache_wire=impl_->config.atrous_iterations==2u?0x56434233u:0x56434232u;
                native_atrous.view_chart_wire=0x56434832u;native_atrous.view_chart_focal_bounds={impl_->view_chart_focal_bounds.x,impl_->view_chart_focal_bounds.y,impl_->view_chart_focal_bounds.z,impl_->view_chart_focal_bounds.w};
            }
            if (stationary_final) {
                native_atrous.scratch_a=native_address(spatial_source);
                native_atrous.scratch_b=native_address(impl_->spatial[0].view());
                native_atrous.atlas=native_address(impl_->stationary_atlas.view());
                native_atrous.atlas_cells=uint32_t(impl_->stationary_atlas.size());
                native_atrous.atlas_side=impl_->atlas_side;
                if(impl_->stationary_moments.size()) {
                    native_atrous.atlas_moments=native_address(impl_->stationary_moments.view());
                    native_atrous.cell_coefficients=native_address(impl_->cell_coefficients.view());
                }
                native_atrous.atlas_inverse_dx=impl_->atlas_inverse_dx;
                native_atrous.atlas_inverse_span=impl_->atlas_inverse_span;
                {
                    const auto *rigid=resources.rigid_instance_motion();
                    if(!rigid || !rigid->structurally_valid())
                        return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::ResourceMismatch,"Missing stationary geometry identity."});
                    native_atrous.stationary_weights=native_address(impl_->stationary_weights.view());
                    native_atrous.weights_refresh=native_address(impl_->weights_refresh.view());
                    native_atrous.rigid_records=native_address(rigid->records);
                    native_atrous.rigid_count=uint32_t(rigid->records.size());
                    native_atrous.rebuild_weights=!impl_->stationary_cache_valid || !rigid->history_valid ||
                        context.geometry_motion_model==backend::TraceRequest::GeometryMotionModel::Deforming;
                }
                if(auto error=impl_->spatial_native_runtime.dispatch_stationary(
                        encoded_atrous::native::extent_profile(impl_->extent),native_atrous,commands))
                    return backend::Result<backend::ProcessOutput>::failure(error);
                impl_->spatial_native_submission=true;
                break;
            }
            if (native_tiled_atrous && iteration < 3u) {
                if (const backend::Error dispatch_error = (fused_first && iteration == 0u)
                        ? backend::Error{} :
                        impl_->spatial_native_runtime.dispatch_atrous_pass(
                            encoded_atrous::native::extent_profile(impl_->extent),
                            1u << iteration, native_atrous, commands)) {
                    return backend::Result<backend::ProcessOutput>::failure(
                        dispatch_error);
                }
                impl_->spatial_native_submission = true;
            } else {
                commands << impl_->atrous_shader(atrous).dispatch(
                    make_uint2(impl_->extent.width, impl_->extent.height));
            }
            spatial_source = destination;
            if (iteration == 0u && impl_->interstage_processor) {
                std::uint64_t regular_material_normals{},panel_samples{};
                if (backend::Error error = impl_->interstage_processor(
                        context, resources, spatial_source,
                        impl_->spatial_native_status.view(), native_atrous.hold_previous,
                        native_atrous.regular_measurement_geometry,regular_material_normals,panel_samples,
                        native_atrous.register_motion == 2u ? native_atrous.resident_angular_residuals : 0u,
                        impl_->native_trace_scratch ? native_atrous.resident_angular_residuals : 0u,
                        impl_->spatial_native_runtime.regular_r2_coverage(native_atrous), commands)) {
                    return backend::Result<backend::ProcessOutput>::failure(error);
                }
                if(!impl_->resident_scratch && !impl_->stationary_path())
                    native_atrous.regular_material_normals=regular_material_normals;
                if(!impl_->resident_scratch && native_atrous.register_motion && !impl_->stationary_path())
                    native_atrous.panel_samples=panel_samples;
        if(native_atrous.register_motion && !impl_->resident_scratch) {
            native_atrous.motion_cell_sites=native_address(impl_->motion_cell_sites.view());
            native_atrous.motion_cell_values=native_address(impl_->motion_cell_values.view());
            native_atrous.motion_cell_count=uint32_t(impl_->motion_cell_sites.size());
            native_atrous.atlas=native_address(impl_->stationary_atlas.view());
            native_atrous.atlas_cells=uint32_t(impl_->stationary_atlas.size());
            native_atrous.atlas_side=impl_->atlas_side;
            native_atrous.atlas_inverse_dx=impl_->atlas_inverse_dx;
            native_atrous.atlas_inverse_span=impl_->atlas_inverse_span;
            native_atrous.stationary_weights=native_address(impl_->stationary_weights.view());
            if(!impl_->resident_scratch && impl_->config.atrous_iterations==3u &&
               impl_->stationary_weights.size()*sizeof(uint4)>=size_t(7)*impl_->extent.area()*sizeof(uint))
                native_atrous.regular_repeat_state=native_atrous.stationary_weights;
            // These two buffers belong to stationary reconstruction until the
            // camera first moves. A moving session can return to that owner
            // only after a history reset. Save a real RGB exposure in the
            // cell-moment bank, before the next trace overwrites its source.
            if(native_atrous.regular_repeat_state && (impl_->extent.width==4608u || impl_->extent.width==11520u) &&
               impl_->cell_coefficients.size()*sizeof(uint)>=impl_->extent.area()*sizeof(float4) &&
               impl_->stationary_moments.size()==impl_->stationary_atlas.size() &&
               impl_->stationary_moments.size()*sizeof(uint4)>=impl_->extent.area()*sizeof(half4)) {
                native_atrous.atlas_moments=native_address(impl_->stationary_moments.view());
                native_atrous.previous_held_raw=impl_->held_raw_committed;
                if(native_atrous.hold_state) {
                    native_atrous.cell_coefficients=native_address(impl_->cell_coefficients.view());
                    impl_->held_raw_pending=Impl::HeldRawStorage::HeldSeries;
                } else {
                    impl_->held_raw_pending=Impl::HeldRawStorage::MovingExposure;
                }
            }
            if(native_atrous.hold_same){
                const auto *rigid=resources.rigid_instance_motion();
                if(!rigid || !rigid->structurally_valid())
                    return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::ResourceMismatch,"Missing fixed-ray geometry."});
                native_atrous.rigid_records=native_address(rigid->records);
                native_atrous.rigid_count=uint32_t(rigid->records.size());
            }

            native_atrous.field_measurements=native_address(spatial_source);
            auto field=native_atrous;
            field.scratch_a=field.field_measurements;
            if(auto e=impl_->spatial_native_runtime.dispatch_motion_field(
                    encoded_atrous::native::extent_profile(impl_->extent),field,commands))
                return backend::Result<backend::ProcessOutput>::failure(e);
            impl_->spatial_native_submission=true;
        }
            }

        }
        const bool native_two_pass = native_tiled_atrous &&
            impl_->config.atrous_iterations == 2u;
        if(impl_->stationary_path() || (native_two_pass&&!impl_->config.responsive_angular_radiometry)){
            impl_->demodulated_output={};
        }else if(impl_->config.atrous_iterations==3u&&impl_->config.responsive_angular_radiometry){
            native_atrous.scratch_a=native_address(spatial_source);native_atrous.scratch_b=native_address(impl_->spatial_bank(0));native_atrous.canonical=native_address(impl_->canonical_view());native_atrous.final_energy=impl_->config.energy_compensation;native_atrous.deterministic_environment=context.deterministic_environment?1u:0u;
            auto panel=native_atrous;
            // The moving panel only writes this previous-exposure bank. The
            // original held consumer and subsequent submission retain their
            // own full parameters; no panel-history or normal field is reused.
            if(captured_exposure_in_es1)panel.atlas_moments=0u;
            if(auto error=impl_->spatial_native_runtime.dispatch_final(encoded_atrous::native::extent_profile(impl_->extent),panel,commands))return backend::Result<backend::ProcessOutput>::failure(error);
            impl_->spatial_native_submission=true;
            // This is the pre-presentation observation. Energy/Half/modulation
            // belong solely to the fused canonical output, never to M history.
            // Quality publishes canonical only. The post-M RGB banks have
            // disjoint reconstruction owners and are not a unified current view.
            impl_->demodulated_output={};
        }else{
            EncodedFinalParam final{};
            final.source = spatial_source.descriptor();

            final.temporal = native_two_pass
                ? spatial_source.descriptor()
                : impl_->history_radiance[write_index].view().descriptor();
            {
                final.packed_observation =
                    packed_observation->view_as<uint4>().descriptor();
                final.static_view_id = prefilter.static_view_id;
                final.static_inverse_ray_length =
                    prefilter.static_inverse_ray_length;
                final.static_view_x = prefilter.static_view_x;
                final.static_focal_x = prefilter.static_focal_x;
                final.static_focal_y = prefilter.static_focal_y;
                final.world_to_local_row0 = prefilter.world_to_local_row0;
                final.world_to_local_row1 = prefilter.world_to_local_row1;
                final.world_to_local_row2 = prefilter.world_to_local_row2;
                final.focal_distance = prefilter.focal_distance;
            }
            // The raw dense radiance has already been consumed by init/temporal at
            // this point. Reuse it for the final FP32 result instead of owning a
            // second full-frame output allocation.

            final.resolved_signal = (native_two_pass
                       ? impl_->spatial_bank(1).descriptor()
                       : impl_->history_radiance[write_index].view().descriptor());

            if(impl_->config.atrous_iterations==3u&&impl_->config.responsive_angular_radiometry){
                final.view_chart_ids=impl_->view_chart_ids.view().descriptor();final.view_chart_offsets=impl_->view_chart_offsets.view().descriptor();final.frame_status=impl_->spatial_native_status.view().descriptor();final.view_chart_templates=impl_->view_chart_templates;final.template_id_bits=impl_->template_id_bits;
            }
            final.energy_compensation = impl_->config.energy_compensation;
            if(impl_->config.atrous_iterations==3u&&impl_->config.responsive_angular_radiometry){
                native_atrous.scratch_a=native_address(spatial_source);native_atrous.scratch_b=native_address(impl_->history_radiance[write_index].view());native_atrous.final_energy=impl_->config.energy_compensation;
                if(auto error=impl_->spatial_native_runtime.dispatch_final(encoded_atrous::native::extent_profile(impl_->extent),native_atrous,commands))return backend::Result<backend::ProcessOutput>::failure(error);
                impl_->spatial_native_submission=true;
            }else{            commands << impl_->final_shader(final).dispatch(
                make_uint2(impl_->extent.width, impl_->extent.height));
            }

            {
                impl_->demodulated_output = native_two_pass
                    ? impl_->spatial_bank(1)
                    : impl_->history_radiance[write_index].view();
            }
        }
        if(native_atrous.hold_state) {
            if(auto error=impl_->spatial_native_runtime.dispatch_held_es(
                encoded_atrous::native::extent_profile(impl_->extent),native_atrous,commands))
                return backend::Result<backend::ProcessOutput>::failure(error);
            impl_->spatial_native_submission=true;
        }
        if(native_atrous.register_motion && impl_->resident_scratch) {
            native_atrous.scratch_a=native_address(impl_->spatial_bank(1));
            native_atrous.scratch_b=native_address(impl_->spatial_bank(0));
            native_atrous.aperture_cell_bits=impl_->resident_scratch?impl_->aperture_cell_bits:0u;
            native_atrous.atlas=impl_->resident_scratch?native_address(impl_->registered_cells.view()):native_address(impl_->stationary_atlas.view());
            native_atrous.atlas_cells=impl_->resident_scratch?impl_->aperture_cell_count:uint32_t(impl_->stationary_atlas.size());
            native_atrous.atlas_side=impl_->atlas_side;
            native_atrous.atlas_inverse_dx=impl_->atlas_inverse_dx;
            native_atrous.atlas_inverse_span=impl_->atlas_inverse_span;
            if(auto error=impl_->spatial_native_runtime.dispatch_resident_registered(
                      encoded_atrous::native::extent_profile(impl_->extent),native_atrous,commands))
                return backend::Result<backend::ProcessOutput>::failure(error);
            impl_->spatial_native_submission=true;
        }
    }

    impl_->telemetry.frame_index = context.frame_index;
    impl_->telemetry.history_valid_before = use_history;
    impl_->telemetry.temporal_used = use_history;
    impl_->telemetry.output_ready = true;
    impl_->history_pending_commit = false;

    {
        return backend::Result<backend::ProcessOutput>::success(
            {{}, false, true});
    }
}

backend::Result<backend::PresentOutput> EncodedFilterCore::present(
    const backend::FrameContext &,
    const backend::FrameResources &,
    const backend::ProcessOutput &output,
    CommandBatch &) noexcept {
    if (!impl_->telemetry.output_ready ||
        output.radiance.size() != impl_->extent.area()) {
        return backend::Result<backend::PresentOutput>::failure({
            backend::ErrorCode::InvalidState,
            "encoded filter output is unavailable for presentation."});
    }
    return backend::Result<backend::PresentOutput>::success({
        false,
        "lfs_encoded_spatial_output"});
}

const EncodedConfig &EncodedFilterCore::config() const noexcept {
    return impl_->config;
}

const EncodedTelemetry &EncodedFilterCore::telemetry() const noexcept {
    return impl_->telemetry;
}

bool EncodedFilterCore::set_phase_frame_state(const lightfield::PhaseFrameState &state) noexcept {
    if (!impl_->telemetry.prepared || !state.valid() ||
        state.layout_generation != impl_->layout_generation ||
        state.layout_hash != impl_->layout_hash ||
        state.static_ray_count != impl_->extent.area() ||
        state.static_focal_x_count != impl_->extent.width ||
        state.static_focal_y_count != impl_->extent.height) {
        impl_->phase_state_bound = false;
        return false;
    }
    if(!impl_->prepare_view_chart(state)){impl_->phase_state_bound=false;return false;}
    impl_->motion_current=state.previous_valid && (impl_->motion_committed || !state.repeated_camera());
    impl_->phase_state = state;
    impl_->phase_state_bound = true;
    return true;
}

void EncodedFilterCore::invalidate_phase_frame_state() noexcept {
    impl_->phase_state_bound = false;
}

void EncodedFilterCore::invalidate_demodulated_current_view() noexcept {
    impl_->invalidate_native_current();
}

BufferView<float4> EncodedFilterCore::canonical_output_view() const noexcept {return impl_->canonical_view();}

BufferView<half4> EncodedFilterCore::demodulated_output_view() const noexcept {
    return impl_->demodulated_output;
}

BufferView<uint> EncodedFilterCore::frame_status_view() const noexcept {
    return impl_->spatial_native_status.view();
}

void EncodedFilterCore::set_interstage_processor(InterstageProcessor processor) {
    impl_->interstage_processor = std::move(processor);
}

} // namespace lfs::denoise
