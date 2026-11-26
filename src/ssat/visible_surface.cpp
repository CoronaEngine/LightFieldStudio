#include "ssat/visible_surface.h"
#include "base/mgr/global.h"
#include "base/mgr/pipeline.h"
#include "base/scattering/material.h"
#include "base/sensor/frame_buffer.h"
#include "rhi/resources/shader.h"

namespace lfs::denoise::resident {
using namespace ocarina;
struct GuideArguments {
    BufferDesc<vision::RayData> rays;
    BufferDesc<TriangleHit> hits;
    BufferDesc<vision::source::PackedSurfaceObservationV1> output;
    unsigned width{};
    unsigned frame{};
};
}
OC_PARAM_STRUCT(lfs::denoise::resident, GuideArguments, rays, hits, output, width, frame) {};

namespace lfs::denoise::resident {
using namespace ocarina;
using namespace vision;

struct VisibleSurface::Impl {
    Device* device{};
    Pipeline* pipeline{};
    std::uint64_t generation{};
    Shader<void(GuideArguments)> shader{};
};
VisibleSurface::VisibleSurface() : impl_(std::make_unique<Impl>()) {}
VisibleSurface::~VisibleSurface()=default;

backend::Error VisibleSurface::prepare(const backend::FrameContext& c) noexcept {
    try {
        Pipeline* pipeline=Global::instance().pipeline();
        if(!pipeline || &pipeline->device()!=c.device)
            return {backend::ErrorCode::InvalidState,"Primary surface data belongs to a different renderer."};
        if(impl_->device==c.device && impl_->pipeline==pipeline &&
           impl_->generation==c.scene_generation && impl_->shader.impl())return {};
        Kernel extract=[&](Var<GuideArguments> p){
            const Uint index=dispatch_id();
            auto& sampler=pipeline->renderer().sampler();
            sampler->load_data();
            sampler->set_seed(make_uint2(index%p.width,index/p.width),p.frame,0u);
            auto ray_data=p.rays.read(index);
            RayVar ray=ray_data->to_ray();
            const Float3 origin=ray->origin();
            const Float3 direction=ray->direction();
            TriangleHitVar hit=p.hits.read(index);
            Float4 albedo_value = make_float4(1.f);
            Float4 normal_value = make_float4(0.f);
            Float depth_value = std::numeric_limits<float>::infinity();
            Uint validity_value = 0u;
            Uint instance_value = InvalidUI32;
            Uint primitive_value = InvalidUI32;
            Uint material_value = InvalidUI32;
            Float4 world_velocity_value = make_float4(0.f);
            Uint motion_validity_value = 0u;
            Uint surface_flags_value = SourceSurfaceFlagNone;
            Float diffuse_factor_value = 1.f;
            Float roughness_value = 1.f;
            Float reactivity_value = 0.f;
            $if(hit->is_hit()) {
                Interaction interaction = pipeline->geometry()
                    .compute_surface_interaction(hit, ray, true);
                Float3 geometric_normal = normalize(interaction.ng);
                geometric_normal = select(
                    dot(geometric_normal, -direction) < 0.f,
                    -geometric_normal,
                    geometric_normal);
                depth_value = max(length(interaction.pos - origin), 0.f);
                normal_value = make_float4(geometric_normal, 0.f);
                validity_value = 1u;
                instance_value = hit.inst_id;
                primitive_value = hit.prim_id;
                material_value = interaction.material_id();
                surface_flags_value = select(
                    interaction.has_emission(),
                    static_cast<uint>(SourceSurfaceFlagEmissive),
                    0u);

                $if(interaction.has_material()) {
                    TSpectrum spectrum = pipeline->renderer().spectrum();
                    SampledWavelengths wavelengths{spectrum->dimension()};
                    pipeline->scene().materials().dispatch(
                        interaction.material_id(),
                        [&](const Material *material) {
                            MaterialEvaluator evaluator = material->create_evaluator(
                                interaction, wavelengths);
                            albedo_value = make_float4(
                                spectrum->linear_srgb(
                                    evaluator.albedo(interaction.wo), wavelengths),
                                1.f);
                            const Uint bxdf_flags = evaluator.flag();
                            surface_flags_value = surface_flags_value |
                                select(
                                    BxDFFlag::is_diffuse(bxdf_flags),
                                    static_cast<uint>(SourceSurfaceFlagDiffuse),
                                    0u) |
                                select(
                                    BxDFFlag::is_glossy(bxdf_flags),
                                    static_cast<uint>(SourceSurfaceFlagGlossy),
                                    0u) |
                                select(
                                    (bxdf_flags & BxDFFlag::NearSpec) != 0u ||
                                        BxDFFlag::is_specular(bxdf_flags),
                                    static_cast<uint>(SourceSurfaceFlagNearSpecular),
                                    0u) |
                                select(
                                    BxDFFlag::is_transmission(bxdf_flags),
                                    static_cast<uint>(SourceSurfaceFlagTransmission),
                                    0u);
                            diffuse_factor_value = clamp(
                                evaluator.diffuse_factor(), 0.f, 1.f);
                            roughness_value = clamp(
                                evaluator.roughness(), 0.f, 1.f);
                            reactivity_value = max(
                                1.f - roughness_value,
                                max(
                                    select(
                                        (bxdf_flags & BxDFFlag::NearSpec) != 0u ||
                                            BxDFFlag::is_specular(bxdf_flags),
                                        1.f,
                                        0.f),
                                    max(
                                        select(
                                            BxDFFlag::is_transmission(bxdf_flags),
                                            0.75f,
                                            0.f),
                                        select(
                                            interaction.has_emission(),
                                            1.f,
                                            0.f))));
                        });
                };
            };
            const auto pack_oct16 = [](const Float3 &normal) {
                const Float denominator = max(
                    abs(normal.x) + abs(normal.y) + abs(normal.z),
                    1.0e-20f);
                const Float2 projected = normal.xy() / denominator;
                const Float sign_x = select(
                    projected.x >= 0.0f, 1.0f, -1.0f);
                const Float sign_y = select(
                    projected.y >= 0.0f, 1.0f, -1.0f);
                const Float2 folded = make_float2(
                    (1.0f - abs(projected.y)) * sign_x,
                    (1.0f - abs(projected.x)) * sign_y);
                const Float2 encoded = make_float2(
                    select(normal.z >= 0.0f, projected.x, folded.x),
                    select(normal.z >= 0.0f, projected.y, folded.y));
                const Uint x = cast<uint>(clamp(
                    round((encoded.x * 0.5f + 0.5f) * 65535.0f),
                    0.0f,
                    65535.0f));
                const Uint y = cast<uint>(clamp(
                    round((encoded.y * 0.5f + 0.5f) * 65535.0f),
                    0.0f,
                    65535.0f));
                return x | (y << 16u);
            };
            Var<source::PackedSurfaceObservationV1> observation;
            const Ulong invalid_key =
                (cast<ulong>(0xffffffffu) << 32u) |
                cast<ulong>(0xffffffffu);
            const Ulong surface_key =
                (cast<ulong>(instance_value) << 32u) |
                cast<ulong>(material_value);
            const Uint domain = select(
                validity_value != 0u,
                static_cast<uint>(source::PackedSurfaceDomain::Surface),
                static_cast<uint>(source::PackedSurfaceDomain::Environment));
            const Uint roughness = cast<uint>(clamp(
                round(roughness_value * 255.0f), 0.0f, 255.0f));
            const Uint reactivity = cast<uint>(clamp(
                round(reactivity_value * 255.0f), 0.0f, 255.0f));
            observation.surface_key = select(
                validity_value != 0u, surface_key, invalid_key);
            observation.hit_distance = depth_value;
            observation.packed_geometric_normal_oct16 = select(
                validity_value != 0u,
                pack_oct16(normal_value.xyz()),
                0u);
            observation.primitive_id = primitive_value;
            observation.metadata = domain |
                ((surface_flags_value & source::PackedSurfaceFlagsMask) <<
                 source::PackedSurfaceFlagsShift) |
                (roughness << source::PackedSurfaceRoughnessShift) |
                (reactivity << source::PackedSurfaceReactivityShift);
            Var<source::PackedHalf4> packed_albedo;
            const Uint packed_albedo_xy = as<uint>(cast<half2>(
                make_float2(albedo_value.x, albedo_value.y)));
            const Uint packed_albedo_z_diffuse = as<uint>(cast<half2>(
                make_float2(albedo_value.z, diffuse_factor_value)));
            packed_albedo.bits = cast<ulong>(packed_albedo_xy) |
                (cast<ulong>(packed_albedo_z_diffuse) << 32u);
            observation.linear_albedo_rgb_diffuse_factor = packed_albedo;
            p.output.write(index, observation);
        };
        impl_->shader=c.device->compile(extract,"lfs_visible_surface_from_primary");
        impl_->device=c.device;impl_->pipeline=pipeline;impl_->generation=c.scene_generation;
        return {};
    } catch(const std::exception& e) { return {backend::ErrorCode::BackendFailure,e.what()}; }
}
backend::Result<ocarina::BufferView<backend::PackedSurfaceObservationV1>>
VisibleSurface::encode(const backend::FrameContext& c,ocarina::BufferView<ocarina::float4> radiance,
                       ocarina::CommandBatch& commands) noexcept {
    using Result=backend::Result<ocarina::BufferView<backend::PackedSurfaceObservationV1>>;
    auto* pipeline=Global::instance().pipeline();
    if(!pipeline || pipeline!=impl_->pipeline || c.device!=impl_->device ||
       c.scene_generation!=impl_->generation || !impl_->shader.impl())
        return Result::failure({backend::ErrorCode::InvalidState,"Visible surface kernel is not bound to this scene."});
    auto* fb=pipeline->frame_buffer();
    const auto raw=fb->active_primary_radiance_view();
    const auto hit=fb->active_primary_visibility_view(pipeline->frame_index());
    const auto rays=fb->active_primary_rays_view();
    if(raw.handle()!=radiance.handle() || raw.offset_in_byte()!=radiance.offset_in_byte() ||
        raw.size()!=c.sample_extent.area() || hit.size()!=raw.size() || rays.size()!=raw.size())
        return Result::failure({backend::ErrorCode::ResourceMismatch,"Primary hit or ray data does not match the current radiance."});
    static_assert(sizeof(vision::RayData)==sizeof(backend::PackedSurfaceObservationV1));
    auto out=rays.view_as<backend::PackedSurfaceObservationV1>();
    GuideArguments args{rays.descriptor(),hit.descriptor(),out.descriptor(),c.sample_extent.width,pipeline->frame_index()};
    commands << impl_->shader(args).dispatch(unsigned(raw.size()));
    return Result::success(out);
}
} // namespace lfs::denoise::resident
