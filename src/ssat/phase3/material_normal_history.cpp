#include "ssat/phase3/material_normal_history.h"
#include "base/mgr/global.h"
#include "base/mgr/pipeline.h"
#include "base/scattering/material.h"
#include "base/sensor/frame_buffer.h"
#include "rhi/resources/shader.h"
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace lfs::denoise {
using namespace ocarina;
struct MaterialNormalArguments {
    BufferDesc<vision::RayData> rays;
    BufferDesc<TriangleHit> hits;
    BufferDesc<backend::PackedSurfaceObservationV1> geometry;
    BufferDesc<uint> normals;
    unsigned width{},frame{};
};
struct MaterialTransformArguments { BufferDesc<float4> rows; };
struct MaterialInterfaceArguments { BufferDesc<uint> indices; uint first; };
}
OC_PARAM_STRUCT(lfs::denoise,MaterialNormalArguments,rays,hits,geometry,normals,width,frame) {};
OC_PARAM_STRUCT(lfs::denoise,MaterialTransformArguments,rows) {};
OC_PARAM_STRUCT(lfs::denoise,MaterialInterfaceArguments,indices,first) {};

namespace lfs::denoise {
using namespace ocarina;
using namespace vision;
namespace {
constexpr unsigned kUnknownNormal=0xffffffffu;
Uint encode_normal(const Float3 &value) {
    const Float3 n=normalize(value);
    const Float2 uv=n.xy()/max(abs(n.x)+abs(n.y)+abs(n.z),1.e-20f);
    const Float2 folded=make_float2((1.f-abs(uv.y))*select(uv.x>=0.f,1.f,-1.f),
                                    (1.f-abs(uv.x))*select(uv.y>=0.f,1.f,-1.f));
    const Float2 oct=select(n.z>=0.f,uv,folded);
    const Uint a=cast<uint>(clamp(round((oct.x*.5f+.5f)*65535.f),0.f,65535.f));
    const Uint b=cast<uint>(clamp(round((oct.y*.5f+.5f)*65535.f),0.f,65535.f));
    const Uint bits=a|(b<<16u);
    // All four octahedral corners represent the south pole.
    return select(bits==kUnknownNormal,0u,bits);
}


// The frozen material topology includes every bound slot; a bound normal
// contributes an extra ShaderNodeSlot. Compare against a normal-free graph
// constructed by the same plugin. Unrecognized or heterogeneous graphs use
// the complete lobe evaluator. The witnesses are private, unregistered nodes.
bool has_only_vertex_frames(Pipeline *pipeline) {
    std::map<std::string,std::set<uint64_t>> witnesses;
    bool all=true;
    pipeline->scene().materials().for_each_instance([&](const SP<Material> &material) {
        if(!all)return;
        const std::string type{material->impl_type()};
        // The frozen diffuse lobe cannot request a directional normal. The
        // black-body plugin has no shading-frame slots at all. Their textures
        // cannot require separate evaluation of a different material's Ns.
        if(type=="diffuse" || type=="black_body")return;
        if(type!="metal" && type!="mirror" && type!="glass" && type!="substrate" &&
           type!="diffuse" && type!="plastic" && type!="black_body") {
            all=false;return;
        }
        auto [it,first]=witnesses.try_emplace(type);
        if(first) {
            // Both frozen scalar and anisotropic roughness layouts can use
            // the same untextured vertex frame.
            for(bool two_values:{false,true}) {
                MaterialDesc desc(type);
                desc.name="Studio vertex frame witness";
                desc.set_value("color",DataWrap::array({.5f,.5f,.5f}));
                if(two_values)desc.set_value("roughness",DataWrap::array({.1f,.1f}));
                else desc.set_value("roughness",.1f);
                if(type=="metal")desc.set_value("material_name","Al");
                const auto witness=Material::create_root(desc);
                it->second.insert(witness->topology_hash());
            }
        }
        all=it->second.contains(material->topology_hash());
    });
    return all;
}

Uint vertex_frame_normal(Geometry &geometry,const TriangleHitVar &hit,const Uint &material_id) {
    auto inst=geometry.data()->instances().read(hit.inst_id);
    Uint result=kUnknownNormal;
    $if(inst.mat_id==material_id && inst.mat_id!=InvalidUI32) {
        auto mesh=geometry.data()->mesh_handles().read(inst.mesh_id);
        auto tri=geometry.get_triangle(mesh.triangle_buffer,hit.prim_id);
        auto [v0,v1,v2]=geometry.get_vertices(mesh.vertex_buffer,tri);
        auto object_to_world=Transform(inst->o2w());
        Float3 n=hit->triangle_lerp(v0->normal(),v1->normal(),v2->normal());
        $if(!is_zero(n)) {
            result=encode_normal(normalize(object_to_world.apply_normal(n)));
        } $else {
            Float3 p0=object_to_world.apply_point(v0->position());
            Float3 p1=object_to_world.apply_point(v1->position());
            Float3 p2=object_to_world.apply_point(v2->position());
            result=encode_normal(normalize(cross(p0-p2,p1-p2)));
        };
    };
    return result;
}
Shader<void(MaterialNormalArguments)> make_shader(Device &device,Pipeline *pipeline,bool vertex) {
    const bool fixed_spectrum=pipeline->renderer().spectrum()->impl_type()=="srgb";
    Kernel gather=[&](Var<MaterialNormalArguments> args) {
        const Uint i=dispatch_id();
        const auto observation=args.geometry.read(i);
        const Uint cls=(observation.metadata>>2u)&127u;
        Uint result=kUnknownNormal;
        $if((observation.metadata&3u)==1u && (cls==2u || cls==4u)) {
            TriangleHitVar hit=args.hits.read(i);
            $if(hit->is_hit() && hit.inst_id==cast<uint>(observation.surface_key>>32u) &&
                hit.prim_id==observation.primitive_id) {
                if(vertex) {
                    result=vertex_frame_normal(pipeline->geometry(),hit,cast<uint>(observation.surface_key));
                } else if(fixed_spectrum) {
                    RayVar ray=args.rays.read(i)->to_ray();
                    Interaction interaction=pipeline->geometry().compute_surface_interaction(hit,ray,true);
                    $if(interaction.has_material() && interaction.material_id()==cast<uint>(observation.surface_key)) {
                        const TSpectrum sp=pipeline->renderer().spectrum();
                        auto &sampler=pipeline->renderer().sampler();
                        sampler->load_data();
                        sampler->set_seed(make_uint2(i%args.width,i/args.width),args.frame,0u);
                        const auto wavelengths=sp->sample_wavelength(sampler);
                        pipeline->scene().materials().dispatch(interaction.material_id(),[&](const Material *material) {
                            auto evaluator=material->create_evaluator(interaction,wavelengths);
                            if(!MaterialRegistry::instance().individual_ns()) {
                                result=encode_normal(evaluator.shading_frame().normal());
                            } else {
                                evaluator.dispatch([&](const Lobe *lobe) {
                                    if(!lobe->is_multi()) {
                                        result=encode_normal(lobe->shading_frame().normal());
                                    } else {
                                        Uint first=kUnknownNormal;
                                        Bool consistent=true;
                                        static_cast<const LobeSet*>(lobe)->for_each([&](const WeightedLobe &child) {
                                            $if(child.weight()>0.f && child.sample_weight()>0.f) {
                                                const Uint code=encode_normal(child->shading_frame().normal());
                                                consistent=consistent && (first==kUnknownNormal || first==code);
                                                first=code;
                                            };
                                        });
                                        result=select(consistent,first,kUnknownNormal);
                                    }
                                });
                            }
                        });
                    };
                }
            };
        };
        args.normals.write(i,result);
    };
    return device.compile(gather,vertex?"lfs_vertex_bsdf_normal_from_primary":"lfs_committed_bsdf_normal_from_primary");
}
template<typename T> std::uint64_t address(BufferView<T> view) {
    return std::uint64_t(view.handle())+view.offset_in_byte();
}

Shader<void(MaterialInterfaceArguments)> make_interface_shader(Device &device,Pipeline *pipeline) {
    std::set<uint64_t> uniform_glass;
    bool has_glass=false;
    pipeline->scene().materials().for_each_instance([&](const SP<Material> &m) {
        has_glass=has_glass || std::string{m->impl_type()}=="glass";
    });
    if(has_glass && pipeline->renderer().spectrum()->impl_type()=="srgb") {
        // A numeric IOR has exactly this plugin's constant graph layout.
        // A texture, spectral curve or unrecognized graph gets no single-axis claim.
        for(bool two:{false,true}) {
            MaterialDesc desc("glass");desc.name="Studio constant glass interface";
            desc.set_value("color",DataWrap::array({.5f,.5f,.5f}));
            if(two)desc.set_value("roughness",DataWrap::array({.1f,.1f}));
            else desc.set_value("roughness",.1f);
            uniform_glass.insert(Material::create_root(desc)->topology_hash());
        }
    }
    Kernel kernel=[&](Var<MaterialInterfaceArguments> args) {
        const Uint i=dispatch_id();
        const auto instance=pipeline->geometry().data()->instances().read(i);
        Float eta=0.f;
        $if(instance.mat_id!=InvalidUI32) {
            pipeline->scene().materials().dispatch(instance.mat_id,[&](const Material *m) {
                if(std::string{m->impl_type()}!="glass")return;
                if(!uniform_glass.contains(m->topology_hash())) {eta=-1.f;return;}
                // The supported IOR is independent of position, UV and wavelength.
                // A front-facing frame retrieves the actual plugin value before its
                // inside/outside conversion; the consuming actual ray supplies that sign.
                Interaction it(false);
                it.pos=make_float3(0.f);it.uv=make_float2(0.f);
                it.wo=make_float3(0.f,0.f,1.f);it.ng=it.wo;it.ng_local=it.wo;
                it.shading.set(make_float3(1.f,0.f,0.f),make_float3(0.f,1.f,0.f),it.ng);
                const auto sp=pipeline->renderer().spectrum();
                auto &sampler=pipeline->renderer().sampler();sampler->load_data();
                sampler->set_seed(make_uint2(i,0u),0u,0u);
                const auto wavelengths=sp->sample_wavelength(sampler);
                auto material=m->create_evaluator(it,wavelengths);
                // The frozen DielectricLobe inherits the unit default for
                // relative_eta(). Its public Z coordinate is encoded directly
                // from the Fresnel spectrum that the glass BSDF samples.
                material.dispatch([&](const Lobe *lobe) {
                    const Float z=lobe->to_ratio_z();
                    eta=DielectricLobe::ior_lower+z*(DielectricLobe::ior_upper-DielectricLobe::ior_lower);
                });
            });
        };
        args.indices.write(args.first+i,as<uint>(eta));
    };
    return device.compile(kernel,"lfs_primary_material_interface");
}

Shader<void(MaterialTransformArguments)> make_transform_shader(Device &device,Pipeline *pipeline) {
    Kernel kernel=[&](Var<MaterialTransformArguments> args) {
        const Uint i=dispatch_id();
        auto instance=pipeline->geometry().data()->instances().read(i);
        const auto forward=instance->o2w();
        const auto normal=transpose(inverse(forward));
        for(uint j=0;j<3;++j) {
            args.rows.write(i*6u+j,make_float4(forward[0][j],forward[1][j],forward[2][j],forward[3][j]));
            args.rows.write(i*6u+3u+j,make_float4(normal[0][j],normal[1][j],normal[2][j],normal[3][j]));
        }
    };
    return device.compile(kernel,"lfs_material_normal_instance_transforms");
}
}

struct MaterialNormalHistory::Impl {
    Pipeline *pipeline{};Device *device{};
    std::uint64_t scene{},layout{},layout_hash{};
    std::array<Buffer<uint>,2> normals{};
    int committed{-1},pending{-1};
    Shader<void(MaterialNormalArguments)> shader{};
    Shader<void(MaterialTransformArguments)> transform_shader{};
    Shader<void(MaterialInterfaceArguments)> interface_shader{};
    Buffer<uint> meshes{};
    Buffer<float4> transforms{};
    std::vector<mr_native::MaterialVertexMesh> table,staging,source_meshes;
    std::vector<std::byte> table_instances;
    bool simple{},table_ready{},table_pending{};
    std::vector<std::byte> committed_instances,pending_instances;
    std::vector<std::uint64_t> compiled_materials,committed_materials,pending_materials;
};
MaterialNormalHistory::MaterialNormalHistory():impl_(std::make_unique<Impl>()) {}
MaterialNormalHistory::~MaterialNormalHistory()=default;
void MaterialNormalHistory::reset() noexcept {impl_->committed=impl_->pending=-1;}
void MaterialNormalHistory::abort() noexcept {
    impl_->pending=-1;
    if(impl_->table_pending)impl_->table_ready=false;
    impl_->table_pending=false;
}
void MaterialNormalHistory::commit() noexcept {
    if(impl_->pending<0)return;
    if(impl_->table_pending)impl_->table_ready=true;
    impl_->table_pending=false;
    if(impl_->pending!=impl_->committed) {
        impl_->committed_instances.swap(impl_->pending_instances);
        impl_->committed_materials.swap(impl_->pending_materials);
    }
    impl_->committed=impl_->pending;impl_->pending=-1;
}

backend::Result<MaterialNormalViews> MaterialNormalHistory::stage(const backend::FrameContext &c,BufferView<float4> raw,
    BufferView<backend::PackedSurfaceObservationV1> packed,bool same_pose,bool history,bool may_fuse,CommandBatch &commands) noexcept {
    using Result=backend::Result<MaterialNormalViews>;
    try {
        auto *pipeline=Global::instance().pipeline();
        if(!pipeline || &pipeline->device()!=c.device || !pipeline->frame_buffer())
            return Result::failure({backend::ErrorCode::InvalidState,"Material normals require the owning renderer session."});
        auto *fb=pipeline->frame_buffer();
        const size_t count=c.sample_extent.area();
        const auto native_raw=fb->active_primary_radiance_view();
        const auto native_packed=fb->source_packed_observation_view();
        const auto hits=fb->active_primary_visibility_view(pipeline->frame_index());
        const auto rays=fb->active_primary_rays_view();
        if(raw.size()!=count || packed.size()!=count || rays.size()!=count || hits.size()!=count ||
           native_raw.size()!=count || native_packed.size()!=count || address(raw)!=address(native_raw) ||
           address(packed)!=address(native_packed))
            return Result::failure({backend::ErrorCode::ResourceMismatch,"Primary hit, ray and packed geometry belong to different frames."});
        auto &s=*impl_;
        const auto instance_count=pipeline->geometry().data()->instances().host_buffer().size();
        const bool new_owner=s.normals[0].size()!=count+instance_count || s.pipeline!=pipeline || s.device!=c.device || s.scene!=c.scene_generation ||
                             s.layout!=c.layout_generation || s.layout_hash!=c.layout_hash;
        std::vector<std::uint64_t> topology;
        s.pending_materials.clear();
        topology.push_back(MaterialRegistry::instance().individual_ns());
        topology.push_back(pipeline->renderer().spectrum()->hash());
        pipeline->scene().materials().for_each_instance([&](const SP<Material> &m) {
            topology.push_back(m->topology_hash());
            s.pending_materials.push_back(m->hash());
        });
        if(new_owner || topology!=s.compiled_materials || !s.shader.impl()) {
            s.simple=has_only_vertex_frames(pipeline);
            s.shader=make_shader(*c.device,pipeline,s.simple);
            s.interface_shader=make_interface_shader(*c.device,pipeline);
            if(s.simple)s.transform_shader=make_transform_shader(*c.device,pipeline);
            s.table_ready=s.table_pending=false;
            s.compiled_materials=std::move(topology);
            s.pipeline=pipeline;s.device=c.device;s.scene=c.scene_generation;
            s.layout=c.layout_generation;s.layout_hash=c.layout_hash;
            s.normals[0]=c.device->create_buffer<uint>(count+instance_count,"lfs.bsdf_normal.a");
            s.normals[1]=c.device->create_buffer<uint>(count+instance_count,"lfs.bsdf_normal.b");
            reset();
        }
        // These are the frozen geometry manager's authoritative host records.
        // A stationary camera may still observe a rigid edit; no GPU readback
        // is needed to detect it before reusing the last material measurement.
        const auto &instances=pipeline->geometry().data()->instances().host_buffer();
        const size_t bytes=instances.size()*sizeof(InstanceData);
        const bool previous=history && s.committed>=0;
        const bool reuse=previous && same_pose && c.geometry_motion_model!=backend::TraceRequest::GeometryMotionModel::Deforming &&
            s.committed_instances.size()==bytes && (!bytes || !std::memcmp(s.committed_instances.data(),instances.data(),bytes)) &&
            s.pending_materials==s.committed_materials;
        s.pending=reuse?s.committed:(s.committed==0?1:0);
        mr_native::MaterialVertexFrame frame{};
        if(!reuse) {
            if(instance_count) {
                // The frozen CUDA OCBuffer indexes its owning handle directly.
                // Express the tail in the actual device address calculation.
                MaterialInterfaceArguments optical{s.normals[s.pending].descriptor(),uint(count)};
                commands<<s.interface_shader(optical).dispatch(uint(instance_count));
            }
            s.pending_instances.resize(bytes);
            if(bytes)std::memcpy(s.pending_instances.data(),instances.data(),bytes);
            // The frozen OptiX primary writer uses OCBuffer<TriangleHit>[i].
            // This ABI indexes descriptor.handle[i] even when the public
            // per-frame view describes the second half of the owning buffer.
            // Borrow that exact range from its full framebuffer owner.
            const auto &visibility=fb->visibility_buffer();
            const bool owns_primary=visibility.size()>=count &&
                std::uint64_t(hits.descriptor().handle)==std::uint64_t(visibility.handle());
            if(may_fuse && s.simple && !instances.empty() && owns_primary) {
                using Row=mr_native::MaterialVertexMesh;
                static_assert(sizeof(Row)==32 && sizeof(ocarina::TriangleHit)==16 &&
                              sizeof(ocarina::Vertex)==40 && offsetof(ocarina::Vertex,n)==12 &&
                              sizeof(ocarina::Triangle)==12);
                s.source_meshes.clear();
                pipeline->geometry().data()->for_each_mesh([&](Mesh *mesh,uint i) {
                    if(i>=s.source_meshes.size())s.source_meshes.resize(i+1);
                    auto v=mesh->vertex_buffer().view();
                    auto t=mesh->triangle_buffer().view();
                    s.source_meshes[i]={address(v),address(t),uint(v.size()),uint(t.size()),0u,0u};
                });
                s.staging.clear();
                for(const auto &instance:instances) {
                    if(instance.mesh_id>=s.source_meshes.size())break;
                    auto row=s.source_meshes[instance.mesh_id];
                    row.material=instance.mat_id;
                    s.staging.push_back(row);
                }
                if(s.staging.size()==instances.size()) {
                    const bool mesh_changed=s.table.size()!=s.staging.size() ||
                        std::memcmp(s.staging.data(),s.table.data(),s.staging.size()*sizeof(Row))!=0;
                    const bool inst_changed=s.table_instances.size()!=bytes ||
                        std::memcmp(s.pending_instances.data(),s.table_instances.data(),bytes)!=0;
                    if(s.meshes.size()!=s.staging.size()*sizeof(Row)/sizeof(uint)) {
                        s.meshes=c.device->create_buffer<uint>(s.staging.size()*sizeof(Row)/sizeof(uint),"lfs.normal_meshes");
                        s.transforms=c.device->create_buffer<float4>(s.staging.size()*6u,"lfs.normal_transforms");
                        s.table_ready=false;
                    }
                    if(mesh_changed || !s.table_ready) {
                        s.table.swap(s.staging);
                        commands<<s.meshes.upload(reinterpret_cast<const uint*>(s.table.data()));
                    }
                    if(mesh_changed || inst_changed || !s.table_ready) {
                        commands<<s.transform_shader(MaterialTransformArguments{s.transforms.descriptor()}).dispatch(uint(s.table.size()));
                        s.table_instances=s.pending_instances;
                        s.table_pending=true;
                    }
                    frame={address(visibility.view(0,count)),address(s.meshes.view()),address(s.transforms.view()),uint(s.table.size()),0u};
                }
            }
            if(!frame.hits) {
                MaterialNormalArguments arguments{rays.descriptor(),hits.descriptor(),packed.descriptor(),s.normals[s.pending].descriptor(),
                                                   c.sample_extent.width,pipeline->frame_index()};
                commands<<s.shader(arguments).dispatch(uint(count));
            }
        }
        return Result::success({address(s.normals[s.pending].view()),previous?address(s.normals[s.committed].view()):0u,frame});
    } catch(const std::exception &e) {return Result::failure({backend::ErrorCode::BackendFailure,e.what()});}
}
} // namespace lfs::denoise
