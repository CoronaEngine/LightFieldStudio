#include "ssat/native_aov_policy.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "ssat/resident_storage.h"
#include "base/mgr/global.h"
#include "base/mgr/pipeline.h"
#include "base/sensor/frame_buffer.h"

namespace lfs::denoise::resident {
bool dedicated(ocarina::Device* device) noexcept {
    auto* p=vision::Global::instance().pipeline();
    return p && &p->device()==device && p->frame_buffer() && p->frame_buffer()->impl_type()=="ssat_resident";
}

std::uint64_t geometry_bank(ocarina::Device* device, std::uint64_t frame, std::size_t count) noexcept {
    if(!dedicated(device)) return 0;
    auto hit=vision::Toolkit::frame_buffer().active_primary_visibility_view(unsigned(frame));
    return hit.size()==count ? hit.handle()+hit.offset_in_byte() : 0;
}

std::uint64_t inactive_geometry_bank(ocarina::Device *device, std::size_t count) noexcept {
    if (!dedicated(device)) return 0;
    const auto *pipeline = vision::Global::instance().pipeline();
    return geometry_bank(device, pipeline->frame_index() ^ 1u, count);
}

bool native_packed_owner(ocarina::Device *device) noexcept {
    if (!dedicated(device)) return false;
    const auto plugin=GetModuleHandleW(L"vision-framebuffer-ssat_resident.dll");
    return plugin && GetProcAddress(plugin,"lfs_activate_declared_native_framebuffer");
}

TraceWorkspaces trace_workspaces(ocarina::Device *device, std::size_t count, std::uint64_t packed_surface) noexcept {
    if (!native_packed_owner(device)) return {};
    auto &framebuffer=vision::Toolkit::frame_buffer();
    const auto packed=framebuffer.source_packed_observation_view();
    const auto rays=framebuffer.active_primary_rays_view();
    const auto hits=framebuffer.visibility_buffer().view();
    static_assert(sizeof(vision::RayData)==32u && sizeof(ocarina::TriangleHit)==16u);
    if (packed.size()!=count || rays.size()!=count || hits.size()!=2ull*count ||
        packed.handle()+packed.offset_in_byte()!=packed_surface) return {};
    const auto ray_base=rays.handle()+rays.offset_in_byte(), hit_base=hits.handle()+hits.offset_in_byte();
    if (!ray_base || !hit_base || ray_base==hit_base || ray_base==packed_surface || hit_base==packed_surface) return {};
    const auto bank_bytes=std::uint64_t(count)*16u;
    return {ray_base,ray_base+bank_bytes,hit_base,hit_base+bank_bytes};
}

}
bool lfs::denoise::resident::activate_shared_aovs(ocarina::Device* device,unsigned width,unsigned height,bool ordered,
                                      const lfs::backend::ResourceRequirementSet& demand,bool separate_preview) noexcept {
    auto* p=vision::Global::instance().pipeline();
    if(!p || &p->device()!=device)return false;
    auto* f=p->frame_buffer();
    if(!f || f->raytracing_resolution().x!=width || f->raytracing_resolution().y!=height)return false;
    // Other source semantics may need their own motion or front-geometry resources.
    // Their production route uses the stock framebuffer. Independent preview AOVs
    // on this declared graph retain three separate full-size native owners.
    if(!can_share_dense_aovs(demand,false))return false;
    const auto dll=GetModuleHandleW(L"vision-framebuffer-ssat_resident.dll");
    if(!dll)return false;
    using Activate=bool(*)(vision::FrameBuffer*,bool,bool);
    const auto configure=reinterpret_cast<Activate>(GetProcAddress(dll,"lfs_activate_declared_native_framebuffer"));
    return configure && configure(f,ordered,can_share_dense_aovs(demand,separate_preview));
}
