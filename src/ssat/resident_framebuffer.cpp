#include "base/mgr/pipeline.h"
#include "base/sensor/frame_buffer.h"
#include "rhi/resources/shader.h"

namespace lfs::denoise::resident {

using namespace ocarina;
using namespace vision;

class RadianceAovFramebuffer final : public FrameBuffer {
    bool external_active_{};
    bool borrowed_aovs_{};

    void borrow(RegistrableBuffer<float4> &aov, BufferView<float4> owner) {
        if (aov.handle() == owner.handle() && aov.size() == owner.size()) return;
        aov.set_bindless_array(bindless_array());
        aov.update_buffer(Buffer<float4>(owner));
    }

    bool configure_aovs(bool share_before_trace) {
        const auto count = frame_buffer_size();
        const auto raw = active_primary_radiance_view();
        if (!count || raw.size() != count || !raw.handle() || raw.offset_in_byte() != 0) return false;

        if (share_before_trace) {
            // The external GBuffer emits three same-index FLOAT4 stores. Its
            // consumer reads only the separate original Packed32. The following
            // PT launch covers every slot and overwrites all four raw components.
            borrow(albedo_aov_, raw);
            borrow(normal_aov_, raw);
            borrow(depth_aov_, raw);
        } else {
            if (borrowed_aovs_ || albedo_aov_.size() != count) prepare_albedo_aov();
            if (borrowed_aovs_ || normal_aov_.size() != count) prepare_normal_aov();
            if (borrowed_aovs_ || depth_aov_.size() != count) prepare_depth_aov();
        }
        borrowed_aovs_ = share_before_trace;
        const auto albedo = albedo_aov_.descriptor();
        const auto normal = normal_aov_.descriptor();
        const auto depth = depth_aov_.descriptor();
        const auto raw_address = raw.descriptor().handle;
        if (!albedo.handle || !normal.handle || !depth.handle ||
            albedo.size != count || normal.size != count || depth.size != count) return false;
        if (share_before_trace) {
            return albedo.handle == raw_address && normal.handle == raw_address && depth.handle == raw_address;
        }
        return albedo.handle != normal.handle && albedo.handle != depth.handle && normal.handle != depth.handle &&
               albedo.handle != raw_address && normal.handle != raw_address && depth.handle != raw_address;
    }

public:
    explicit RadianceAovFramebuffer(const FrameBufferDesc &desc) : FrameBuffer(desc) {}
    RadianceAovFramebuffer() = default;
    VS_MAKE_PLUGIN_NAME_FUNC

    RayState custom_generate_ray(const Sensor *sensor, const SensorSample &sample) const noexcept override {
        return sensor->generate_ray(sample);
    }

    void prepare() noexcept override {
        encode_data();
        datas().reset_device_buffer_immediately(device(), "ssat.framebuffer.encoded_data");
        datas().register_self();
        datas().upload_immediately();
        init_hit_buffer();
        prepare_rt_buffer();
        prepare_rays();
    }

    CommandBatch gamma_correct() const noexcept override {
        return external_active_ ? CommandBatch{} : FrameBuffer::gamma_correct();
    }
    CommandBatch post_path_tracing(uint frame) const noexcept override {
        return external_active_ ? CommandBatch{} : FrameBuffer::post_path_tracing(frame);
    }
    CommandBatch render_final(uint frame) const noexcept override {
        return external_active_ ? CommandBatch{} : FrameBuffer::render_final(frame);
    }

    bool activate(bool ordered, bool share_before_trace) {
        if (!configure_aovs(share_before_trace)) return false;
        external_active_ = true;
        view_texture_ = {};
        destroy_motion_vectors();
        destroy_indirect_lighting();
        auto_manage_direct_lighting(ordered);
        return true;
    }
};

} // namespace lfs::denoise::resident

VS_MAKE_CLASS_CREATOR_HOTFIX(lfs::denoise::resident, RadianceAovFramebuffer)

extern "C" __declspec(dllexport) bool lfs_activate_declared_native_framebuffer(
    vision::FrameBuffer *object, bool ordered, bool share_before_trace) {
    auto *framebuffer = dynamic_cast<lfs::denoise::resident::RadianceAovFramebuffer *>(object);
    return framebuffer && framebuffer->activate(ordered, share_before_trace);
}
