#pragma once

#include "backend/frame_contract.h"
#include "ssat/phase1/phase_frame_state.h"
#include "lightfield_framebuffer.h"

#include <memory>

namespace lfs::lightfield {

struct LightFieldTelemetry {
    backend::Extent2D sample_extent{};
    backend::Extent2D output_extent{};
    uint64_t frame_index{};
    uint64_t layout_generation{};
    uint64_t layout_hash{};
    uint64_t ray_bank_device_bytes{};
    uint64_t motion_device_bytes{};
    uint64_t canonical_present_device_bytes{};
    uint64_t static_lf_table_device_bytes{};
    uint64_t total_owned_device_bytes{};
    uint64_t calibration_readback_bytes{};
    uint32_t calibration_synchronizations{};
    uint64_t dense_target_count{};
    uint64_t submitted_ray_count{};
    uint64_t radiance_write_count{};
    uint64_t allocation_generation{};
    uint32_t allocation_count{};
    uint32_t reset_count{};
    bool prepared{};
    bool rays_generated{};
    bool factorized_primary_ray_source{};
    bool motion_generated{};
    bool history_valid_before{};
    bool canonical_present_ready{};
    bool srgb_transfer_applied{};
};

class LightFieldProcessor final : public backend::FrameProcessor {
private:
    struct Impl;
    std::unique_ptr<Impl> impl_{};

public:
    LightFieldProcessor(
        LenticularParams lenticular,
        LightFieldGeometryParams geometry,
        backend::FrameProcessor *denoiser = nullptr);
    LightFieldProcessor(LightFieldProcessor &&) noexcept;
    LightFieldProcessor &operator=(LightFieldProcessor &&) noexcept;
    LightFieldProcessor(const LightFieldProcessor &) = delete;
    LightFieldProcessor &operator=(const LightFieldProcessor &) = delete;
    ~LightFieldProcessor() noexcept override;

    void set_camera(const backend::CameraState &camera) noexcept;

    backend::Error reset(backend::ResetReason reason) noexcept override;
    backend::Error prepare(const backend::FrameContext &context) noexcept override;
    backend::Error enqueue_commit(
        const backend::FrameContext &context,
        ocarina::CommandBatch &commands) noexcept override;
    backend::Error validate_submission(
        const backend::FrameContext &context) noexcept override;
    void finalize_success(
        const backend::FrameContext &context) noexcept override;
    void abort_submission(
        const backend::FrameContext &context) noexcept override;
    backend::Result<backend::ResourceRequirementSet> resource_requirements(
        const backend::FrameContext &context) noexcept override;
    backend::Result<backend::PrimaryRayInput> primary_rays(
        const backend::FrameContext &context,
        ocarina::CommandBatch &commands) noexcept override;
    backend::Result<backend::ProcessOutput> process(
        const backend::FrameContext &context,
        const backend::FrameResources &resources,
        ocarina::CommandBatch &commands) noexcept override;
    backend::Result<backend::PresentOutput> present(
        const backend::FrameContext &context,
        const backend::FrameResources &resources,
        const backend::ProcessOutput &output,
        ocarina::CommandBatch &commands) noexcept override;

    [[nodiscard]] const LightFieldTelemetry &telemetry() const noexcept;
};

} // namespace lfs::lightfield
