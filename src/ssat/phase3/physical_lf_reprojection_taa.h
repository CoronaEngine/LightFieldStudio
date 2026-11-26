#pragma once
#include "ssat/phase3/native/early_measurement_params.h"

#include "ssat/denoise_pipeline.h"
#include "ssat/native/denoise_native_types.h"
#include "ssat/phase1/phase_frame_state.h"
#include "lightfield_framebuffer.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace lfs::denoise {

struct PhysicalLfTaaConfig {
    Lane lane{Lane::Quality};
    LenticularParams lenticular{};
    LightFieldGeometryParams geometry{};
    bool responsive_temporal_rcrs{};
    bool responsive_log_history{};
    bool response_direction_glossy_only{};
};

struct PhysicalLfTaaTelemetry {
    backend::Extent2D sample_extent{};
    backend::Extent2D output_extent{};
    uint64_t frame_index{};
    uint64_t history_bytes{};
    uint64_t canonical_bytes{};
    uint64_t allocation_generation{};
    uint32_t allocation_count{};
    uint32_t reset_count{};
    uint32_t support_radius{};
    uint32_t support_max_taps{};
    bool prepared{};
    bool phase_state_valid{};
    bool history_valid_before{};
    bool temporal_used{};
    bool output_ready{};
};

class PhysicalLfReprojectionTaa final :
    public ITemporalDenoiser,
    public IResolvedCurrentConsumer,
    public lightfield::PhaseFrameStateConsumer {
private:
    struct Impl;
    std::unique_ptr<Impl> impl_{};
    ModuleTelemetry module_telemetry_{"PhysicalLfReprojectionTAA"};

public:
    explicit PhysicalLfReprojectionTaa(PhysicalLfTaaConfig config);
    PhysicalLfReprojectionTaa(PhysicalLfReprojectionTaa &&) noexcept;
    PhysicalLfReprojectionTaa &operator=(
        PhysicalLfReprojectionTaa &&) noexcept;
    PhysicalLfReprojectionTaa(const PhysicalLfReprojectionTaa &) = delete;
    PhysicalLfReprojectionTaa &operator=(
        const PhysicalLfReprojectionTaa &) = delete;
    ~PhysicalLfReprojectionTaa() noexcept override;
    [[nodiscard]] std::uint64_t previous_measurement_geometry() const noexcept;
    [[nodiscard]] std::uint64_t current_regular_measurement_geometry() const noexcept;
    [[nodiscard]] std::uint64_t current_regular_material_normals() const noexcept;
    [[nodiscard]] std::uint64_t current_panel_samples() const noexcept;

    [[nodiscard]] TemporalMethod method() const noexcept override;
    [[nodiscard]] SignalDomain input_domain() const noexcept override;
    [[nodiscard]] SignalDomain output_domain() const noexcept override;
    [[nodiscard]] const ModuleTelemetry &module_telemetry() const noexcept override;
    [[nodiscard]] bool requires_raw_radiance_with_native_current()
        const noexcept override { return true; }
    [[nodiscard]] const PhysicalLfTaaTelemetry &telemetry() const noexcept;
    [[nodiscard]] bool set_resolved_current_view(
        const ResolvedCurrentView &view) noexcept override;
    void invalidate_resolved_current_view() noexcept override;
    [[nodiscard]] bool set_phase_frame_state(
        const lightfield::PhaseFrameState &state) noexcept override;
    void invalidate_phase_frame_state() noexcept override;

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
    backend::Result<backend::ProcessOutput> process(
        const backend::FrameContext &context,
        const backend::FrameResources &resources,
        ocarina::CommandBatch &commands) noexcept override;
    backend::Result<backend::PresentOutput> present(
        const backend::FrameContext &context,
        const backend::FrameResources &resources,
        const backend::ProcessOutput &output,
        ocarina::CommandBatch &commands) noexcept override;
};

} // namespace lfs::denoise
