#pragma once

#include "ssat/phase2/encoded_filter_core.h"
#include "ssat/denoise_pipeline.h"
#include "ssat/phase1/phase_frame_state.h"

namespace lfs::denoise {

class EncodedAtrousSpatial final :
    public ISpatialDenoiser,
    public IResolvedCurrentProvider,
    public ICompositionSourceProvider,
    public lightfield::PhaseFrameStateConsumer {
private:
    EncodedFilterCore processor_;
    ModuleTelemetry telemetry_{"EncodedAtrousSpatial"};
    ResolvedCurrentView current_view_{};
    CompositionView composition_view_{};

public:
    void set_interstage_processor(EncodedFilterCore::InterstageProcessor processor) {
        processor_.set_interstage_processor(std::move(processor));
    }
    explicit EncodedAtrousSpatial(EncodedConfig config = {});
    [[nodiscard]] const EncodedConfig &configuration() const noexcept { return processor_.config(); }

    [[nodiscard]] SpatialMethod method() const noexcept override;
    [[nodiscard]] SignalDomain input_domain() const noexcept override;
    [[nodiscard]] SignalDomain output_domain() const noexcept override;
    [[nodiscard]] const ModuleTelemetry &module_telemetry() const noexcept override;
    [[nodiscard]] const ResolvedCurrentView *
    resolved_current_view() const noexcept override;
    [[nodiscard]] const CompositionView *
    composition_source_view() const noexcept override;
    [[nodiscard]] bool set_phase_frame_state(
        const lightfield::PhaseFrameState &state) noexcept override;
    void invalidate_phase_frame_state() noexcept override;

    backend::Error reset(backend::ResetReason reason) noexcept override;
    backend::Error prepare(const backend::FrameContext &context) noexcept override;
    backend::Error enqueue_commit(const backend::FrameContext &context,
        ocarina::CommandBatch &commands) noexcept override;
    void finalize_success(const backend::FrameContext &context) noexcept override;
    void abort_submission(const backend::FrameContext &context) noexcept override;
    backend::Error validate_submission(
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
