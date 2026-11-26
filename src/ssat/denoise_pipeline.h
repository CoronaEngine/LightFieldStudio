#pragma once

#include "ssat/denoise_types.h"
#include "ssat/phase1/phase_frame_state.h"

#include <memory>
#include <vector>

namespace lfs::denoise {

class ISpatialDenoiser : public backend::FrameProcessor {
public:
    ~ISpatialDenoiser() noexcept override = default;

    [[nodiscard]] virtual SpatialMethod method() const noexcept = 0;
    [[nodiscard]] virtual SignalDomain input_domain() const noexcept = 0;
    [[nodiscard]] virtual SignalDomain output_domain() const noexcept = 0;
    [[nodiscard]] virtual const ModuleTelemetry &module_telemetry() const noexcept = 0;

};

class IResolvedCurrentConsumer {
public:
    virtual ~IResolvedCurrentConsumer() noexcept = default;
    [[nodiscard]] virtual bool set_resolved_current_view(
        const ResolvedCurrentView &view) noexcept = 0;
    virtual void invalidate_resolved_current_view() noexcept = 0;
};

class IResolvedCurrentProvider {
public:
    virtual ~IResolvedCurrentProvider() noexcept = default;
    [[nodiscard]] virtual const ResolvedCurrentView *
    resolved_current_view() const noexcept = 0;
};

class ICompositionSourceProvider {
public:
    virtual ~ICompositionSourceProvider() noexcept = default;
    [[nodiscard]] virtual const CompositionView *
    composition_source_view() const noexcept = 0;
};

class ITemporalDenoiser : public backend::FrameProcessor {
public:
    ~ITemporalDenoiser() noexcept override = default;
    // Optional frame-local display scratch owned by the composer. History
    // remains owned by Temporal; the composer authorizes its canonical writer.

    [[nodiscard]] virtual TemporalMethod method() const noexcept = 0;
    [[nodiscard]] virtual SignalDomain input_domain() const noexcept = 0;
    [[nodiscard]] virtual SignalDomain output_domain() const noexcept = 0;
    [[nodiscard]] virtual const ModuleTelemetry &module_telemetry() const noexcept = 0;
    // Native Spatial handoff normally replaces raw Radiance. Carrier-domain
    // temporal filters still need the original 1 spp sample as the only new
    // evidence; the Spatial value is a prior, not another measurement.
    [[nodiscard]] virtual bool
    requires_raw_radiance_with_native_current() const noexcept {
        return false;
    }
};

struct PipelineConfig {
    Lane lane{Lane::Quality};
};

class DenoisePipeline final :
    public backend::FrameProcessor,
    public lightfield::PhaseFrameStateConsumer {
private:
    struct Impl;
    std::unique_ptr<Impl> impl_{};

public:
    DenoisePipeline(
        PipelineConfig config,
        std::unique_ptr<ISpatialDenoiser> spatial,
        std::unique_ptr<ITemporalDenoiser> temporal);
    DenoisePipeline(DenoisePipeline &&) noexcept;
    DenoisePipeline &operator=(DenoisePipeline &&) noexcept;
    DenoisePipeline(const DenoisePipeline &) = delete;
    DenoisePipeline &operator=(const DenoisePipeline &) = delete;
    ~DenoisePipeline() noexcept override;

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

    [[nodiscard]] const PipelineConfig &config() const noexcept;
    [[nodiscard]] const ISpatialDenoiser *spatial() const noexcept;
    [[nodiscard]] ITemporalDenoiser *temporal() noexcept;
    [[nodiscard]] const ITemporalDenoiser *temporal() const noexcept;
    [[nodiscard]] const ModuleTelemetry &composer_telemetry() const noexcept;
};

} // namespace lfs::denoise
