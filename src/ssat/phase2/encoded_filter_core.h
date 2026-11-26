#pragma once

#include "backend/frame_contract.h"
#include "ssat/denoise_types.h"
#include "ssat/phase1/phase_frame_state.h"

#include <array>
#include <cstdint>
#include <memory>
#include <functional>
#include <vector>

namespace lfs::denoise {

struct EncodedConfig {
    float sigma_luminance{32.0f};
    float sigma_normal{32.0f};
    float sigma_depth{1.0f};
    float sigma_albedo{4.0f};
    float energy_compensation{1.04f};
    uint32_t atrous_iterations{3u};
    bool angular_compatibility_spatial{};
    bool angular_compatibility_prefilter_only{};
    bool responsive_angular_radiometry{};
    bool angular_compatibility_aperture_only{};
    bool responsive_radiometry_glossy_only{};
    bool operator==(const EncodedConfig &) const = default;
};

struct EncodedTelemetry {
    backend::SampleDomain domain{backend::SampleDomain::ImagePixelGrid};
    backend::Extent2D extent{};
    uint64_t frame_index{};
    uint64_t layout_generation{};
    uint64_t layout_hash{};
    uint32_t reset_count{};
    uint32_t atrous_iterations{};
    uint32_t allocation_count{};
    uint64_t internal_history_bytes{};
    bool prepared{};
    bool history_valid_before{};
    bool temporal_used{};
    bool output_ready{};
    bool internal_radiance_fp16{};
    bool spatial_init_step1_fused{};
};

class EncodedFilterCore final : public backend::FrameProcessor {
private:
    struct Impl;
    std::unique_ptr<Impl> impl_{};

public:
    using InterstageProcessor = std::function<backend::Error(
        const backend::FrameContext &, const backend::FrameResources &,
        ocarina::BufferView<ocarina::half4>, ocarina::BufferView<ocarina::uint>, std::uint64_t &, std::uint64_t &, std::uint64_t &, std::uint64_t &,
        std::uint64_t, std::uint64_t,
        RegularR2Coverage, ocarina::CommandBatch &)>;
    void set_interstage_processor(InterstageProcessor processor);
    explicit EncodedFilterCore(EncodedConfig config = {});
    EncodedFilterCore(EncodedFilterCore &&) noexcept;
    EncodedFilterCore &operator=(EncodedFilterCore &&) noexcept;
    EncodedFilterCore(const EncodedFilterCore &) = delete;
    EncodedFilterCore &operator=(const EncodedFilterCore &) = delete;
    ~EncodedFilterCore() noexcept override;

    backend::Error reset(backend::ResetReason reason) noexcept override;
    backend::Error prepare(const backend::FrameContext &context) noexcept override;
    backend::Error enqueue_commit(
        const backend::FrameContext &context,
        ocarina::CommandBatch &commands) noexcept override;
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
    backend::Error validate_submission(
        const backend::FrameContext &context) noexcept override;

    [[nodiscard]] const EncodedConfig &config() const noexcept;
    [[nodiscard]] const EncodedTelemetry &telemetry() const noexcept;
    void invalidate_demodulated_current_view() noexcept;
    [[nodiscard]] bool set_phase_frame_state(
        const lightfield::PhaseFrameState &state) noexcept;
    void invalidate_phase_frame_state() noexcept;
    [[nodiscard]] ocarina::BufferView<ocarina::half4>
    demodulated_output_view() const noexcept;
    [[nodiscard]] ocarina::BufferView<ocarina::float4> canonical_output_view() const noexcept;
    [[nodiscard]] ocarina::BufferView<ocarina::uint>
    frame_status_view() const noexcept;
};

} // namespace lfs::denoise
