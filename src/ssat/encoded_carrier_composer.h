#pragma once

#include "ssat/denoise_pipeline.h"

#include <memory>

namespace lfs::denoise {

// Owns ordinary composition or validates a borrowed fused canonical output.
// Canonical ownership is explicit; it never forms a recursive color history.
class EncodedCarrierComposer final {
private:
    struct Impl;
    std::unique_ptr<Impl> impl_{};

public:
    explicit EncodedCarrierComposer(bool borrow_canonical=false);
    EncodedCarrierComposer(EncodedCarrierComposer &&) noexcept;
    EncodedCarrierComposer &operator=(EncodedCarrierComposer &&) noexcept;
    EncodedCarrierComposer(const EncodedCarrierComposer &) = delete;
    EncodedCarrierComposer &operator=(const EncodedCarrierComposer &) = delete;
    ~EncodedCarrierComposer() noexcept;

    backend::Error reset() noexcept;
    backend::Error prepare(const backend::FrameContext &context) noexcept;
    backend::Error validate_submission() noexcept;
    void clear_submission() noexcept;

    [[nodiscard]] ocarina::BufferView<ocarina::float4>
    canonical_view() const noexcept;
    [[nodiscard]] const ModuleTelemetry &telemetry() const noexcept;

    backend::Result<backend::PresentOutput> compose_native(
        const backend::FrameContext &context,
        const backend::FrameResources &resources,
        const CompositionView &source,
        ocarina::CommandBatch &commands) noexcept;
};

} // namespace lfs::denoise
