#pragma once

#include "backend/frame_contract.h"
#include "ssat/native/encoded_carrier_composer_native_params.h"

#include "rhi/resources/external_compute.h"

#include <filesystem>

namespace lfs::denoise::native {

class EncodedCarrierComposerRuntime {
private:
    ocarina::Device *device_{};
    ocarina::ExternalComputeModule module_{};
    ocarina::ExternalComputeKernel compose_demodulated_half4_{},guard_canonical_{};
    ocarina::SP<ocarina::ExternalComputeStatus> pending_status_{};

    [[nodiscard]] backend::Error dispatch(
        const ocarina::ExternalComputeKernel &kernel,
        const EncodedCarrierComposeParams &params,
        ocarina::CommandBatch &commands) noexcept;

public:
    [[nodiscard]] backend::Error initialize(ocarina::Device &device) noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] backend::Error dispatch_demodulated_half4(
        const EncodedCarrierComposeParams &params,
        ocarina::CommandBatch &commands) noexcept;
    [[nodiscard]] backend::Error dispatch_canonical_guard(const EncodedCarrierComposeParams &params,ocarina::CommandBatch &commands) noexcept;
    [[nodiscard]] backend::Error validate_submission() const noexcept;
    void clear_submission() noexcept;
};

[[nodiscard]] std::filesystem::path encoded_carrier_composer_fatbin_path();

} // namespace lfs::denoise::native
