#include "ssat/runtime_paths.h"
#include "ssat/native/encoded_carrier_composer_native_runtime.h"

#include <cstdlib>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace lfs::denoise::native {
namespace {

constexpr std::uint32_t kBlockSize = 256u;

[[nodiscard]] std::vector<std::byte> read_binary(
    const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return {};
    const std::streamsize size = stream.tellg();
    if (size <= 0) return {};
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char *>(bytes.data()), size);
    return stream ? bytes : std::vector<std::byte>{};
}

[[nodiscard]] backend::Error status_error(
    const ocarina::ExternalComputeStatus &status,
    const char *operation) {
    if (status.ok()) return {};
    std::string message = operation;
    message += " failed with external compute status ";
    message += std::to_string(static_cast<std::uint32_t>(status.code));
    message += " (backend code ";
    message += std::to_string(status.backend_code);
    message += ")";
    return {backend::ErrorCode::BackendFailure, std::move(message)};
}

} // namespace

std::filesystem::path encoded_carrier_composer_fatbin_path() {
    return lfs::denoise::runtime_directory()/"lfs_encoded_carrier_composer.fatbin";
}

backend::Error EncodedCarrierComposerRuntime::initialize(
    ocarina::Device &device) noexcept {
    if (device_ == &device && valid()) return {};
    clear_submission();
    const std::vector<std::byte> image =
        read_binary(encoded_carrier_composer_fatbin_path());
    if (image.empty()) {
        return {backend::ErrorCode::BackendFailure,
                "Encoded carrier composer fatbin is missing or empty."};
    }
    module_ = device.create_external_compute_module(
        {image.data(), image.size()}, "lfs.denoise.encoded_carrier_composer");
    if (!module_.valid()) {
        return status_error(module_.status(), "Encoded carrier module load");
    }
    compose_demodulated_half4_ = module_.kernel(
        "lfs_encoded_carrier_compose_demodulated_half4");
    guard_canonical_=module_.kernel("lfs_encoded_carrier_guard_canonical");
    if (!compose_demodulated_half4_.valid()||!guard_canonical_.valid()) {
        return status_error(
            compose_demodulated_half4_.status(),
            "Demodulated carrier entry lookup");
    }

    device_ = &device;
    return {};
}

bool EncodedCarrierComposerRuntime::valid() const noexcept {
    return device_ != nullptr && module_.valid() &&
        compose_demodulated_half4_.valid()&&guard_canonical_.valid();
}

backend::Error EncodedCarrierComposerRuntime::dispatch(
    const ocarina::ExternalComputeKernel &kernel,
    const EncodedCarrierComposeParams &params,
    ocarina::CommandBatch &commands) noexcept {
    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(params.output_width) * params.output_height;
    if (!valid() || pending_status_) {
        return {backend::ErrorCode::InvalidState,
                "Encoded carrier composer is unavailable."};
    }
    if (pixel_count == 0u || params.source == 0u ||
        params.canonical == 0u ||
        (params.frame_status_enabled != 0u && params.frame_status == 0u)) {
        return {backend::ErrorCode::InvalidArgument,
                "Encoded carrier composer resources are invalid."};
    }
    const std::uint32_t blocks = static_cast<std::uint32_t>(
        (pixel_count + kBlockSize - 1u) / kBlockSize);
    const auto bytes = std::span<const std::byte>{
        reinterpret_cast<const std::byte *>(&params), sizeof(params)};
    ocarina::ExternalComputeDispatchCommand *command = kernel.dispatch(
        bytes, ocarina::make_uint3(blocks, 1u, 1u),
        ocarina::make_uint3(kBlockSize, 1u, 1u));
    if (command == nullptr) {
        return {backend::ErrorCode::BackendFailure,
                "Encoded carrier composer returned no command."};
    }
    pending_status_ = command->status();
    if (!pending_status_ || !pending_status_->ok()) {
        const backend::Error error = pending_status_
            ? status_error(*pending_status_, "Encoded carrier dispatch")
            : backend::Error{backend::ErrorCode::BackendFailure,
                "Encoded carrier composer returned no status."};
        command->recycle();
        pending_status_.reset();
        return error;
    }
    commands << command;
    return {};
}

backend::Error EncodedCarrierComposerRuntime::dispatch_demodulated_half4(
    const EncodedCarrierComposeParams &params,
    ocarina::CommandBatch &commands) noexcept {
    if (params.current_observation == 0u) {
        return {backend::ErrorCode::InvalidArgument,
                "Demodulated carrier composition requires surface observations."};
    }
    return dispatch(compose_demodulated_half4_, params, commands);
}

backend::Error EncodedCarrierComposerRuntime::dispatch_canonical_guard(const EncodedCarrierComposeParams &params,ocarina::CommandBatch &commands) noexcept {return dispatch(guard_canonical_,params,commands);}

backend::Error EncodedCarrierComposerRuntime::validate_submission() const noexcept {
    if (!pending_status_) {
        return {backend::ErrorCode::InvalidState,
                "Encoded carrier composer was not submitted."};
    }
    return status_error(*pending_status_, "Encoded carrier execution");
}

void EncodedCarrierComposerRuntime::clear_submission() noexcept {
    pending_status_.reset();
}

} // namespace lfs::denoise::native
