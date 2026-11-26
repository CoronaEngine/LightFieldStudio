#include "ssat/encoded_carrier_composer.h"

#include "ssat/native/encoded_carrier_composer_native_runtime.h"

#include "dsl/dsl.h"
#include "math/base.h"
#include "math/basic_types.h"
#include "rhi/resources/shader.h"

#include <exception>
#include <string>

namespace lfs::denoise {
using namespace ocarina;

namespace {

template<typename T>
[[nodiscard]] std::uint64_t address(const BufferView<T> &view) noexcept {
    return static_cast<std::uint64_t>(view.handle()) +
        static_cast<std::uint64_t>(view.offset_in_byte());
}

} // namespace

struct EncodedCarrierComposer::Impl {
    Device *device{};
    backend::Extent2D sample_extent{};
    backend::Extent2D output_extent{};
    std::uint64_t layout_generation{};
    std::uint64_t layout_hash{};
    Buffer<float4> canonical{};
    BufferView<float4> borrowed_canonical{};
    bool borrow_canonical{};
    native::EncodedCarrierComposerRuntime native_runtime{};
    ModuleTelemetry telemetry{"EncodedCarrierComposer"};
    bool prepared{};
    bool native_dispatched{};
    bool submission_output_ready{};

    void clear_submission() noexcept {
        native_runtime.clear_submission();
        native_dispatched = false;
        submission_output_ready = false;borrowed_canonical={};
    }
};

EncodedCarrierComposer::EncodedCarrierComposer(bool borrow_canonical)
    : impl_(std::make_unique<Impl>()) {impl_->borrow_canonical=borrow_canonical;}
EncodedCarrierComposer::EncodedCarrierComposer(
    EncodedCarrierComposer &&) noexcept = default;
EncodedCarrierComposer &EncodedCarrierComposer::operator=(
    EncodedCarrierComposer &&) noexcept = default;
EncodedCarrierComposer::~EncodedCarrierComposer() noexcept = default;

backend::Error EncodedCarrierComposer::reset() noexcept {
    impl_->clear_submission();
    impl_->telemetry.output_ready = false;
    return {};
}

backend::Error EncodedCarrierComposer::prepare(
    const backend::FrameContext &context) noexcept {
    if (context.device == nullptr || context.stream == nullptr ||
        context.sample_domain != backend::SampleDomain::EncodedSubpixelGrid ||
        !context.sample_extent.valid() || !context.output_extent.valid() ||
        context.sample_extent.width != context.output_extent.width * 3u ||
        context.sample_extent.height != context.output_extent.height) {
        return {backend::ErrorCode::ResourceMismatch,
                "EncodedCarrierComposer requires a dense 3W by H layout."};
    }
    const bool changed = impl_->device != context.device ||
        impl_->sample_extent != context.sample_extent ||
        impl_->output_extent != context.output_extent ||
        impl_->layout_generation != context.layout_generation ||
        impl_->layout_hash != context.layout_hash;
    if (!changed && impl_->prepared) return {};

    impl_->device = context.device;
    impl_->sample_extent = context.sample_extent;
    impl_->output_extent = context.output_extent;
    impl_->layout_generation = context.layout_generation;
    impl_->layout_hash = context.layout_hash;
    impl_->clear_submission();
    try {
        if(!impl_->borrow_canonical)impl_->canonical = impl_->device->create_buffer<float4>(
            static_cast<std::size_t>(context.output_extent.area()),
            "lfs.denoise.encoded_carrier_composer.canonical");
    } catch (const std::exception &exception) {
        return {backend::ErrorCode::BackendFailure, exception.what()};
    } catch (...) {
        return {backend::ErrorCode::BackendFailure,
                "EncodedCarrierComposer allocation or compilation failed."};
    }
    if (backend::Error error =
            impl_->native_runtime.initialize(*impl_->device)) {
        return error;
    }
    impl_->telemetry.owned_device_bytes =
        (impl_->borrow_canonical?0u:context.output_extent.area()*sizeof(float4));
    impl_->telemetry.allocation_count = impl_->borrow_canonical?0u:1u;
    impl_->telemetry.write_bytes =
        context.output_extent.area() * sizeof(float4);
    impl_->prepared = true;
    return {};
}

backend::Error EncodedCarrierComposer::validate_submission() noexcept {
    if (impl_->native_dispatched) {
        return impl_->native_runtime.validate_submission();
    }
    return impl_->telemetry.output_ready
        && impl_->submission_output_ready
        ? backend::Error{}
        : backend::Error{backend::ErrorCode::InvalidState,
                         "EncodedCarrierComposer produced no output."};
}

void EncodedCarrierComposer::clear_submission() noexcept {
    impl_->clear_submission();
}

BufferView<float4> EncodedCarrierComposer::canonical_view() const noexcept {
    return impl_->borrow_canonical?impl_->borrowed_canonical:impl_->canonical.view();
}

const ModuleTelemetry &EncodedCarrierComposer::telemetry() const noexcept {
    return impl_->telemetry;
}

backend::Result<backend::PresentOutput>
EncodedCarrierComposer::compose_native(
    const backend::FrameContext &context,
    const backend::FrameResources &resources,
    const CompositionView &source,
    CommandBatch &commands) noexcept {
    impl_->clear_submission();
    impl_->telemetry.write_bytes = impl_->output_extent.area() * sizeof(float4);
    if (!impl_->prepared || !source.valid() ||
        source.sample_extent != impl_->sample_extent ||
        source.frame_index != context.frame_index ||
        source.layout_generation != context.layout_generation ||
        source.layout_hash != context.layout_hash ||
        context.output_extent != impl_->output_extent) {
        return backend::Result<backend::PresentOutput>::failure({
            backend::ErrorCode::ResourceMismatch,
            "Native composition source does not match the current frame."});
    }
    if(impl_->borrow_canonical){
        if(source.encoding!=CompositionEncoding::CanonicalDisplayFloat4||source.packed_source_stride_bytes!=sizeof(float4)||!source.frame_status_address)return backend::Result<backend::PresentOutput>::failure({backend::ErrorCode::ResourceMismatch,"Fused canonical output contract mismatch."});
        impl_->borrowed_canonical=BufferView<float4>{source.packed_source_address,static_cast<size_t>(context.output_extent.area())};
        native::EncodedCarrierComposeParams guard{};guard.source=source.packed_source_address;guard.canonical=source.packed_source_address;guard.frame_status=source.frame_status_address;guard.frame_status_enabled=1u;guard.output_width=context.output_extent.width;guard.output_height=context.output_extent.height;
        if(auto error=impl_->native_runtime.dispatch_canonical_guard(guard,commands))return backend::Result<backend::PresentOutput>::failure(error);
        impl_->native_dispatched=true;impl_->submission_output_ready=true;impl_->telemetry.output_ready=true;impl_->telemetry.dispatch_count=1u;impl_->telemetry.read_bytes=context.output_extent.area()*sizeof(uint32_t);impl_->telemetry.write_bytes=0u; // Successful guard only reads status; failure fills canonical explicitly.
        return backend::Result<backend::PresentOutput>::success({true,"ssat_fused_final_composition",impl_->borrowed_canonical});
    }
    if (source.encoding != CompositionEncoding::DemodulatedLinearHalf4 || source.packed_source_stride_bytes != sizeof(native::Half4)) {
        return backend::Result<backend::PresentOutput>::failure({backend::ErrorCode::ResourceMismatch, "Composition requires demodulated linear Half4."});
    }
    const auto *observation =
        resources.view<backend::PackedSurfaceObservationV1>(
            backend::Semantic::PackedSurfaceObservation,
            backend::TemporalSlot::Current, 0u);
    if (observation == nullptr ||
        observation->size() != impl_->sample_extent.area()) {
        return backend::Result<backend::PresentOutput>::failure({
            backend::ErrorCode::ResourceMissing,
            "LF spatial composition requires a dense current observation."});
    }
    native::EncodedCarrierComposeParams params{};
    params.source = source.packed_source_address;
    if (context.deterministic_environment) {
        const auto *radiance = resources.view<float4>(
            backend::Semantic::Radiance, backend::TemporalSlot::Current, 0u);
        if (radiance == nullptr || radiance->size() != impl_->sample_extent.area()) {
            return backend::Result<backend::PresentOutput>::failure({
                backend::ErrorCode::ResourceMissing,
                "Direct environment composition requires current traced radiance."});
        }
        params.current_radiance = address(*radiance);
        params.deterministic_environment = 1u;
    }
    params.current_observation = address(*observation);
    params.canonical = address(impl_->canonical.view());
    params.frame_status = source.frame_status_address;
    params.output_width = impl_->output_extent.width;
    params.output_height = impl_->output_extent.height;
    params.frame_status_enabled = source.frame_status_address != 0u ? 1u : 0u;
    const backend::Error dispatch_error = impl_->native_runtime.dispatch_demodulated_half4(params, commands);
    if (dispatch_error) {
        return backend::Result<backend::PresentOutput>::failure(dispatch_error);
    }
    impl_->native_dispatched = true;
    impl_->telemetry.read_bytes =
        source.sample_extent.area() *
        (source.packed_source_stride_bytes +
         sizeof(backend::PackedSurfaceObservationV1) +
         (context.deterministic_environment ? sizeof(float4) : 0u));
    impl_->telemetry.dispatch_count = 1u;
    impl_->telemetry.output_ready = true;
    impl_->submission_output_ready = true;
    return backend::Result<backend::PresentOutput>::success({
        true,
        "encoded_carrier_composer_demodulated_half4",
        impl_->canonical.view()});
}

} // namespace lfs::denoise
