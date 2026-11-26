#include "ssat/phase2/encoded_atrous_spatial.h"

namespace lfs::denoise {

EncodedAtrousSpatial::EncodedAtrousSpatial(EncodedConfig config)
    : processor_(config) {}

SpatialMethod EncodedAtrousSpatial::method() const noexcept {
    return SpatialMethod::EncodedAtrous;
}
SignalDomain EncodedAtrousSpatial::input_domain() const noexcept {
    return SignalDomain::RawPhysicalRgbSecondMoment;
}
SignalDomain EncodedAtrousSpatial::output_domain() const noexcept {
    return SignalDomain::DemodulatedRgbVariance;
}
const ModuleTelemetry &EncodedAtrousSpatial::module_telemetry() const noexcept {
    return telemetry_;
}
const ResolvedCurrentView *EncodedAtrousSpatial::resolved_current_view()
    const noexcept {
    return current_view_.valid() ? &current_view_ : nullptr;
}
const CompositionView *EncodedAtrousSpatial::composition_source_view()
    const noexcept {
    return composition_view_.valid() ? &composition_view_ : nullptr;
}
bool EncodedAtrousSpatial::set_phase_frame_state(
    const lightfield::PhaseFrameState &state) noexcept {
    const bool ok=processor_.set_phase_frame_state(state);
    if(ok){telemetry_.owned_device_bytes=processor_.telemetry().internal_history_bytes;telemetry_.allocation_count=processor_.telemetry().allocation_count;}
    return ok;
}
void EncodedAtrousSpatial::invalidate_phase_frame_state() noexcept {
    current_view_ = {};
    composition_view_ = {};
    processor_.invalidate_phase_frame_state();
}
backend::Error EncodedAtrousSpatial::reset(backend::ResetReason reason) noexcept {
    current_view_ = {};
    composition_view_ = {};
    processor_.invalidate_phase_frame_state();
    telemetry_.read_bytes = 0u;
    telemetry_.write_bytes = 0u;
    telemetry_.dispatch_count = 0u;
    telemetry_.output_ready = false;
    return processor_.reset(reason);
}
backend::Error EncodedAtrousSpatial::prepare(
    const backend::FrameContext &context) noexcept {
    const backend::Error error = processor_.prepare(context);
    if (!error) {
        const EncodedTelemetry &core = processor_.telemetry();
        telemetry_.owned_device_bytes = core.internal_history_bytes;
        telemetry_.allocation_count = core.allocation_count;
    }
    return error;
}
backend::Error EncodedAtrousSpatial::enqueue_commit(
    const backend::FrameContext &context,
    ocarina::CommandBatch &commands) noexcept {
    return processor_.enqueue_commit(context, commands);
}
void EncodedAtrousSpatial::finalize_success(
    const backend::FrameContext &context) noexcept {
    processor_.finalize_success(context);
}
void EncodedAtrousSpatial::abort_submission(
    const backend::FrameContext &context) noexcept {
    current_view_ = {};
    composition_view_ = {};
    processor_.abort_submission(context);
}
backend::Error EncodedAtrousSpatial::validate_submission(
    const backend::FrameContext &context) noexcept {
    return processor_.validate_submission(context);
}
backend::Result<backend::ResourceRequirementSet>
EncodedAtrousSpatial::resource_requirements(
    const backend::FrameContext &context) noexcept {
    return processor_.resource_requirements(context);
}
backend::Result<backend::ProcessOutput> EncodedAtrousSpatial::process(
    const backend::FrameContext &context,
    const backend::FrameResources &resources,
    ocarina::CommandBatch &commands) noexcept {
    backend::Result<backend::ProcessOutput> result =
        processor_.process(context, resources, commands);
    telemetry_.output_ready = result.ok();
    const std::uint64_t count = context.sample_extent.area();
    const std::uint32_t passes = processor_.config().atrous_iterations;
    telemetry_.dispatch_count = 3u + passes;
    if(passes==3u&&processor_.config().responsive_angular_radiometry)++telemetry_.dispatch_count; // Separate exact-source reflectiveR4 dispatch.
    const bool fused_first = processor_.telemetry().spatial_init_step1_fused;
    telemetry_.dispatch_count -= fused_first ? 1u : 0u;
    // Logical resource-size estimate, not measured cache/DRAM transactions.
    // Fusion removes the full-frame prefilter write and step1 input stream;
    // shared-tile halo reuse/recomputation is not represented by this estimate.
    telemetry_.read_bytes = count * (
        sizeof(ocarina::float4) + sizeof(ocarina::float4) +
        sizeof(std::uint32_t) +
        static_cast<std::uint64_t>(passes + 2u - (fused_first ? 1u : 0u)) *
            sizeof(ocarina::half4));
    telemetry_.read_bytes += count * (sizeof(backend::PackedSurfaceObservationV1) + sizeof(float) + sizeof(std::uint8_t));
    if(passes==3u&&processor_.config().responsive_angular_radiometry){
        // Worst-case logical upper bound: all sites reflective. Cache/DRAM and
        // actual runtime class coverage are measured separately, never inferred.
        telemetry_.read_bytes+=count*(4u+25u*(4u+32u+8u+4u+12u)+9u*(4u+32u+8u+4u+12u));
    }
    telemetry_.write_bytes = count * (
        static_cast<std::uint64_t>(passes + 4u - (fused_first ? 1u : 0u)) *
            sizeof(ocarina::half4) +
        0u);
    if(passes==3u&&processor_.config().responsive_angular_radiometry)telemetry_.read_bytes+=count*32u;
    // Final writes canonical directly instead of a full encoded Half4.
    if(processor_.canonical_output_view().size()!=0u){telemetry_.write_bytes-=count*sizeof(ocarina::half4);telemetry_.write_bytes+=context.output_extent.area()*sizeof(ocarina::float4);}
    if (passes == 2u && processor_.canonical_output_view().size() != 0u) {
        telemetry_.read_bytes -= count * sizeof(ocarina::half4);
        telemetry_.write_bytes -= count * sizeof(ocarina::half4);
        // Bound for exact-view geometry, sample and slot reads over 25 sources.
        telemetry_.read_bytes += count * (32u + 25u * (4u+32u+8u+4u+12u));
    }
    // Logical bound, not measured DRAM transactions.

    current_view_ = {};
    composition_view_ = {};
    if (result.ok()) {
        const ocarina::BufferView<ocarina::half4> signal =
            processor_.demodulated_output_view();
        if (signal.size() == context.sample_extent.area()) {
            const ocarina::BufferView<ocarina::uint> frame_status =
                processor_.frame_status_view();
            current_view_.domain=SignalDomain::DemodulatedRgbVariance;
            current_view_.encoding=NativeCurrentEncoding::LinearHalf4;
            current_view_.extent=context.sample_extent;
            current_view_.packed_signal_address=signal.handle()+signal.offset_in_byte();
            current_view_.packed_signal_stride_bytes=sizeof(ocarina::half4);
            current_view_.packed_signal=signal.view_as<std::uint8_t>();
            current_view_.frame_index=context.frame_index;
            current_view_.layout_generation=context.layout_generation;
            current_view_.layout_hash=context.layout_hash;
            current_view_.frame_status_address = frame_status.size() != 0u
                ? static_cast<std::uint64_t>(frame_status.handle()) +
                      static_cast<std::uint64_t>(frame_status.offset_in_byte())
                : 0u;
            current_view_.frame_status = frame_status.view_as<std::uint8_t>();

        }
        const auto canonical=processor_.canonical_output_view();
        const auto frame_status=processor_.frame_status_view();
        if(canonical.size()==context.output_extent.area()){
            composition_view_={CompositionEncoding::CanonicalDisplayFloat4,context.sample_extent,
                canonical.handle()+canonical.offset_in_byte(),sizeof(ocarina::float4),canonical.view_as<uint8_t>(),
                frame_status.handle()+frame_status.offset_in_byte(),frame_status.view_as<uint8_t>(),
                context.frame_index,context.layout_generation,context.layout_hash};
        }else if(current_view_.valid()){
            composition_view_={CompositionEncoding::DemodulatedLinearHalf4,context.sample_extent,
                current_view_.packed_signal_address,current_view_.packed_signal_stride_bytes,current_view_.packed_signal,
                current_view_.frame_status_address,current_view_.frame_status,
                context.frame_index,context.layout_generation,context.layout_hash};
        }else return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::InvalidState,"Spatial reconstruction did not publish its output."});
        if(!composition_view_.valid())return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::InvalidState,"Spatial composition view is invalid."});

    }
    return result;
}
backend::Result<backend::PresentOutput> EncodedAtrousSpatial::present(
    const backend::FrameContext &context,
    const backend::FrameResources &resources,
    const backend::ProcessOutput &output,
    ocarina::CommandBatch &commands) noexcept {
    return processor_.present(context, resources, output, commands);
}

} // namespace lfs::denoise
