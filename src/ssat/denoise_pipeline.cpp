#include "ssat/denoise_pipeline.h"
#include "ssat/phase3/physical_lf_reprojection_taa.h"
#include "ssat/encoded_carrier_composer.h"
#include "ssat/phase2/encoded_atrous_spatial.h"
#include <utility>
namespace lfs::denoise {
struct DenoisePipeline::Impl {
    PipelineConfig config;
    std::unique_ptr<ISpatialDenoiser> spatial;
    std::unique_ptr<ITemporalDenoiser> temporal;
    EncodedCarrierComposer composer;
    Impl(PipelineConfig c,std::unique_ptr<ISpatialDenoiser> s,std::unique_ptr<ITemporalDenoiser> t)
        :config(c),spatial(std::move(s)),temporal(std::move(t)),composer(true){}
};
DenoisePipeline::DenoisePipeline(PipelineConfig c,std::unique_ptr<ISpatialDenoiser> s,std::unique_ptr<ITemporalDenoiser> t)
    :impl_(std::make_unique<Impl>(c,std::move(s),std::move(t))){}
DenoisePipeline::DenoisePipeline(DenoisePipeline &&) noexcept=default;
DenoisePipeline &DenoisePipeline::operator=(DenoisePipeline &&) noexcept=default;
DenoisePipeline::~DenoisePipeline() noexcept=default;
bool DenoisePipeline::set_phase_frame_state(const lightfield::PhaseFrameState &state) noexcept {
    bool accepted=true;
    for(auto *p:{static_cast<backend::FrameProcessor *>(impl_->spatial.get()),static_cast<backend::FrameProcessor *>(impl_->temporal.get())})
        if(auto *c=dynamic_cast<lightfield::PhaseFrameStateConsumer *>(p))accepted=c->set_phase_frame_state(state)&&accepted;
    return accepted;
}
void DenoisePipeline::invalidate_phase_frame_state() noexcept {
    for(auto *p:{static_cast<backend::FrameProcessor *>(impl_->spatial.get()),static_cast<backend::FrameProcessor *>(impl_->temporal.get())})
        if(auto *c=dynamic_cast<lightfield::PhaseFrameStateConsumer *>(p))c->invalidate_phase_frame_state();
}
backend::Error DenoisePipeline::reset(backend::ResetReason reason) noexcept {
    if(auto e=impl_->composer.reset())return e;
    if(auto e=impl_->spatial->reset(reason))return e;
    return impl_->temporal->reset(reason);
}
backend::Error DenoisePipeline::prepare(const backend::FrameContext &c) noexcept {
    if(auto e=impl_->composer.prepare(c))return e;
    if(auto e=impl_->spatial->prepare(c))return e;
    return impl_->temporal->prepare(c);
}
backend::Error DenoisePipeline::enqueue_commit(const backend::FrameContext &c,ocarina::CommandBatch &cmd) noexcept {
    if(auto e=impl_->spatial->enqueue_commit(c,cmd))return e;
    return impl_->temporal->enqueue_commit(c,cmd);
}
backend::Error DenoisePipeline::validate_submission(const backend::FrameContext &c) noexcept {
    if(auto e=impl_->spatial->validate_submission(c))return e;
    if(auto e=impl_->temporal->validate_submission(c))return e;
    return impl_->composer.validate_submission();
}
void DenoisePipeline::finalize_success(const backend::FrameContext &c) noexcept {
    impl_->spatial->finalize_success(c);impl_->temporal->finalize_success(c);impl_->composer.clear_submission();
}
void DenoisePipeline::abort_submission(const backend::FrameContext &c) noexcept {
    impl_->temporal->abort_submission(c);impl_->spatial->abort_submission(c);impl_->composer.clear_submission();
    if(auto *p=dynamic_cast<IResolvedCurrentConsumer *>(impl_->temporal.get()))p->invalidate_resolved_current_view();
}
backend::Result<backend::ResourceRequirementSet>
DenoisePipeline::resource_requirements(
    const backend::FrameContext &context) noexcept {
    const bool native_spatial_handoff = dynamic_cast<IResolvedCurrentProvider *>(impl_->spatial.get()) != nullptr &&
        dynamic_cast<IResolvedCurrentConsumer *>(impl_->temporal.get()) != nullptr;
    backend::ResourceRequirementSet merged;
    for (backend::FrameProcessor *processor : {
             static_cast<backend::FrameProcessor *>(impl_->spatial.get()),
             static_cast<backend::FrameProcessor *>(impl_->temporal.get())}) {
        if (processor == nullptr) continue;
        backend::Result<backend::ResourceRequirementSet> requirements =
            processor->resource_requirements(context);
        if (!requirements.ok()) {
            return backend::Result<backend::ResourceRequirementSet>::failure(
                requirements.error());
        }
        for (const backend::ResourceRequirement &item :
             requirements.value().items()) {
            if (native_spatial_handoff && processor == impl_->temporal.get() &&
                !impl_->temporal->requires_raw_radiance_with_native_current() &&
                item.semantic == backend::Semantic::Radiance &&
                item.temporal_slot == backend::TemporalSlot::Current &&
                item.layer == 0u) {
                continue;
            }
            backend::Result<void> added = merged.add(item);
            if (!added.ok()) {
                return backend::Result<backend::ResourceRequirementSet>::failure(
                    added.error());
            }
        }
        if (requirements.value().requires_rigid_instance_motion()) {
            merged.require_rigid_instance_motion();
        }
    }
    return backend::Result<backend::ResourceRequirementSet>::success(
        std::move(merged));
}
backend::Result<backend::ProcessOutput> DenoisePipeline::process(const backend::FrameContext &context,
    const backend::FrameResources &resources,ocarina::CommandBatch &commands) noexcept {
    auto *spatial=dynamic_cast<EncodedAtrousSpatial *>(impl_->spatial.get());
    if(!spatial)return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::InvalidState,"Spatial reconstruction is unavailable."});
    bool used_history=false;
    spatial->set_interstage_processor([&](const backend::FrameContext &ctx,const backend::FrameResources &res,
        ocarina::BufferView<ocarina::half4> signal,ocarina::BufferView<ocarina::uint> status,std::uint64_t &prior,std::uint64_t &current_regular,
        std::uint64_t &regular_material_normals, std::uint64_t &panel_samples,
        std::uint64_t held_angular_residuals,std::uint64_t current_angular_residuals, RegularR2Coverage regular_r2_coverage,
        ocarina::CommandBatch &cmd)->backend::Error {
        ResolvedCurrentView view{};
        view.domain=SignalDomain::DemodulatedRgbVariance;view.encoding=NativeCurrentEncoding::LinearHalf4;
        view.extent=ctx.sample_extent;view.packed_signal_address=signal.handle()+signal.offset_in_byte();
        view.packed_signal_stride_bytes=sizeof(ocarina::half4);view.packed_signal=signal.view_as<uint8_t>();
        view.frame_index=ctx.frame_index;view.layout_generation=ctx.layout_generation;view.layout_hash=ctx.layout_hash;
        view.frame_status_address=status.handle()+status.offset_in_byte();view.frame_status=status.view_as<uint8_t>();
        view.held_angular_residuals=held_angular_residuals;
        view.current_angular_residuals=current_angular_residuals;
        view.regular_r2_coverage=regular_r2_coverage;
        view.frame_status_encoding=impl_->config.lane==Lane::Quality?FrameStatusEncoding::StatusAndReconstructionCoverageUInt32:
            (status.size()>1u?FrameStatusEncoding::StatusAndGridTileCoverageUInt32:FrameStatusEncoding::ScalarUInt32);
        auto *consumer=dynamic_cast<IResolvedCurrentConsumer *>(impl_->temporal.get());
        if(!consumer||!consumer->set_resolved_current_view(view))return {backend::ErrorCode::ResourceMismatch,"Temporal reconstruction rejected the spatial observation."};
        auto output=impl_->temporal->process(ctx,res,cmd);if(!output.ok())return output.error();
        used_history=output.value().temporal_history_used;
        if(auto *measurement=dynamic_cast<PhysicalLfReprojectionTaa*>(impl_->temporal.get())) {
            prior=measurement->previous_measurement_geometry();
            current_regular=measurement->current_regular_measurement_geometry();
            regular_material_normals=measurement->current_regular_material_normals();
            panel_samples=measurement->current_panel_samples();
        }
        return {};
    });
    auto output=spatial->process(context,resources,commands);spatial->set_interstage_processor({});
    if(output.ok())output.value().temporal_history_used=used_history;
    return output;
}
backend::Result<backend::PresentOutput> DenoisePipeline::present(const backend::FrameContext &context,
    const backend::FrameResources &resources,const backend::ProcessOutput &,ocarina::CommandBatch &commands) noexcept {
    auto *provider=dynamic_cast<ICompositionSourceProvider *>(impl_->spatial.get());
    const auto *source=provider?provider->composition_source_view():nullptr;
    if(!source)return backend::Result<backend::PresentOutput>::failure({backend::ErrorCode::InvalidState,"The spatial output is unavailable."});
    return impl_->composer.compose_native(context,resources,*source,commands);
}
const PipelineConfig &DenoisePipeline::config() const noexcept{return impl_->config;}
const ISpatialDenoiser *DenoisePipeline::spatial() const noexcept{return impl_->spatial.get();}
ITemporalDenoiser *DenoisePipeline::temporal() noexcept{return impl_->temporal.get();}
const ITemporalDenoiser *DenoisePipeline::temporal() const noexcept{return impl_->temporal.get();}
const ModuleTelemetry &DenoisePipeline::composer_telemetry() const noexcept{return impl_->composer.telemetry();}
} // namespace lfs::denoise
