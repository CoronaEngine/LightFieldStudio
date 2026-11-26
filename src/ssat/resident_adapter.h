#pragma once
#include "ssat/native_aov_policy.h"
#include "ssat/phase1/phase_frame_state.h"

namespace lfs::denoise::resident {
class ResidentAdapter final : public backend::FrameProcessor, public lightfield::PhaseFrameStateConsumer {
    backend::FrameProcessor& child_;
    lightfield::PhaseFrameStateConsumer& phase_;
    backend::TraceRequest::ContinuationSplitMode continuation_;
    bool independent_aovs_;
public:
    ResidentAdapter(backend::FrameProcessor& child,lightfield::PhaseFrameStateConsumer& phase,
                    backend::TraceRequest::ContinuationSplitMode continuation=backend::TraceRequest::ContinuationSplitMode::PrimaryResponsiveSpatiallyStratified,
                    bool independent_aovs=false)
        :child_(child),phase_(phase),continuation_(continuation),independent_aovs_(independent_aovs){}
    bool set_phase_frame_state(const lightfield::PhaseFrameState& state) noexcept override{return phase_.set_phase_frame_state(state);}
    void invalidate_phase_frame_state() noexcept override{phase_.invalidate_phase_frame_state();}
    backend::Error reset(backend::ResetReason reason) noexcept override{return child_.reset(reason);}
    backend::Error prepare(const backend::FrameContext& c) noexcept override {
        const auto demand=child_.resource_requirements(c);
        if(!demand.ok())return demand.error();
        const bool ordered=continuation_==backend::TraceRequest::ContinuationSplitMode::Off;
        if(!activate_shared_aovs(c.device,c.sample_extent.width,c.sample_extent.height,ordered,demand.value(),independent_aovs_))
            return {backend::ErrorCode::ResourceMismatch,"Cannot bind the frozen packed surface producer for the declared inputs"};
        return child_.prepare(c);
    }
    backend::Result<backend::ResourceRequirementSet> resource_requirements(const backend::FrameContext& c) noexcept override{return child_.resource_requirements(c);}
    backend::Result<backend::ProcessOutput> process(const backend::FrameContext& c,const backend::FrameResources& inputs,ocarina::CommandBatch& commands) noexcept override {
        return child_.process(c,inputs,commands);
    }
    backend::Result<backend::PresentOutput> present(const backend::FrameContext& c,const backend::FrameResources& inputs,
            const backend::ProcessOutput& signals,ocarina::CommandBatch& commands) noexcept override {
        return child_.present(c,inputs,signals,commands);
    }
    backend::Error enqueue_commit(const backend::FrameContext& c,ocarina::CommandBatch& commands) noexcept override{return child_.enqueue_commit(c,commands);}
    backend::Error validate_submission(const backend::FrameContext& c) noexcept override{return child_.validate_submission(c);}
    void finalize_success(const backend::FrameContext& c) noexcept override{child_.finalize_success(c);}
    void abort_submission(const backend::FrameContext& c) noexcept override{child_.abort_submission(c);}
};
}
