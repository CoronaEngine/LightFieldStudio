#pragma once
#include "backend/frame_contract.h"

namespace lfs::denoise::resident {
inline bool can_share_dense_aovs(const lfs::backend::ResourceRequirementSet& demand,bool separate_preview) noexcept {
    using namespace lfs::backend;
    if(separate_preview || !demand.contains(Semantic::Radiance) || !demand.contains(Semantic::PackedSurfaceObservation))return false;
    for(const auto& r:demand.items()){
        if(r.temporal_slot!=TemporalSlot::Current || r.layer || r.retain_for_previous)return false;
        if(r.semantic==Semantic::PackedSurfaceObservation && r.value_type==ValueType::PackedSurfaceObservation32)continue;
        if(r.semantic==Semantic::Radiance && (r.value_type==ValueType::Float4 ||
            r.value_type==ValueType::Float4InterleavedRgbMeasurementVariance))continue;
        return false;
    }
    return true;
}
bool activate_shared_aovs(ocarina::Device* device,unsigned width,unsigned height,bool ordered,
                          const lfs::backend::ResourceRequirementSet& demand,bool separate_preview) noexcept;
}
