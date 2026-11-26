#pragma once
#include "backend/frame_contract.h"
#include "ssat/phase3/native/early_measurement_params.h"
#include "rhi/resources/external_compute.h"
#include <vector>

namespace lfs::denoise::mr_native {
class Runtime {
    ocarina::Device *device_{};
    ocarina::ExternalComputeModule module_{};
    ocarina::ExternalComputeKernel measurement_{}, regular_measurement_{}, certified_measurement_{}, certified_vertex_measurement_{}, vertex_measurement_{}, stationary_{}, commit_geometry_{}, restore_geometry_{};

    bool logarithmic_{};
    std::vector<ocarina::SP<ocarina::ExternalComputeStatus>> pending_{};
public:
    backend::Error initialize(ocarina::Device &device,bool logarithmic=false) noexcept;
    backend::Error dispatch(const Params &params, ocarina::CommandBatch &commands, bool regular_layout) noexcept;
    backend::Error commit_history(const Params& params,ocarina::CommandBatch& commands) noexcept;
    backend::Error validate() const noexcept;
    void clear() noexcept { pending_.clear(); }
};
}
