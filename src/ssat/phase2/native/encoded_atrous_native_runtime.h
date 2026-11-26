#pragma once

#include "backend/frame_contract.h"
#include "ssat/native/reconstruction_coverage.h"
#include "ssat/phase2/native/encoded_atrous_native_params.h"

#include "rhi/resources/external_compute.h"

#include <filesystem>
#include <vector>

namespace lfs::denoise::encoded_atrous::native {

enum class ExtentProfile : std::uint32_t {
    Invalid = 0u,
    K2 = 1u,
    K4 = 2u,
    K8 = 3u,
};

enum class ExecutionProfile : std::uint32_t {
    General,
    PlainCurrent,
    GlossyMeasurement,
};

[[nodiscard]] ExtentProfile extent_profile(
    backend::Extent2D sample_extent) noexcept;

class EncodedAtrousNativeRuntime {
public:
    struct Kernels {
        ocarina::ExternalComputeKernel init_prefilter{};
        ocarina::ExternalComputeKernel init_step1{};
        ocarina::ExternalComputeKernel step1{};
        ocarina::ExternalComputeKernel finish_registered{}, build_epipolar{};
        ocarina::ExternalComputeKernel geometry_check{}, build_weights{}, cached_color{}, build_cells{}, apply_cells{}, update_fixed{};
        ocarina::ExternalComputeKernel collect_held_es{}, present_held_es{}, capture_held_es1{}, capture_held_environment_es1{};
        ocarina::ExternalComputeKernel build_held_view_cells{}, present_held_view_cells{};
        ocarina::ExternalComputeKernel step2{};
        ocarina::ExternalComputeKernel present_panel_samples{};
        ocarina::ExternalComputeKernel regular_es1{},regular_carrier_r2{},regular_routed_r2{},regular_moving_chart{},regular_measured_chart{},step4{},step4_cached{},directional_refinement{},directional_chart{},final{},moving_panel{},resident_panel{};
    };

private:
    ocarina::Device *device_{};
    ocarina::ExternalComputeModule module_{};
    ExecutionProfile execution_profile_{ExecutionProfile::General};
    Kernels k2_{};
    Kernels k4_{};
    Kernels k8_{};
    std::vector<ocarina::SP<ocarina::ExternalComputeStatus>> pending_{};

    [[nodiscard]] backend::Error enqueue(
        const ocarina::ExternalComputeKernel &kernel,
        const EncodedAtrousNativeParams &params,
        ocarina::uint3 grid,
        ocarina::uint3 block,
        ocarina::CommandBatch &commands) noexcept;

public:
    [[nodiscard]] backend::Error initialize(
        ocarina::Device &device,
        ExecutionProfile profile = ExecutionProfile::General) noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] RegularR2Coverage regular_r2_coverage(const EncodedAtrousNativeParams &params) const noexcept {
        if (execution_profile_ != ExecutionProfile::GlossyMeasurement || params.resident_storage ||
            !params.register_motion || params.hold_same) return RegularR2Coverage::Original;
        return params.deterministic_environment ? RegularR2Coverage::DeterministicEnvironment : RegularR2Coverage::AllSources;
    }
    [[nodiscard]] backend::Error dispatch_atrous_pass(
        ExtentProfile profile,
        std::uint32_t step,
        const EncodedAtrousNativeParams &params,
        ocarina::CommandBatch &commands) noexcept;
    [[nodiscard]] backend::Error dispatch_init_prefilter(
        ExtentProfile profile,
        const EncodedAtrousNativeParams &params,
        ocarina::CommandBatch &commands,
        bool fuse_step1 = false) noexcept;
    [[nodiscard]] backend::Error dispatch_motion_field(ExtentProfile profile, const EncodedAtrousNativeParams &params, ocarina::CommandBatch &commands) noexcept;
    [[nodiscard]] backend::Error dispatch_resident_registered(ExtentProfile profile, const EncodedAtrousNativeParams &params, ocarina::CommandBatch &commands) noexcept;
    [[nodiscard]] backend::Error dispatch_held_es(ExtentProfile profile, const EncodedAtrousNativeParams &params, ocarina::CommandBatch &commands) noexcept;
    [[nodiscard]] backend::Error dispatch_stationary(ExtentProfile profile, const EncodedAtrousNativeParams &params, ocarina::CommandBatch &commands) noexcept;
    [[nodiscard]] backend::Error validate_submission() const noexcept;
    [[nodiscard]] backend::Error dispatch_final(ExtentProfile profile,const EncodedAtrousNativeParams &params,ocarina::CommandBatch &commands) noexcept;
    void clear_submission() noexcept;
};

[[nodiscard]] std::filesystem::path encoded_atrous_native_fatbin_path(
    ExecutionProfile profile = ExecutionProfile::General);

} // namespace lfs::denoise::encoded_atrous::native
