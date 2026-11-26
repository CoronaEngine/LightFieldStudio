#include "ssat/runtime_paths.h"
#include "ssat/phase2/native/encoded_atrous_native_runtime.h"

#include <cstdlib>
#include <cmath>
#include <fstream>
#include <span>
#include <sstream>
#include <string>

namespace lfs::denoise::encoded_atrous::native {
namespace {

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
    return {backend::ErrorCode::BackendFailure,
            std::string{operation} + " failed with external status " +
                std::to_string(static_cast<std::uint32_t>(status.code)) +
                " (backend " + std::to_string(status.backend_code) + ")."};
}

[[nodiscard]] bool valid(const EncodedAtrousNativeRuntime::Kernels &kernels, ExecutionProfile profile) {
    return (profile==ExecutionProfile::General || (kernels.step4_cached.valid() && kernels.regular_es1.valid() && kernels.regular_moving_chart.valid())) && (profile!=ExecutionProfile::PlainCurrent || kernels.regular_carrier_r2.valid()) && (profile!=ExecutionProfile::GlossyMeasurement||(kernels.final.valid()&&kernels.moving_panel.valid()&&kernels.resident_panel.valid()&&kernels.regular_routed_r2.valid()))&&(profile!=ExecutionProfile::GlossyMeasurement||(kernels.step4.valid()&&kernels.step4_cached.valid()&&kernels.directional_refinement.valid()&&kernels.directional_chart.valid()))&&kernels.init_prefilter.valid() && kernels.step1.valid() &&
        kernels.step2.valid() && (profile!=ExecutionProfile::GlossyMeasurement || kernels.step4.valid()) && (profile == ExecutionProfile::General || kernels.init_step1.valid());
}

} // namespace

ExtentProfile extent_profile(backend::Extent2D extent) noexcept {
    if (extent == backend::Extent2D{4608u, 2048u}) return ExtentProfile::K2;
    if (extent == backend::Extent2D{11520u, 2160u}) return ExtentProfile::K4;
    if (extent == backend::Extent2D{23040u, 4320u}) return ExtentProfile::K8;
    return ExtentProfile::Invalid;
}

std::filesystem::path encoded_atrous_native_fatbin_path(ExecutionProfile profile) {
    const char *name = profile == ExecutionProfile::PlainCurrent
        ? "lfs_encoded_atrous_plain.fatbin" : (profile==ExecutionProfile::GlossyMeasurement
            ? "lfs_encoded_atrous_glossy.fatbin" : "lfs_encoded_atrous_native.fatbin");
    return lfs::denoise::runtime_directory()/name;
}

backend::Error EncodedAtrousNativeRuntime::initialize(
    ocarina::Device &device, ExecutionProfile profile) noexcept {
    if (profile != ExecutionProfile::General && profile != ExecutionProfile::PlainCurrent && profile != ExecutionProfile::GlossyMeasurement)
        return {backend::ErrorCode::Unsupported, "Unknown ES execution profile."};
    if (device_ == &device && valid() && execution_profile_ == profile) return {};
    device_ = nullptr;
    execution_profile_ = profile;
    k2_ = {}; k4_ = {}; k8_ = {};
    clear_submission();
    const std::vector<std::byte> image = read_binary(encoded_atrous_native_fatbin_path(profile));
    if (image.empty()) {
        return {backend::ErrorCode::BackendFailure,
                "Encoded A-Trous native fatbin is missing or empty."};
    }
    module_ = device.create_external_compute_module(
        {image.data(), image.size()}, profile == ExecutionProfile::PlainCurrent
            ? "lfs.encoded_atrous.plain" : (profile==ExecutionProfile::GlossyMeasurement ? "lfs.encoded_atrous.glossy_measurement" : "lfs.encoded_atrous.native"));
    if (!module_.valid()) {
        return status_error(module_.status(),
            "Encoded A-Trous native module load");
    }
    const auto bind = [&](Kernels &kernels, const char *suffix) {
        kernels.init_prefilter = module_.kernel(
            (std::string{"lfs_encoded_atrous_init_prefilter_"} + suffix).c_str());
        if (profile == ExecutionProfile::PlainCurrent || profile == ExecutionProfile::GlossyMeasurement) {
            kernels.init_step1 = module_.kernel(
                (std::string{"lfs_encoded_atrous_init_step1_"} + suffix).c_str());
        }

        if(profile!=ExecutionProfile::General)
            kernels.regular_es1=module_.kernel((std::string{"lfs_regular_deterministic_es1_"}+suffix).c_str());
        if (profile != ExecutionProfile::General)
            kernels.regular_moving_chart=module_.kernel((std::string{"lfs_regular_moving_view_chart_"}+suffix).c_str());
        if(profile==ExecutionProfile::PlainCurrent)
            kernels.regular_carrier_r2=module_.kernel((std::string{"lfs_regular_carrier_r2_"}+suffix).c_str());
        kernels.step1 = module_.kernel(
            (std::string{"lfs_encoded_atrous_step1_"} + suffix).c_str());
        kernels.step2 = module_.kernel(
            (std::string{profile==ExecutionProfile::GlossyMeasurement?"lfs_encoded_atrous_step2_coverage_":"lfs_encoded_atrous_step2_"} + suffix).c_str());
        if(profile==ExecutionProfile::GlossyMeasurement)kernels.step4=module_.kernel((std::string{"lfs_encoded_atrous_refinement_coverage_"}+suffix).c_str());
        if(profile==ExecutionProfile::GlossyMeasurement) {
            kernels.final=module_.kernel((std::string{"lfs_encoded_atrous_view_final_"}+suffix).c_str());
            kernels.moving_panel=module_.kernel((std::string{"lfs_moving_panel_"}+suffix).c_str());
            kernels.resident_panel=module_.kernel((std::string{"lfs_resident_panel_"}+suffix).c_str());
            kernels.regular_routed_r2 = module_.kernel((std::string{"lfs_regular_routed_r2_"} + suffix).c_str());
            kernels.directional_refinement=module_.kernel((std::string{"lfs_regular_directional_refinement_"}+suffix).c_str());
            kernels.directional_chart=module_.kernel((std::string{"lfs_regular_directional_view_chart_"}+suffix).c_str());
            if(std::string_view(suffix)!="8k")
                kernels.regular_measured_chart=module_.kernel((std::string{"lfs_regular_measured_chart_"}+suffix).c_str());
        }
        if(profile==ExecutionProfile::GlossyMeasurement) {
            kernels.build_cells=module_.kernel((std::string{"lfs_build_cell_weights_"}+suffix).c_str());
            kernels.apply_cells=module_.kernel((std::string{"lfs_apply_cell_weights_"}+suffix).c_str());
        }
        if(profile!=ExecutionProfile::General) {
            kernels.geometry_check=module_.kernel("lfs_stationary_check_geometry");
            kernels.update_fixed=module_.kernel("lfs_update_fixed_observation");
            if (std::string_view(suffix)!="8k") {
                if(profile==ExecutionProfile::PlainCurrent)kernels.present_panel_samples=module_.kernel("lfs_present_panel_samples");
                kernels.collect_held_es=module_.kernel("lfs_collect_held_es");
                kernels.capture_held_es1=module_.kernel((std::string{"lfs_capture_held_es1_"}+suffix).c_str());
                kernels.capture_held_environment_es1=module_.kernel((std::string{"lfs_capture_held_environment_es1_"}+suffix).c_str());
                kernels.present_held_es=module_.kernel((std::string{"lfs_present_held_es_"}+suffix).c_str());
                if (profile==ExecutionProfile::GlossyMeasurement) {
                    kernels.build_held_view_cells=module_.kernel((std::string{"lfs_build_held_view_cells_"}+suffix).c_str());
                    kernels.present_held_view_cells=module_.kernel((std::string{"lfs_present_held_view_cells_"}+suffix).c_str());
                }
            }
            kernels.build_weights=module_.kernel((std::string{"lfs_build_static_weights_"}+suffix).c_str());
            kernels.cached_color=module_.kernel((std::string{"lfs_cached_static_color_"}+suffix).c_str());
        }
        if(profile!=ExecutionProfile::General) {
            kernels.build_epipolar=module_.kernel((std::string{"lfs_build_epipolar_"}+suffix).c_str());
            kernels.finish_registered=module_.kernel((std::string{"lfs_finish_registered_"}+suffix).c_str());
        }
        if(profile!=ExecutionProfile::General)kernels.step4_cached=module_.kernel((std::string{"lfs_encoded_atrous_view_chart_"}+suffix).c_str());
    };
    bind(k2_, "2k");
    bind(k4_, "4k");
    bind(k8_, "8k");
    if (!::lfs::denoise::encoded_atrous::native::valid(k2_, execution_profile_) ||
        !::lfs::denoise::encoded_atrous::native::valid(k4_, execution_profile_) ||
        !::lfs::denoise::encoded_atrous::native::valid(k8_, execution_profile_)) {
        return {backend::ErrorCode::BackendFailure,
                "Encoded A-Trous native entry lookup failed."};
    }
    device_ = &device;
    return {};
}

bool EncodedAtrousNativeRuntime::valid() const noexcept {
    return device_ != nullptr && module_.valid() &&
        ::lfs::denoise::encoded_atrous::native::valid(k2_, execution_profile_) &&
        ::lfs::denoise::encoded_atrous::native::valid(k4_, execution_profile_) &&
        ::lfs::denoise::encoded_atrous::native::valid(k8_, execution_profile_);
}

backend::Error EncodedAtrousNativeRuntime::enqueue(
    const ocarina::ExternalComputeKernel &kernel,
    const EncodedAtrousNativeParams &params,
    ocarina::uint3 grid,
    ocarina::uint3 block,
    ocarina::CommandBatch &commands) noexcept {
    const auto bytes = std::span<const std::byte>{
        reinterpret_cast<const std::byte *>(&params), sizeof(params)};
    ocarina::ExternalComputeDispatchCommand *command =
        kernel.dispatch(bytes, grid, block);
    const auto status = command ? command->status() : nullptr;
    if (!command || !status || !status->ok()) {
        if (command) command->recycle();
        return status
            ? status_error(*status, "Encoded A-Trous native dispatch")
            : backend::Error{backend::ErrorCode::BackendFailure,
                             "Encoded A-Trous dispatch returned no status."};
    }
    pending_.push_back(status);
    commands << command;
    return {};
}

backend::Error EncodedAtrousNativeRuntime::dispatch_atrous_pass(
    ExtentProfile profile,
    std::uint32_t step,
    const EncodedAtrousNativeParams &params,
    ocarina::CommandBatch &commands) noexcept {
    if ((execution_profile_ == ExecutionProfile::PlainCurrent &&
         !plain_current_compatible(params)) || (execution_profile_==ExecutionProfile::GlossyMeasurement && !glossy_measurement_compatible(params)) || !valid() || profile == ExtentProfile::Invalid ||
        (step != 1u && step != 2u && !(step==4u && execution_profile_==ExecutionProfile::GlossyMeasurement)) || params.count == 0u ||
        !retained_current_compatible(params)) {
        return {backend::ErrorCode::InvalidState,
                "Encoded A-Trous pass dispatch contract is invalid."};
    }
    const Kernels *kernels = profile == ExtentProfile::K2 ? &k2_ :
        profile == ExtentProfile::K4 ? &k4_ : &k8_;
    const std::uint32_t width = profile == ExtentProfile::K2 ? 4608u :
        profile == ExtentProfile::K4 ? 11520u : 23040u;
    const std::uint32_t height = profile == ExtentProfile::K2 ? 2048u :
        profile == ExtentProfile::K4 ? 2160u : 4320u;
    const bool plain = execution_profile_ == ExecutionProfile::PlainCurrent;
    const bool directional=step==4u && !params.resident_storage && params.register_motion;
    const bool carrier=plain && step==2u && !params.resident_storage;
    const bool routed = step == 2u && regular_r2_coverage(params) != RegularR2Coverage::Original;
    const bool parity_step2=step==2u && (plain || execution_profile_==ExecutionProfile::GlossyMeasurement);
    const unsigned rows = plain || parity_step2 || step==4u ? 8u : 16u;
    const unsigned spacing = step==4u?4u:(parity_step2 ? 2u : 1u);
    if((step==4u || (plain && step==2u)) && (!params.view_chart_block_count || params.view_chart_cache_wire!=(plain?0x56434233u:0x56434232u)))return {backend::ErrorCode::InvalidState,"Missing view-chart cache binding; no fallback."};
    const auto error=enqueue(
        directional ? kernels->directional_refinement :
        step == 4u ? kernels->step4 :
        step == 1u ? kernels->step1 :
        carrier ? kernels->regular_carrier_r2 :
        routed ? kernels->regular_routed_r2 : kernels->step2,
        params,
        ocarina::make_uint3((width + 32u * spacing - 1u) / (32u * spacing),
                            (height + rows * spacing - 1u) / (rows * spacing), spacing * spacing),
        ocarina::make_uint3(32u, rows, 1u), commands);
    if(error)return error;
    if ((plain && step == 2u) || step == 4u) {
        if (!kernels->step4_cached.valid() || params.view_chart_block_count == 0u)
            return {backend::ErrorCode::InvalidState,"Calibrated view reconstruction is unavailable."};
        auto view_params = params;
        // The first dispatch uses the opposite half-buffer as its destination.
        view_params.scratch_a = params.scratch_b;
        view_params.scratch_b = params.scratch_a;
        const bool moving_cache = !params.resident_storage && params.register_motion == 1u &&
            !params.hold_same && params.view_chart_source_count != 0u;
        const bool measured_chart=moving_cache && execution_profile_==ExecutionProfile::GlossyMeasurement &&
            profile!=ExtentProfile::K8 && params.regular_measurement_geometry && params.regular_repeat_state;
        if(measured_chart && !kernels->regular_measured_chart.valid())
            return {backend::ErrorCode::InvalidState,"Measured view chart kernel is unavailable."};
        const auto &chart=measured_chart?kernels->regular_measured_chart:
            (moving_cache?kernels->regular_moving_chart:(directional?kernels->directional_chart:kernels->step4_cached));
        if(auto err=enqueue(chart,view_params,ocarina::make_uint3(params.view_chart_block_count,1u,1u),
                            ocarina::make_uint3(256u,1u,1u),commands))return err;
        if(plain && !params.resident_storage && params.register_motion && params.panel_samples &&
           (profile==ExtentProfile::K2 || profile==ExtentProfile::K4)) {
            if(!kernels->present_panel_samples.valid())return {backend::ErrorCode::InvalidState,"Ordinary panel sample kernel unavailable."};
            return enqueue(kernels->present_panel_samples,params,ocarina::make_uint3((params.count+255u)/256u,1u,1u),
                           ocarina::make_uint3(256u,1u,1u),commands);
        }
        return {};
    }

    return {};
}

backend::Error EncodedAtrousNativeRuntime::dispatch_final(ExtentProfile profile,const EncodedAtrousNativeParams &params,ocarina::CommandBatch &commands) noexcept {
 if(execution_profile_!=ExecutionProfile::GlossyMeasurement||!valid()||profile==ExtentProfile::Invalid||!glossy_measurement_compatible(params)||!params.canonical||!params.view_chart_ids||!params.view_chart_offsets||params.view_chart_wire!=0x56434832u||!std::isfinite(params.final_energy)||!(params.final_energy>0.f))return {backend::ErrorCode::InvalidState,"Native view final binding invalid; no fallback."};
 const Kernels *k=profile==ExtentProfile::K2?&k2_:profile==ExtentProfile::K4?&k4_:&k8_;
 const auto &panel=params.resident_storage?k->resident_panel:
     (params.register_motion && !params.hold_same?k->moving_panel:k->final);
 return enqueue(panel,params,ocarina::make_uint3((params.count+255u)/256u,1u,1u),ocarina::make_uint3(256u,1u,1u),commands);
}

backend::Error EncodedAtrousNativeRuntime::dispatch_motion_field(ExtentProfile profile,
    const EncodedAtrousNativeParams &params, ocarina::CommandBatch &commands) noexcept {
    if(params.resident_storage || profile==ExtentProfile::Invalid || !params.regular_measurement_geometry || !params.atlas || !params.atlas_side || !params.scratch_a ||
       (execution_profile_==ExecutionProfile::GlossyMeasurement && !params.regular_repeat_state) ||
       !params.motion_cell_sites || !params.motion_cell_values || !params.motion_cell_count)
        return {backend::ErrorCode::InvalidState,"Incomplete aperture radiance field."};
    const auto &k=profile==ExtentProfile::K2?k2_:(profile==ExtentProfile::K4?k4_:k8_);
    if(!k.build_epipolar.valid())return {backend::ErrorCode::BackendFailure,"Missing aperture field kernel."};
    if(!params.static_view_count || params.atlas_cells%params.static_view_count)
        return {backend::ErrorCode::InvalidState,"Invalid diffuse aperture lattice."};
    if(params.hold_state) {
        if(!params.hold_same || !params.hold_previous || !k.collect_held_es.valid())
            return {backend::ErrorCode::InvalidState,"Missing repeated optical observations."};
        if(auto e=enqueue(k.collect_held_es,params,ocarina::make_uint3((params.count+255u)/256u,1u,1u),
                          ocarina::make_uint3(256u,1u,1u),commands))return e;
    }
    unsigned diffuse=2u*(params.atlas_cells/params.static_view_count);
    const bool bands=execution_profile_==ExecutionProfile::GlossyMeasurement &&
        (profile==ExtentProfile::K2 || profile==ExtentProfile::K4) && !params.resident_storage &&
        !params.hold_same && params.register_motion && params.static_view_count>1u;
    if(bands && profile==ExtentProfile::K4) {
        const unsigned side=params.atlas_side;
        const unsigned columns=(11520u+3u*side-1u)/(3u*side);
        const unsigned atlas_rows=(2160u+side-1u)/side;
        diffuse=2u*columns*((atlas_rows+1u)/2u);
    }
    const auto cells=bands?((params.motion_cell_count+255u)&~255u)+8u*diffuse:params.motion_cell_count+diffuse;
    return enqueue(k.build_epipolar,params,ocarina::make_uint3((cells+255u)/256u,1u,1u),
                   ocarina::make_uint3(256u,1u,1u),commands);
}

backend::Error EncodedAtrousNativeRuntime::dispatch_held_es(ExtentProfile profile,
    const EncodedAtrousNativeParams &params, ocarina::CommandBatch &commands) noexcept {
    if((profile!=ExtentProfile::K2 && profile!=ExtentProfile::K4) || !params.hold_state ||
       !params.hold_same || !params.hold_previous || !params.canonical || !params.view_chart_offsets)
        return {backend::ErrorCode::InvalidState,"Incomplete repeated optical observations."};
    const auto &k=profile==ExtentProfile::K2?k2_:k4_;
    if(!k.present_held_es.valid())return {backend::ErrorCode::BackendFailure,"Missing held optical reconstruction."};
    if(params.cell_coefficients && params.atlas_moments) {
        if(execution_profile_!=ExecutionProfile::GlossyMeasurement || params.resident_storage ||
           !params.atlas || !params.atlas_side || !params.atlas_cells || !params.regular_repeat_state ||
           !k.build_held_view_cells.valid() || !k.present_held_view_cells.valid())
            return {backend::ErrorCode::InvalidState,"Incomplete held view-cell reconstruction."};
        const auto block=ocarina::make_uint3(256u,1u,1u);
        if(auto e=enqueue(k.build_held_view_cells,params,ocarina::make_uint3((params.atlas_cells+255u)/256u,1u,1u),
                          block,commands))return e;
        return enqueue(k.present_held_view_cells,params,ocarina::make_uint3((params.count+255u)/256u,1u,1u),block,commands);
    }
    return enqueue(k.present_held_es,params,ocarina::make_uint3((params.count+255u)/256u,1u,1u),
                   ocarina::make_uint3(256u,1u,1u),commands);
}

backend::Error EncodedAtrousNativeRuntime::dispatch_resident_registered(ExtentProfile profile,
    const EncodedAtrousNativeParams &params, ocarina::CommandBatch &commands) noexcept {
    if(!params.resident_storage || profile==ExtentProfile::Invalid || !params.atlas || !params.atlas_side || !params.atlas_cells ||
       !params.scratch_a || !params.scratch_b || !params.canonical || !params.register_motion)
        return {backend::ErrorCode::InvalidState,"Incomplete active motion reconstruction."};
    const auto &k=profile==ExtentProfile::K2?k2_:(profile==ExtentProfile::K4?k4_:k8_);
    if(!k.finish_registered.valid())
        return {backend::ErrorCode::BackendFailure,"Missing motion registration kernel."};
    return enqueue(k.finish_registered,params,ocarina::make_uint3((params.count+255u)/256u,1u,1u),
                   ocarina::make_uint3(256u,1u,1u),commands);
}

backend::Error EncodedAtrousNativeRuntime::dispatch_stationary(
    ExtentProfile profile, const EncodedAtrousNativeParams &params, ocarina::CommandBatch &commands) noexcept {
    if(profile==ExtentProfile::Invalid || !params.atlas || !params.atlas_side || !params.atlas_cells ||
       !params.canonical || !params.scratch_a || !params.scratch_b || !params.view_chart_offsets)
        return {backend::ErrorCode::InvalidState,"Incomplete stationary-view atlas."};
    const auto &k=profile==ExtentProfile::K2?k2_:(profile==ExtentProfile::K4?k4_:k8_);
    const auto block=ocarina::make_uint3(256,1,1);
    if(!params.stationary_weights || !params.weights_refresh || !params.rigid_records ||
       !k.geometry_check.valid() || !k.build_weights.valid() || !k.cached_color.valid())
        return {backend::ErrorCode::ResourceMissing,"Missing stationary filter or rigid motion data."};
    const auto grid=ocarina::make_uint3((params.count+255u)/256u,1,1);
    if(auto e=enqueue(k.geometry_check,params,ocarina::make_uint3(1,1,1),block,commands))return e;
    if(params.hold_state){
        if(!params.hold_previous || !k.update_fixed.valid())return {backend::ErrorCode::InvalidState,"Missing fixed-ray transition state."};
        if(auto e=enqueue(k.update_fixed,params,grid,block,commands))return e;
    }
    if(auto e=enqueue(k.build_weights,params,grid,block,commands))return e;
    if(execution_profile_==ExecutionProfile::GlossyMeasurement) {
        if(!params.cell_coefficients || !k.build_cells.valid() || !k.apply_cells.valid())
            return {backend::ErrorCode::ResourceMissing,"Missing stationary cell coefficients or kernels."};
        const auto cells=ocarina::make_uint3((params.atlas_cells+255)/256,1,1);
        if(auto e=enqueue(k.build_cells,params,cells,block,commands))return e;
        if(auto e=enqueue(k.apply_cells,params,cells,block,commands))return e;
    }
    return enqueue(k.cached_color,params,grid,block,commands);
}

backend::Error EncodedAtrousNativeRuntime::dispatch_init_prefilter(
    ExtentProfile profile,
    const EncodedAtrousNativeParams &params,
    ocarina::CommandBatch &commands,
    bool fuse_step1) noexcept {
    if (fuse_step1 && execution_profile_ == ExecutionProfile::General) {
        return {backend::ErrorCode::InvalidState, "Fused ES requires an explicit current observation profile."};
    }
    if ((execution_profile_ == ExecutionProfile::PlainCurrent &&
         !plain_current_compatible(params)) || (execution_profile_==ExecutionProfile::GlossyMeasurement && !glossy_measurement_compatible(params)) || !valid() || profile == ExtentProfile::Invalid || params.count == 0u ||
        !retained_current_compatible(params)) {
        return {backend::ErrorCode::InvalidState,
                "Encoded init/prefilter dispatch contract is invalid."};
    }
    const Kernels *kernels = profile == ExtentProfile::K2 ? &k2_ :
        profile == ExtentProfile::K4 ? &k4_ : &k8_;
    const std::uint32_t width = profile == ExtentProfile::K2 ? 4608u :
        profile == ExtentProfile::K4 ? 11520u : 23040u;
    const std::uint32_t height = profile == ExtentProfile::K2 ? 2048u :
        profile == ExtentProfile::K4 ? 2160u : 4320u;
    const bool direct_environment=!params.resident_storage && params.register_motion && params.deterministic_environment;
    const auto &step1=params.hold_state?
        (direct_environment?kernels->capture_held_environment_es1:kernels->capture_held_es1):
        (direct_environment?kernels->regular_es1:kernels->init_step1);
    if(params.hold_state && (!fuse_step1 || params.resident_storage || !step1.valid()))
        return {backend::ErrorCode::ResourceMismatch,"Repeated-ray ES1 capture requires the regular fused producer."};
    return enqueue(
        fuse_step1?step1:kernels->init_prefilter, params,
        ocarina::make_uint3((width + 31u) / 32u,
                            (height + (15u)) / (16u), 1u),
        ocarina::make_uint3(32u, 16u, 1u), commands);
}

backend::Error EncodedAtrousNativeRuntime::validate_submission() const noexcept {
    if (pending_.empty()) {
        return {backend::ErrorCode::InvalidState,
                "Encoded A-Trous native submission is empty."};
    }
    for (const auto &status : pending_) {
        if (!status || !status->ok()) {
            return status
                ? status_error(*status, "Encoded A-Trous queued dispatch")
                : backend::Error{backend::ErrorCode::BackendFailure,
                                 "Encoded A-Trous lost dispatch status."};
        }
    }
    return {};
}

void EncodedAtrousNativeRuntime::clear_submission() noexcept {
    pending_.clear();
}

} // namespace lfs::denoise::encoded_atrous::native
