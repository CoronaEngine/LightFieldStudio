#include "ssat/phase3/native/early_measurement_runtime.h"
#include "ssat/runtime_paths.h"
#include <fstream>
#include <span>

namespace lfs::denoise::mr_native {
backend::Error Runtime::initialize(ocarina::Device &device,bool logarithmic) noexcept {

    if (device_==&device && logarithmic_==logarithmic && module_.valid()) return {};
    try {
        const auto path=runtime_directory()/"lfs_early_measurement.fatbin";
        std::ifstream stream(path,std::ios::binary|std::ios::ate);
        if(!stream || stream.tellg()<=0) return {backend::ErrorCode::BackendFailure,"MR native module missing."};
        std::vector<std::byte> bytes(static_cast<size_t>(stream.tellg()));stream.seekg(0);stream.read(reinterpret_cast<char *>(bytes.data()),bytes.size());
        if(!stream) return {backend::ErrorCode::BackendFailure,"MR native module read failed."};
        module_=device.create_external_compute_module({bytes.data(),bytes.size()},"lfs.measurement_reconstruction");
        measurement_=module_.kernel(logarithmic?"lfs_early_measurement_rgb_primary_ray":"lfs_early_measurement_f16_rms");
        regular_measurement_ = module_.kernel(logarithmic
            ? "lfs_regular_measurement_rgb_primary_ray" : "lfs_regular_measurement_f16_rms");
        certified_measurement_ = module_.kernel(logarithmic
            ? "lfs_certified_measurement_rgb_primary_ray" : "lfs_certified_measurement_f16_rms");
        if(logarithmic)
            certified_vertex_measurement_ = module_.kernel("lfs_certified_vertex_measurement_rgb_primary_ray");
        stationary_ = module_.kernel(logarithmic ? "lfs_stationary_measurement_glossy" : "lfs_stationary_measurement");
        vertex_measurement_=module_.kernel(logarithmic?"lfs_vertex_measurement_rgb_primary_ray":"lfs_vertex_measurement_f16_rms");
        commit_geometry_=module_.kernel("lfs_commit_visible_geometry");
        restore_geometry_=module_.kernel("lfs_restore_native_history");
        logarithmic_=logarithmic;
        if(!module_.valid() || !measurement_.valid() || !regular_measurement_.valid() || !certified_measurement_.valid() ||
           (logarithmic && !certified_vertex_measurement_.valid()) || !vertex_measurement_.valid() || !stationary_.valid())
            return {backend::ErrorCode::BackendFailure,"MR native entry lookup failed."};
        device_=&device;return {};
    } catch(const std::exception &e) { return {backend::ErrorCode::BackendFailure,e.what()}; }
}
backend::Error Runtime::dispatch(const Params &params,ocarina::CommandBatch &commands, bool regular_layout) noexcept {

    if((params.measurement_write&7u)!=0u||(params.measurement_read&7u)!=0u)return {backend::ErrorCode::ResourceMismatch,"MR16 history addresses require 8B alignment."};
    if(params.history_storage!=kHistoryFp16RgbRms)return {backend::ErrorCode::InvalidState,"Measurement history storage ABI mismatch."};
    if(!device_) return {backend::ErrorCode::InvalidState,"MR dispatch not prepared."};
    if(params.geometry.current_denoiser_path_guide!=0u)return {backend::ErrorCode::ResourceMismatch,"Physical-ray RGB history does not consume sampled secondary guides."};
    const auto safe_halo=[](std::int32_t v) {return v>=-256 && v<=256;};
    const bool certified=logarithmic_ && regular_layout && params.rgb_basis_valid && params.current_panel_samples &&
        !params.stationary_radiance && !params.committed_measurement && params.geometry.view_count<=64u &&
        ((params.geometry.output_width==1536u && params.geometry.output_height==2048u) ||
         (params.geometry.output_width==3840u && params.geometry.output_height==2160u)) &&
        safe_halo(params.rgb_b0x) && safe_halo(params.rgb_b0y) &&
        safe_halo(params.rgb_b1x) && safe_halo(params.rgb_b1y);
    const auto &kernel=params.stationary_radiance ? stationary_ :
        (!regular_layout ? measurement_ : (params.vertex_frame.hits ?
            (certified && logarithmic_ ? certified_vertex_measurement_ : vertex_measurement_) :
            (certified ? certified_measurement_ : regular_measurement_)));
    const auto bytes=std::span<const std::byte>{reinterpret_cast<const std::byte *>(&params),sizeof(params)};
    if(params.committed_measurement && params.geometry.history_valid){
        if(!restore_geometry_.valid() || !params.native_second_hit || !params.geometry.native_first_history_head)
            return {backend::ErrorCode::ResourceMismatch,"Incomplete post-trace history recovery."};
        const unsigned count=3u*params.geometry.output_width*params.geometry.output_height;
        auto *restore=restore_geometry_.dispatch(bytes,ocarina::make_uint3((count+255u)/256u,1u,1u),ocarina::make_uint3(256u,1u,1u));
        const auto checked=restore?restore->status():nullptr;
        if(!restore || !checked || !checked->ok()){
            if(restore)restore->recycle();
            return {backend::ErrorCode::BackendFailure,"Original complete history recovery failed."};
        }
        pending_.push_back(checked);commands<<restore;
    }
    auto *command=kernel.dispatch(bytes,ocarina::make_uint3((params.geometry.output_width*3u+31u)/32u,(params.geometry.output_height+3u)/4u,1u),ocarina::make_uint3(32u,4u,1u));
    const auto status=command?command->status():nullptr;
    if(!command || !status || !status->ok()) {
        if(command)command->recycle();
        return {backend::ErrorCode::BackendFailure,"MR native dispatch submission failed."};
    }
    pending_.push_back(status);commands<<command;
    return {};
}
backend::Error Runtime::commit_history(const Params& params,ocarina::CommandBatch& commands) noexcept {
    if(params.geometry.history_geometry_write || params.history_metadata_staging){
        const auto bytes=std::span<const std::byte>{reinterpret_cast<const std::byte*>(&params),sizeof(params)};
        if(!commit_geometry_.valid())return {backend::ErrorCode::BackendFailure,"Visible geometry commit is unavailable."};
        const unsigned count=params.geometry.output_width*3u*params.geometry.output_height;
        auto* commit=commit_geometry_.dispatch(bytes,ocarina::make_uint3((count+255)/256,1u,1u),ocarina::make_uint3(256,1u,1u));
        const auto result=commit?commit->status():nullptr;
        if(!commit || !result || !result->ok()){
            if(commit)commit->recycle();
            return {backend::ErrorCode::BackendFailure,"Visible geometry commit failed."};
        }
        pending_.push_back(result);commands<<commit;
    }

    return {};
}
backend::Error Runtime::validate() const noexcept {
    for(const auto &status:pending_)if(!status || !status->ok())
        return {backend::ErrorCode::BackendFailure,"MR native execution failed."};
    return {};
}
}
