#include "ssat/denoise_presets.h"
#include "ssat/denoise_pipeline.h"
#include "ssat/phase2/encoded_atrous_spatial.h"
#include "ssat/phase3/physical_lf_reprojection_taa.h"
namespace lfs::denoise {
std::optional<Lane> parse_lane(std::string_view s) noexcept {
    if(s=="quality") return Lane::Quality;
    if(s=="performance") return Lane::Performance;
    return std::nullopt;
}
std::unique_ptr<DenoisePipeline> make_pipeline(Lane lane,
    LenticularParams lenticular, LightFieldGeometryParams geometry) {
    const bool quality=lane==Lane::Quality;
    EncodedConfig spatial{};
    spatial.atrous_iterations=quality?3u:2u;
    spatial.angular_compatibility_spatial=quality;
    spatial.angular_compatibility_prefilter_only=quality;
    spatial.responsive_angular_radiometry=quality;
    spatial.angular_compatibility_aperture_only=quality;
    spatial.responsive_radiometry_glossy_only=quality;
    PhysicalLfTaaConfig temporal{};
    temporal.lane=lane;temporal.lenticular=lenticular;temporal.geometry=geometry;
    temporal.responsive_temporal_rcrs=quality;
    temporal.responsive_log_history=quality;
    temporal.response_direction_glossy_only=quality;
    return std::make_unique<DenoisePipeline>(PipelineConfig{lane},
        std::make_unique<EncodedAtrousSpatial>(spatial),
        std::make_unique<PhysicalLfReprojectionTaa>(temporal));
}
}
