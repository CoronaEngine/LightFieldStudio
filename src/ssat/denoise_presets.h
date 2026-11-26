#pragma once
#include "ssat/denoise_types.h"
#include "lightfield_framebuffer.h"
#include <memory>
#include <optional>
namespace lfs::denoise {
class DenoisePipeline;
std::optional<Lane> parse_lane(std::string_view) noexcept;
std::unique_ptr<DenoisePipeline> make_pipeline(Lane,LenticularParams,LightFieldGeometryParams);
}
