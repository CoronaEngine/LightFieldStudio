#pragma once

#include "ssat/native/denoise_native_types.h"

#include <cstdint>
#include <type_traits>

namespace lfs::denoise::native {

struct EncodedCarrierComposeParams {
    std::uint64_t source{};
    std::uint64_t current_radiance{};
    std::uint64_t current_observation{};
    std::uint64_t canonical{};
    std::uint64_t frame_status{};
    std::uint32_t output_width{};
    std::uint32_t output_height{};
    std::uint32_t frame_status_enabled{};
    std::uint32_t deterministic_environment{};
};
static_assert(std::is_standard_layout_v<EncodedCarrierComposeParams>);
static_assert(std::is_trivially_copyable_v<EncodedCarrierComposeParams>);

} // namespace lfs::denoise::native
