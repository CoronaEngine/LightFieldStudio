#pragma once
#include <cstdint>
namespace lfs::denoise::resident {
[[nodiscard]] constexpr bool requires_resident_framebuffer(std::uint32_t width,std::uint32_t height,
                                                           std::uint64_t physical_device_bytes) noexcept {
    // Excludes calibrated lookup storage and scene textures, so this is a
    // lower bound on the ordinary graph's allocations for this panel.
    constexpr std::uint64_t bytes_per_carrier=168u+32u+80u+16u+5u;
    const std::uint64_t carriers=3ull*width*height;
    return physical_device_bytes!=0 && carriers > physical_device_bytes/bytes_per_carrier;
}
}
