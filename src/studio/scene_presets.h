#pragma once

#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace lfs::studio {

struct DisplayMode {
    uint32_t width;
    uint32_t height;
    const char *id;
    const char *suffix;
};
inline constexpr std::array<DisplayMode, 3> display_modes{{
    {3840, 2160, "4k", "4K"}, {1536, 2048, "2k", "2K"}, {7680, 4320, "8k", "8K"}
}};

struct BundledScene {
    const char *name;
    const char *directory;
};
inline constexpr std::array<BundledScene, 3> scenes{{
    {"CBox", "cbox"}, {"Coffee", "coffee"}, {"SpaceShip", "spaceship"}
}};

inline int bundled_index(std::string id) {
    for (char &ch : id) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    for (size_t i = 0; i < scenes.size(); ++i) {
        if (id == scenes[i].directory) return static_cast<int>(i);
    }
    return -1;
}

inline std::filesystem::path bundled_path(const std::filesystem::path &runtime,
                                          int index, const DisplayMode &mode) {
    const auto &scene = scenes.at(static_cast<size_t>(index));
    return runtime / "assets" / scene.directory / (std::string(scene.name) + mode.suffix + ".json");
}

} // namespace lfs::studio
