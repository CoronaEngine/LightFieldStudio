#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace lfs {

struct PreparedSceneResult {
    bool ok{};
    bool pipeline_framebuffer_rewritten{};
    bool camera_radiance_collector_rewritten{};
    bool integrator_denoiser_removed{};
    bool integrator_denoiser_rewritten{};
    uint32_t copied_relative_resource_count{};
    std::filesystem::path source_scene{};
    std::filesystem::path output_scene{};
    std::string source_hash{};
    std::string output_hash{};
    uint32_t width{1920u};
    uint32_t height{1080u};
    std::vector<std::string> errors{};
};

struct SceneCameraState {
    bool ok{};
    std::array<float, 3u> position{};
    std::array<float, 3u> target{};
    std::array<float, 3u> up{0.f, 1.f, 0.f};
    float vertical_fov_degrees{45.f};
    std::vector<std::string> errors{};
};

[[nodiscard]] PreparedSceneResult prepare_render_scene(
    const std::filesystem::path &source_scene,
    const std::filesystem::path &output_scene,
    uint32_t width = 1920u,
    uint32_t height = 1080u);

enum class StudioFramebuffer { Normal, Resident };
[[nodiscard]] PreparedSceneResult prepare_render_scene(
    const std::filesystem::path& input,const std::filesystem::path& output,
    uint32_t width,uint32_t height,StudioFramebuffer framebuffer);

[[nodiscard]] SceneCameraState load_scene_camera_state(
    const std::filesystem::path &source_scene);

} // namespace lfs
