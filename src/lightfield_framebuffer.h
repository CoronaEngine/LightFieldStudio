#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace lfs {

struct Rgb8 {
    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};
};

struct LenticularParams {
    uint32_t width{384};
    uint32_t height{216};
    uint32_t view_count{45};
    double pitch{49.8};
    double slant{-0.12};
    double center{0.5};
    double offset{24.9};
};

struct LightFieldGeometryParams {
    double focal_distance{20.0};
    double fov_h_degrees{45.0};
    double aspect{1.7777777777777777};
    double array_angle_degrees{30.0};
    double focal_width{0.0};
    double focal_height{0.0};
};

struct LightFieldRaySample {
    uint32_t x{0};
    uint32_t y{0};
    uint32_t channel{0};
    uint32_t view_id{0};
    double focal_local_x{0.0};
    double focal_local_y{0.0};
    double focal_local_z{0.0};
    double camera_local_x{0.0};
    double camera_local_y{0.0};
    double camera_local_z{0.0};
    double direction_local_x{0.0};
    double direction_local_y{0.0};
    double direction_local_z{1.0};
};

struct EncodedImage {
    uint32_t width{0};
    uint32_t height{0};
    std::vector<Rgb8> pixels{};
};





class LightFieldFrameBuffer {
public:
    explicit LightFieldFrameBuffer(LenticularParams params);

    [[nodiscard]] const LenticularParams &params() const noexcept;
    [[nodiscard]] uint32_t pixel_width() const noexcept;
    [[nodiscard]] uint32_t pixel_height() const noexcept;
    [[nodiscard]] uint32_t subpixel_width() const noexcept;
    [[nodiscard]] size_t subpixel_count() const noexcept;

    void set_subpixel(uint32_t x, uint32_t y, uint32_t channel, Rgb8 value);
    [[nodiscard]] EncodedImage encode() const;

private:
    [[nodiscard]] size_t subpixel_index(uint32_t x, uint32_t y, uint32_t channel) const noexcept;

    LenticularParams params_{};
    std::vector<Rgb8> subpixels_{};
};

LenticularParams load_lenticular_params_from_device(
    const std::filesystem::path &device_path,
    std::vector<std::string> &errors);
LenticularParams load_lenticular_params_from_scene(
    const std::filesystem::path &scene_path,
    std::vector<std::string> &errors);
LightFieldGeometryParams load_lightfield_geometry_from_scene(
    const std::filesystem::path &scene_path,
    std::vector<std::string> &errors);
LenticularParams make_preview_params(const LenticularParams &source, uint32_t max_width);
uint32_t view_id_for_subpixel(uint32_t x, uint32_t y, uint32_t channel, const LenticularParams &params);
LightFieldRaySample lightfield_ray_sample_for_subpixel(
    uint32_t x,
    uint32_t y,
    uint32_t channel,
    const LenticularParams &lenticular,
    const LightFieldGeometryParams &geometry);

} // namespace lfs
