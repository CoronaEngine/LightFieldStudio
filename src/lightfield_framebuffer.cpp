#include "lightfield_framebuffer.h"

#include "file_hash.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace lfs {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::string read_text_file(const std::filesystem::path &path, std::vector<std::string> &errors) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        errors.push_back("Missing file: " + path.string());
        return {};
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

double extract_number(const std::string &text, const std::string &key, double fallback) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    const size_t colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return fallback;
    }
    size_t begin = text.find_first_of("-0123456789", colon + 1u);
    if (begin == std::string::npos) {
        return fallback;
    }
    const size_t end = text.find_first_not_of("0123456789+-.eE", begin);
    try {
        return std::stod(text.substr(begin, end == std::string::npos ? end : end - begin));
    } catch (...) {
        return fallback;
    }
}

size_t find_object_body_start(const std::string &text, const std::string &key, size_t start_pos = 0u) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = text.find(needle, start_pos);
    if (key_pos == std::string::npos) {
        return std::string::npos;
    }
    const size_t colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return std::string::npos;
    }
    return text.find('{', colon + 1u);
}

double extract_number_from_pos(
    const std::string &text,
    size_t start_pos,
    const std::string &key,
    double fallback) {
    if (start_pos == std::string::npos) {
        return fallback;
    }
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = text.find(needle, start_pos);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    const size_t colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return fallback;
    }
    const size_t begin = text.find_first_of("-0123456789", colon + 1u);
    if (begin == std::string::npos) {
        return fallback;
    }
    const size_t end = text.find_first_not_of("0123456789+-.eE", begin);
    try {
        return std::stod(text.substr(begin, end == std::string::npos ? end : end - begin));
    } catch (...) {
        return fallback;
    }
}

std::string local_json_escape(const std::string &value) {
    std::string out;
    out.reserve(value.size() + 8u);
    for (char c : value) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(c);
            break;
        }
    }
    return out;
}

uint32_t clamp_dimension(double value, uint32_t fallback) {
    if (!std::isfinite(value) || value < 1.0) {
        return fallback;
    }
    const double rounded = std::round(value);
    if (rounded > static_cast<double>(UINT32_MAX)) {
        return fallback;
    }
    return static_cast<uint32_t>(rounded);
}

uint8_t to_u8(double value) {
    const double clamped = std::clamp(value, 0.0, 1.0);
    return static_cast<uint8_t>(std::lround(clamped * 255.0));
}

Rgb8 view_color(uint32_t view_id, uint32_t view_count, double x, double y) {
    const double denom = view_count > 1u ? static_cast<double>(view_count - 1u) : 1.0;
    const double t = static_cast<double>(view_id) / denom;
    const double wave = 0.5 + 0.5 * std::sin((t * 2.0 * kPi) + (x * kPi));
    return Rgb8{
        to_u8(0.15 + 0.75 * t),
        to_u8(0.12 + 0.65 * (1.0 - t) * (0.65 + 0.35 * y)),
        to_u8(0.10 + 0.80 * wave),
    };
}

LightFieldGeometryParams with_derived_geometry(LightFieldGeometryParams geometry) {
    const double fov_rad = geometry.fov_h_degrees * kPi / 180.0;
    geometry.focal_width = 2.0 * geometry.focal_distance * std::tan(fov_rad * 0.5);
    geometry.focal_height = geometry.aspect != 0.0 ? geometry.focal_width / geometry.aspect : 0.0;
    return geometry;
}

void write_u16(std::ofstream &stream, uint16_t value) {
    stream.put(static_cast<char>(value & 0xffu));
    stream.put(static_cast<char>((value >> 8u) & 0xffu));
}

void write_u32(std::ofstream &stream, uint32_t value) {
    stream.put(static_cast<char>(value & 0xffu));
    stream.put(static_cast<char>((value >> 8u) & 0xffu));
    stream.put(static_cast<char>((value >> 16u) & 0xffu));
    stream.put(static_cast<char>((value >> 24u) & 0xffu));
}

} // namespace

LenticularParams load_lenticular_params_from_device(
    const std::filesystem::path &device_path,
    std::vector<std::string> &errors) {
    LenticularParams params;
    const std::string text = read_text_file(device_path, errors);
    if (text.empty()) {
        return params;
    }

    params.width = clamp_dimension(extract_number(text, "width", params.width), params.width);
    params.height = clamp_dimension(extract_number(text, "height", params.height), params.height);
    const uint32_t views_x = clamp_dimension(extract_number(text, "views_x", 9.0), 9u);
    const uint32_t views_y = clamp_dimension(extract_number(text, "views_y", 5.0), 5u);
    params.view_count = std::max(1u, views_x * views_y);
    params.pitch = std::max(0.001, extract_number(text, "pitch", params.pitch));
    params.slant = extract_number(text, "slant", params.slant);
    params.center = extract_number(text, "center", params.center);
    params.offset = params.pitch * params.center;
    return params;
}

LenticularParams load_lenticular_params_from_scene(
    const std::filesystem::path &scene_path,
    std::vector<std::string> &errors) {
    LenticularParams params;
    const std::string text = read_text_file(scene_path, errors);
    if (text.empty()) {
        return params;
    }

    const size_t pipeline_pos = find_object_body_start(text, "pipeline");
    const size_t framebuffer_pos = find_object_body_start(text, "frame_buffer", pipeline_pos);
    const size_t lenticular_pos = find_object_body_start(text, "lenticular", framebuffer_pos);
    if (lenticular_pos == std::string::npos) {
        errors.push_back("Scene does not contain pipeline.frame_buffer.param.lenticular: " + scene_path.string());
        return params;
    }

    const double width = extract_number_from_pos(text, lenticular_pos, "res_w", std::numeric_limits<double>::quiet_NaN());
    const double height = extract_number_from_pos(text, lenticular_pos, "res_h", std::numeric_limits<double>::quiet_NaN());
    const double view_count = extract_number_from_pos(text, lenticular_pos, "num_views", std::numeric_limits<double>::quiet_NaN());
    const double pitch = extract_number_from_pos(text, lenticular_pos, "pe", std::numeric_limits<double>::quiet_NaN());
    const double angle = extract_number_from_pos(text, lenticular_pos, "angle", std::numeric_limits<double>::quiet_NaN());
    const double offset = extract_number_from_pos(text, lenticular_pos, "offset", std::numeric_limits<double>::quiet_NaN());

    if (!std::isfinite(width) || !std::isfinite(height) || !std::isfinite(view_count) ||
        !std::isfinite(pitch) || !std::isfinite(angle) || !std::isfinite(offset)) {
        errors.push_back("Scene lenticular parameters are incomplete: " + scene_path.string());
        return params;
    }

    params.width = clamp_dimension(width, params.width);
    params.height = clamp_dimension(height, params.height);
    params.view_count = std::max(1u, clamp_dimension(view_count, params.view_count));
    params.pitch = std::max(0.001, pitch);
    params.slant = angle;
    params.offset = offset;
    params.center = params.pitch > 0.0 ? params.offset / params.pitch : params.center;
    return params;
}

LightFieldGeometryParams load_lightfield_geometry_from_scene(
    const std::filesystem::path &scene_path,
    std::vector<std::string> &errors) {
    LightFieldGeometryParams geometry = with_derived_geometry(LightFieldGeometryParams{});
    const std::string text = read_text_file(scene_path, errors);
    if (text.empty()) {
        return geometry;
    }

    const size_t pipeline_pos = find_object_body_start(text, "pipeline");
    const size_t framebuffer_pos = find_object_body_start(text, "frame_buffer", pipeline_pos);
    const size_t geometry_pos = find_object_body_start(text, "geometry", framebuffer_pos);
    if (geometry_pos == std::string::npos) {
        errors.push_back("Scene does not contain pipeline.frame_buffer.param.geometry: " + scene_path.string());
        return geometry;
    }

    const double focal_distance =
        extract_number_from_pos(text, geometry_pos, "d_f", std::numeric_limits<double>::quiet_NaN());
    const double fov_h_degrees =
        extract_number_from_pos(text, geometry_pos, "fov_h_deg", std::numeric_limits<double>::quiet_NaN());
    const double aspect =
        extract_number_from_pos(text, geometry_pos, "aspect", std::numeric_limits<double>::quiet_NaN());
    const double array_angle_degrees =
        extract_number_from_pos(text, geometry_pos, "array_angle_deg", std::numeric_limits<double>::quiet_NaN());

    if (!std::isfinite(focal_distance) || !std::isfinite(fov_h_degrees) ||
        !std::isfinite(aspect) || !std::isfinite(array_angle_degrees)) {
        errors.push_back("Scene light-field geometry parameters are incomplete: " + scene_path.string());
        return geometry;
    }

    geometry.focal_distance = focal_distance;
    geometry.fov_h_degrees = fov_h_degrees;
    geometry.aspect = aspect;
    geometry.array_angle_degrees = array_angle_degrees;
    return with_derived_geometry(geometry);
}

LenticularParams make_preview_params(const LenticularParams &source, uint32_t max_width) {
    LenticularParams preview = source;
    if (max_width == 0u || source.width <= max_width) {
        return preview;
    }

    const double scale = static_cast<double>(max_width) / static_cast<double>(source.width);
    preview.width = max_width;
    preview.height = std::max(1u, static_cast<uint32_t>(std::lround(static_cast<double>(source.height) * scale)));
    preview.pitch = std::max(0.001, source.pitch * scale);
    preview.offset = preview.pitch * source.center;
    return preview;
}

LightFieldFrameBuffer::LightFieldFrameBuffer(LenticularParams params)
    : params_(params) {
    params_.width = std::max(1u, params_.width);
    params_.height = std::max(1u, params_.height);
    params_.view_count = std::max(1u, params_.view_count);
    subpixels_.resize(subpixel_count());
}

const LenticularParams &LightFieldFrameBuffer::params() const noexcept {
    return params_;
}

uint32_t LightFieldFrameBuffer::pixel_width() const noexcept {
    return params_.width;
}

uint32_t LightFieldFrameBuffer::pixel_height() const noexcept {
    return params_.height;
}

uint32_t LightFieldFrameBuffer::subpixel_width() const noexcept {
    return params_.width * 3u;
}

size_t LightFieldFrameBuffer::subpixel_count() const noexcept {
    return static_cast<size_t>(subpixel_width()) * static_cast<size_t>(params_.height);
}

void LightFieldFrameBuffer::set_subpixel(uint32_t x, uint32_t y, uint32_t channel, Rgb8 value) {
    if (x >= params_.width || y >= params_.height || channel >= 3u) {
        return;
    }
    subpixels_[subpixel_index(x, y, channel)] = value;
}

EncodedImage LightFieldFrameBuffer::encode() const {
    EncodedImage image;
    image.width = params_.width;
    image.height = params_.height;
    image.pixels.resize(static_cast<size_t>(image.width) * static_cast<size_t>(image.height));

    for (uint32_t y = 0; y < image.height; ++y) {
        for (uint32_t x = 0; x < image.width; ++x) {
            const Rgb8 r = subpixels_[subpixel_index(x, y, 0u)];
            const Rgb8 g = subpixels_[subpixel_index(x, y, 1u)];
            const Rgb8 b = subpixels_[subpixel_index(x, y, 2u)];
            image.pixels[static_cast<size_t>(y) * image.width + x] = Rgb8{r.r, g.g, b.b};
        }
    }
    return image;
}

size_t LightFieldFrameBuffer::subpixel_index(uint32_t x, uint32_t y, uint32_t channel) const noexcept {
    return static_cast<size_t>(y) * static_cast<size_t>(subpixel_width()) +
           static_cast<size_t>(x) * 3u + static_cast<size_t>(channel);
}

uint32_t view_id_for_subpixel(uint32_t x, uint32_t y, uint32_t channel, const LenticularParams &params) {
    const uint32_t safe_view_count = std::max(1u, params.view_count);
    const float period = std::max(0.001f, static_cast<float>(params.pitch));
    const float d = 3.0f * static_cast<float>(x) +
                    3.0f * static_cast<float>(y) *
                        std::tan(static_cast<float>(params.slant)) +
                    static_cast<float>(std::min(channel, 2u)) +
                    static_cast<float>(params.offset);
    const float wrapped = d - std::floor(d / period) * period;
    const float view_width = period / static_cast<float>(safe_view_count);
    uint32_t view_id = static_cast<uint32_t>(std::floor(wrapped / view_width));
    view_id = std::min(view_id, safe_view_count - 1u);
    return (safe_view_count - 1u) - view_id;
}

LightFieldRaySample lightfield_ray_sample_for_subpixel(
    uint32_t x,
    uint32_t y,
    uint32_t channel,
    const LenticularParams &lenticular,
    const LightFieldGeometryParams &geometry) {
    LightFieldRaySample sample;
    sample.x = x;
    sample.y = y;
    sample.channel = std::min(channel, 2u);
    sample.view_id = view_id_for_subpixel(x, y, sample.channel, lenticular);

    const double pixel_width = geometry.focal_width / static_cast<double>(std::max(1u, lenticular.width));
    const double pixel_height = geometry.focal_height / static_cast<double>(std::max(1u, lenticular.height));
    const double subpixel_offset = (static_cast<double>(sample.channel) + 0.5) * (pixel_width / 3.0);
    sample.focal_local_x =
        -geometry.focal_width * 0.5 + static_cast<double>(x) * pixel_width + subpixel_offset;
    sample.focal_local_y =
        geometry.focal_height * 0.5 - (static_cast<double>(y) + 0.5) * pixel_height;
    sample.focal_local_z = geometry.focal_distance;

    const double denom = lenticular.view_count > 1u ? static_cast<double>(lenticular.view_count - 1u) : 1.0;
    const double u = lenticular.view_count > 1u ? static_cast<double>(sample.view_id) / denom : 0.5;
    const double array_angle_rad = geometry.array_angle_degrees * kPi / 180.0;
    const double view_angle = (u - 0.5) * array_angle_rad;
    sample.camera_local_x = geometry.focal_distance * std::tan(view_angle);
    sample.camera_local_y = 0.0;
    sample.camera_local_z = 0.0;

    const double dx = sample.focal_local_x - sample.camera_local_x;
    const double dy = sample.focal_local_y - sample.camera_local_y;
    const double dz = sample.focal_local_z - sample.camera_local_z;
    const double len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (len > 0.0) {
        sample.direction_local_x = dx / len;
        sample.direction_local_y = dy / len;
        sample.direction_local_z = dz / len;
    }
    return sample;
}


} // namespace lfs
