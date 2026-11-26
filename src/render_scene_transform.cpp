#include "render_scene_transform.h"

#include "file_hash.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string_view>
#include <unordered_set>

namespace lfs {
namespace {

std::string read_text_file(const std::filesystem::path &path, std::vector<std::string> &errors) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        errors.push_back("Missing scene file: " + path.string());
        return {};
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

size_t line_begin_for_pos(const std::string &text, size_t pos) {
    const size_t begin = text.rfind('\n', pos);
    return begin == std::string::npos ? 0u : begin + 1u;
}

std::string indentation_for_pos(const std::string &text, size_t pos) {
    const size_t begin = line_begin_for_pos(text, pos);
    std::string indent;
    for (size_t i = begin; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch != ' ' && ch != '\t') {
            break;
        }
        indent.push_back(ch);
    }
    return indent;
}

size_t find_key_in_range(
    const std::string &text,
    const std::string &key,
    size_t begin,
    size_t end) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = begin;
    while (pos < end) {
        pos = text.find(needle, pos);
        if (pos == std::string::npos || pos >= end) {
            return std::string::npos;
        }
        return pos;
    }
    return std::string::npos;
}

size_t object_value_start_for_key(
    const std::string &text,
    const std::string &key,
    size_t begin,
    size_t end) {
    const size_t key_pos = find_key_in_range(text, key, begin, end);
    if (key_pos == std::string::npos) {
        return std::string::npos;
    }
    const size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos || colon >= end) {
        return std::string::npos;
    }
    const size_t open = text.find('{', colon + 1u);
    if (open == std::string::npos || open >= end) {
        return std::string::npos;
    }
    return open;
}

size_t matching_object_end(const std::string &text, size_t open) {
    if (open >= text.size() || text[open] != '{') {
        return std::string::npos;
    }
    uint32_t depth = 0u;
    bool in_string = false;
    bool escape = false;
    for (size_t i = open; i < text.size(); ++i) {
        const char ch = text[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            continue;
        }
        if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            if (depth == 0u) {
                return std::string::npos;
            }
            --depth;
            if (depth == 0u) {
                return i;
            }
        }
    }
    return std::string::npos;
}

bool parse_number_at(
    const std::string &text,
    size_t &position,
    size_t end,
    float &value) {
    while (position < end &&
           (std::isspace(static_cast<unsigned char>(text[position])) ||
            text[position] == ',')) {
        ++position;
    }
    if (position >= end) {
        return false;
    }
    char *parsed_end = nullptr;
    value = std::strtof(text.c_str() + position, &parsed_end);
    if (parsed_end == text.c_str() + position ||
        parsed_end > text.c_str() + end) {
        return false;
    }
    position = static_cast<size_t>(parsed_end - text.c_str());
    return true;
}

bool read_float3_in_range(
    const std::string &text,
    const std::string &key,
    size_t begin,
    size_t end,
    std::array<float, 3u> &value) {
    const size_t key_pos = find_key_in_range(text, key, begin, end);
    if (key_pos == std::string::npos) {
        return false;
    }
    const size_t open = text.find('[', key_pos + key.size() + 2u);
    const size_t close = open == std::string::npos
        ? std::string::npos
        : text.find(']', open + 1u);
    if (open == std::string::npos || close == std::string::npos || close > end) {
        return false;
    }
    size_t cursor = open + 1u;
    return parse_number_at(text, cursor, close, value[0]) &&
        parse_number_at(text, cursor, close, value[1]) &&
        parse_number_at(text, cursor, close, value[2]);
}

bool read_float_in_range(
    const std::string &text,
    const std::string &key,
    size_t begin,
    size_t end,
    float &value) {
    const size_t key_pos = find_key_in_range(text, key, begin, end);
    if (key_pos == std::string::npos) {
        return false;
    }
    const size_t colon = text.find(':', key_pos + key.size() + 2u);
    if (colon == std::string::npos || colon >= end) {
        return false;
    }
    size_t cursor = colon + 1u;
    return parse_number_at(text, cursor, end, value);
}

std::string framebuffer_block(uint32_t width,uint32_t height,const std::string &indent,StudioFramebuffer mode) {
    const std::string child = indent + "\t";
    const std::string grandchild = child + "\t";
    std::ostringstream out;
    out << "{\n";
    out << child << "\"type\" : \"" << (mode==StudioFramebuffer::Resident ? "ssat_resident" : "normal") << "\",\n";
    out << child << "\"param\" : {\n";
    out << grandchild << "\"accumulation\" : false,\n";
    out << grandchild << "\"resolution\" : [\n";
    out << grandchild << "\t" << width << ",\n";
    out << grandchild << "\t" << height << "\n";
    out << grandchild << "],\n";
    out << grandchild << "\"fb_state\" : 0,\n";
    out << grandchild << "\"tone_mapper\" : {\n";
    out << grandchild << "\t\"type\": \"aces\",\n";
    out << grandchild << "\t\"param\" : {\n";
    out << grandchild << "\t\t\"exposure\" : 1.0\n";
    out << grandchild << "\t}\n";
    out << grandchild << "}\n";
    out << child << "}\n";
    out << indent << "}";
    return out.str();
}

bool replace_framebuffer_with_normal(
    std::string &text,
    uint32_t width,
    uint32_t height,
    std::vector<std::string> &errors, StudioFramebuffer mode) {
    const size_t pipeline_open = object_value_start_for_key(text, "pipeline", 0u, text.size());
    if (pipeline_open == std::string::npos) {
        errors.push_back("Could not locate pipeline object.");
        return false;
    }
    const size_t pipeline_end = matching_object_end(text, pipeline_open);
    if (pipeline_end == std::string::npos) {
        errors.push_back("Pipeline object is malformed.");
        return false;
    }
    const size_t framebuffer_open =
        object_value_start_for_key(text, "frame_buffer", pipeline_open, pipeline_end);
    if (framebuffer_open == std::string::npos) {
        errors.push_back("Could not locate pipeline.frame_buffer object.");
        return false;
    }
    const size_t framebuffer_end = matching_object_end(text, framebuffer_open);
    if (framebuffer_end == std::string::npos || framebuffer_end > pipeline_end) {
        errors.push_back("pipeline.frame_buffer object is malformed.");
        return false;
    }
    text.replace(
        framebuffer_open,
        framebuffer_end - framebuffer_open + 1u,
        framebuffer_block(width,height,indentation_for_pos(text,framebuffer_open),mode));
    return true;
}

bool replace_camera_radiance_collector_with_normal(
    std::string &text,
    uint32_t width,
    uint32_t height,
    std::vector<std::string> &errors, StudioFramebuffer mode) {
    const size_t camera_open = object_value_start_for_key(text, "camera", 0u, text.size());
    if (camera_open == std::string::npos) {
        errors.push_back("Could not locate camera object.");
        return false;
    }
    const size_t camera_end = matching_object_end(text, camera_open);
    if (camera_end == std::string::npos) {
        errors.push_back("Camera object is malformed.");
        return false;
    }
    const size_t radiance_open =
        object_value_start_for_key(text, "radiance_collector", camera_open, camera_end);
    if (radiance_open == std::string::npos) {
        errors.push_back("Could not locate camera.radiance_collector object.");
        return false;
    }
    const size_t radiance_end = matching_object_end(text, radiance_open);
    if (radiance_end == std::string::npos || radiance_end > camera_end) {
        errors.push_back("camera.radiance_collector object is malformed.");
        return false;
    }
    text.replace(
        radiance_open,
        radiance_end - radiance_open + 1u,
        framebuffer_block(width,height,indentation_for_pos(text,radiance_open),mode));
    return true;
}

bool remove_integrator_denoiser(
    std::string &text,
    size_t denoiser_key,
    size_t denoiser_value_end,
    std::vector<std::string> &errors) {
    size_t erase_begin = denoiser_key;
    size_t erase_end = denoiser_value_end + 1u;

    size_t scan = denoiser_key;
    while (scan > 0u && std::isspace(static_cast<unsigned char>(text[scan - 1u]))) {
        --scan;
    }
    if (scan > 0u && text[scan - 1u] == ',') {
        erase_begin = scan - 1u;
    } else {
        scan = erase_end;
        while (scan < text.size() && std::isspace(static_cast<unsigned char>(text[scan]))) {
            ++scan;
        }
        if (scan < text.size() && text[scan] == ',') {
            erase_end = scan + 1u;
        } else {
            errors.push_back("Could not remove integrator denoiser without leaving an invalid comma layout.");
            return false;
        }
    }
    text.erase(erase_begin, erase_end - erase_begin);
    return true;
}

bool set_integrator_max_depth_16(
    std::string &text,
    std::vector<std::string> &errors) {
    // Only match direct members; a denoiser may have its own nested parameters.
    const auto direct_key = [&](std::string_view key, size_t open, size_t end) {
        uint32_t depth = 1u;
        for (size_t i = open + 1u; i < end; ++i) {
            if (text.compare(i, 2u, "//") == 0) {
                const size_t newline = text.find('\n', i + 2u);
                if (newline == std::string::npos) break;
                i = newline;
            } else if (text.compare(i, 2u, "/*") == 0) {
                const size_t close = text.find("*/", i + 2u);
                if (close == std::string::npos) break;
                i = close + 1u;
            } else if (text[i] == '"') {
                const size_t first = i++;
                while (i < end && text[i] != '"') {
                    if (text[i] == '\\') ++i;
                    ++i;
                }
                const size_t colon = text.find_first_not_of(" \t\r\n", i + 1u);
                if (depth == 1u && i < end &&
                    (key.empty() || std::string_view{text}.substr(first + 1u, i - first - 1u) == key) &&
                    colon < end && text[colon] == ':') return first;
            } else if (text[i] == '{' || text[i] == '[') {
                ++depth;
            } else if (text[i] == '}' || text[i] == ']') {
                --depth;
            }
        }
        return std::string::npos;
    };
    const size_t integrator_open =
        object_value_start_for_key(text, "integrator", 0u, text.size());
    const size_t integrator_end = integrator_open == std::string::npos
        ? std::string::npos : matching_object_end(text, integrator_open);
    if (integrator_end == std::string::npos) {
        errors.emplace_back("Trace depth requires a valid integrator object.");
        return false;
    }
    const size_t param_key = direct_key("param", integrator_open, integrator_end);
    if (param_key == std::string::npos) {
        const bool has_members = direct_key({}, integrator_open, integrator_end) != std::string::npos;
        text.insert(integrator_open + 1u,
            std::string{"\"param\":{\"max_depth\":16}"} +
            (has_members ? "," : ""));
        return true;
    }
    const size_t param_colon = text.find(':', param_key);
    const size_t param_open = text.find_first_not_of(" \t\r\n", param_colon + 1u);
    if (param_open >= integrator_end || text[param_open] != '{') {
        errors.emplace_back("Trace depth requires an integrator parameter object.");
        return false;
    }
    const size_t param_end = matching_object_end(text, param_open);
    if (param_end == std::string::npos || param_end > integrator_end) {
        errors.emplace_back("Trace depth requires valid integrator parameters.");
        return false;
    }
    const size_t depth_key = direct_key("max_depth", param_open, param_end);
    if (depth_key == std::string::npos) {
        const bool has_members = direct_key({}, param_open, param_end) != std::string::npos;
        text.insert(param_open + 1u,
            std::string{"\"max_depth\":16"} + (has_members ? "," : ""));
        return true;
    }
    const size_t colon = text.find(':', depth_key);
    const size_t first = colon == std::string::npos
        ? std::string::npos : text.find_first_not_of(" \t\r\n", colon + 1u);
    size_t end = first;
    if (first != std::string::npos) {
        while (end < param_end && std::isdigit(static_cast<unsigned char>(text[end]))) ++end;
    }
    const size_t following = end == std::string::npos
        ? std::string::npos : text.find_first_not_of(" \t\r\n", end);
    if (first == std::string::npos || first >= param_end || end == first ||
        following == std::string::npos ||
        (text[following] != ',' && text[following] != '}')) {
        errors.emplace_back("Integrator max_depth must be an unsigned integer.");
        return false;
    }
    text.replace(first, end - first, "16");
    return true;
}

bool update_integrator_denoiser(
    std::string &text,
    std::vector<std::string> &errors) {

    const size_t integrator_open = object_value_start_for_key(text, "integrator", 0u, text.size());
    if (integrator_open == std::string::npos) {
        errors.push_back("Could not locate integrator object.");
        return false;
    }
    const size_t integrator_end = matching_object_end(text, integrator_open);
    if (integrator_end == std::string::npos) {
        errors.push_back("Integrator object is malformed.");
        return false;
    }
    const size_t denoiser_key = find_key_in_range(text, "denoiser", integrator_open, integrator_end);
    if (denoiser_key == std::string::npos) {
        return true;
    }
    const size_t colon = text.find(':', denoiser_key);
    const size_t denoiser_open = colon == std::string::npos
        ? std::string::npos
        : text.find('{', colon + 1u);
    const size_t denoiser_end = denoiser_open == std::string::npos
        ? std::string::npos
        : matching_object_end(text, denoiser_open);
    if (denoiser_open == std::string::npos ||
        denoiser_end == std::string::npos ||
        denoiser_end > integrator_end) {
        errors.push_back("Integrator denoiser object is malformed.");
        return false;
    }
    return remove_integrator_denoiser(text, denoiser_key, denoiser_end, errors);
}

bool contains_uncommented_token(const std::string &text, const std::string &token) {
    bool in_string = false;
    bool escape = false;
    bool in_line_comment = false;
    for (size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (in_line_comment) {
            if (ch == '\n') {
                in_line_comment = false;
            }
            continue;
        }
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '/' && i + 1u < text.size() && text[i + 1u] == '/') {
            in_line_comment = true;
            ++i;
            continue;
        }
        if (ch == '"') {
            in_string = true;
        }
        if (text.compare(i, token.size(), token) == 0) {
            return true;
        }
    }
    return false;
}

bool position_is_in_line_comment(const std::string &text, size_t pos) {
    const size_t begin = line_begin_for_pos(text, pos);
    const size_t comment = text.find("//", begin);
    return comment != std::string::npos && comment < pos;
}

std::vector<std::string> collect_active_relative_resource_paths(
    const std::string &text,
    std::vector<std::string> &errors) {
    std::vector<std::string> resources;
    std::unordered_set<std::string> seen;
    const std::string needle = "\"fn\"";
    size_t pos = 0u;
    while (true) {
        pos = text.find(needle, pos);
        if (pos == std::string::npos) {
            break;
        }
        if (position_is_in_line_comment(text, pos)) {
            pos += needle.size();
            continue;
        }
        const size_t colon = text.find(':', pos + needle.size());
        if (colon == std::string::npos) {
            errors.push_back("Scene contains an fn key without a value separator.");
            break;
        }
        size_t value_begin = colon + 1u;
        while (value_begin < text.size() && std::isspace(static_cast<unsigned char>(text[value_begin]))) {
            ++value_begin;
        }
        if (value_begin >= text.size() || text[value_begin] != '"') {
            pos = colon + 1u;
            continue;
        }
        const size_t value_end = text.find('"', value_begin + 1u);
        if (value_end == std::string::npos) {
            errors.push_back("Scene contains an unterminated fn string.");
            break;
        }
        const std::string resource = text.substr(value_begin + 1u, value_end - value_begin - 1u);
        const size_t output_key = text.rfind("\"output\"", pos);
        if (output_key != std::string::npos) {
            const size_t output_open = text.find('{', output_key);
            const size_t output_end = output_open == std::string::npos
                ? std::string::npos
                : matching_object_end(text, output_open);
            if (output_open != std::string::npos &&
                output_end != std::string::npos &&
                pos > output_open &&
                pos < output_end) {
                pos = value_end + 1u;
                continue;
            }
        }
        if (!resource.empty() &&
            !std::filesystem::path(resource).is_absolute() &&
            seen.insert(resource).second) {
            resources.push_back(resource);
        }
        pos = value_end + 1u;
    }
    return resources;
}

uint32_t copy_relative_resources_for_scene(
    const std::string &text,
    const std::filesystem::path &source_scene,
    const std::filesystem::path &output_scene,
    std::vector<std::string> &errors) {
    const std::filesystem::path source_dir = source_scene.parent_path();
    const std::filesystem::path output_dir = output_scene.parent_path();
    uint32_t copied = 0u;
    for (const std::string &resource : collect_active_relative_resource_paths(text, errors)) {
        const std::filesystem::path source_path = source_dir / std::filesystem::path(resource);
        const std::filesystem::path output_path = output_dir / std::filesystem::path(resource);
        if (!std::filesystem::exists(source_path)) {
            errors.push_back("Scene relative resource is missing: " + source_path.string());
            continue;
        }
        std::filesystem::create_directories(output_path.parent_path());
        std::error_code ec;
        std::filesystem::copy_file(
            source_path,
            output_path,
            std::filesystem::copy_options::overwrite_existing,
            ec);
        if (ec) {
            errors.push_back(
                "Could not copy scene relative resource " + source_path.string() +
                " -> " + output_path.string() + ": " + ec.message());
            continue;
        }
        ++copied;
    }
    return copied;
}

} // namespace

SceneCameraState load_scene_camera_state(
    const std::filesystem::path &source_scene) {
    SceneCameraState result;
    const std::string text = read_text_file(source_scene, result.errors);
    if (text.empty()) {
        return result;
    }
    const size_t camera_open = object_value_start_for_key(
        text, "camera", 0u, text.size());
    const size_t camera_end = camera_open == std::string::npos
        ? std::string::npos
        : matching_object_end(text, camera_open);
    if (camera_open == std::string::npos || camera_end == std::string::npos) {
        result.errors.push_back("Could not locate a valid camera object.");
        return result;
    }
    if (!read_float3_in_range(
            text, "position", camera_open, camera_end, result.position)) {
        result.errors.push_back("Camera position is missing or invalid.");
    }
    if (!read_float3_in_range(
            text, "target_pos", camera_open, camera_end, result.target)) {
        result.errors.push_back("Camera target_pos is missing or invalid.");
    }
    if (!read_float3_in_range(
            text, "up", camera_open, camera_end, result.up)) {
        result.errors.push_back("Camera up vector is missing or invalid.");
    }
    if (!read_float_in_range(
            text,
            "fov_y",
            camera_open,
            camera_end,
            result.vertical_fov_degrees)) {
        result.errors.push_back("Camera fov_y is missing or invalid.");
    }
    result.ok = result.errors.empty();
    return result;
}

PreparedSceneResult prepare_render_scene(
    const std::filesystem::path &source_scene,
    const std::filesystem::path &output_scene,
    uint32_t width,
    uint32_t height) {
    return prepare_render_scene(source_scene,output_scene,width,height,StudioFramebuffer::Normal);
}
PreparedSceneResult prepare_render_scene(const std::filesystem::path& source_scene,const std::filesystem::path& output_scene,
                                        uint32_t width,uint32_t height, StudioFramebuffer mode) {
    PreparedSceneResult result;
    result.source_scene = source_scene;
    result.output_scene = output_scene;
    result.width = width;
    result.height = height;
    result.source_hash = fnv1a64_file_hex(source_scene);

    if (width == 0u || height == 0u) {
        result.errors.push_back("render scene scene resolution must have positive width and height.");
        return result;
    }

    std::string text = read_text_file(source_scene, result.errors);
    if (text.empty()) {
        return result;
    }
    const bool framebuffer_ok =
        replace_framebuffer_with_normal(text, width, height, result.errors, mode);
    result.pipeline_framebuffer_rewritten = framebuffer_ok;
    const bool radiance_collector_ok =
        replace_camera_radiance_collector_with_normal(text, width, height, result.errors, mode);
    result.camera_radiance_collector_rewritten = radiance_collector_ok;
    const bool trace_depth_ok = set_integrator_max_depth_16(text, result.errors);
    const bool denoiser_ok =
        update_integrator_denoiser(text, result.errors);
    if (denoiser_ok) {
        result.integrator_denoiser_removed =
            true;
        result.integrator_denoiser_rewritten = false;
    }
    if (!framebuffer_ok || !radiance_collector_ok || !trace_depth_ok || !denoiser_ok) {
        return result;
    }
    if (contains_uncommented_token(text, "\"type\" : \"lightfield\"") ||
        contains_uncommented_token(text, "\"type\":\"lightfield\"")) {
        result.errors.push_back("render scene scene still contains an active lightfield framebuffer type.");
        return result;
    }
    if (contains_uncommented_token(text, "\"type\" : \"SSAT\"") ||
        contains_uncommented_token(text, "\"type\":\"SSAT\"")) {
        result.errors.push_back("render scene scene still contains an active SSAT denoiser.");
        return result;
    }

    std::filesystem::create_directories(output_scene.parent_path());
    result.copied_relative_resource_count =
        copy_relative_resources_for_scene(text, source_scene, output_scene, result.errors);
    if (!result.errors.empty()) {
        return result;
    }
    std::ofstream out(output_scene, std::ios::binary);
    if (!out) {
        result.errors.push_back("Could not write render scene scene: " + output_scene.string());
        return result;
    }
    out << text;
    out.close();
    result.output_hash = fnv1a64_file_hex(output_scene);
    result.ok = !result.output_hash.empty();
    if (!result.ok) {
        result.errors.push_back("Could not hash render scene scene: " + output_scene.string());
    }
    return result;
}

} // namespace lfs
