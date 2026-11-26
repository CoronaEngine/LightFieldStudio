#include "ssat/phase1/phase_mapping.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lfs::lightfield {
namespace {

constexpr double kPi = 3.14159265358979323846;

[[nodiscard]] bool finite(PhasePoint3 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z);
}

[[nodiscard]] PhasePoint3 add(PhasePoint3 lhs, PhasePoint3 rhs) noexcept {
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

[[nodiscard]] PhasePoint3 subtract(
    PhasePoint3 lhs,
    PhasePoint3 rhs) noexcept {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

[[nodiscard]] PhasePoint3 multiply(PhasePoint3 value, double scale) noexcept {
    return {value.x * scale, value.y * scale, value.z * scale};
}

[[nodiscard]] double dot(PhasePoint3 lhs, PhasePoint3 rhs) noexcept {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

[[nodiscard]] PhasePoint3 cross(PhasePoint3 lhs, PhasePoint3 rhs) noexcept {
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x};
}

[[nodiscard]] double length(PhasePoint3 value) noexcept {
    return std::sqrt(dot(value, value));
}

[[nodiscard]] PhasePoint3 normalized(PhasePoint3 value) noexcept {
    const double magnitude = length(value);
    return magnitude > 0.0
        ? multiply(value, 1.0 / magnitude)
        : PhasePoint3{0.0, 0.0, 1.0};
}

[[nodiscard]] PhasePoint3 transform_point_impl(
    const PhaseTransform &transform,
    PhasePoint3 point) noexcept {
    return add(
        transform.translation,
        add(
            multiply(transform.axis_x, point.x),
            add(
                multiply(transform.axis_y, point.y),
                multiply(transform.axis_z, point.z))));
}

[[nodiscard]] uint32_t view_id(
    const PhaseLayout &layout,
    uint32_t encoded_x,
    uint32_t encoded_y) noexcept {
    const float period = std::max(static_cast<float>(layout.pitch), 0.001f);
    const float d = static_cast<float>(encoded_x) +
        3.0f * static_cast<float>(encoded_y) *
            std::tan(static_cast<float>(layout.slant)) +
        static_cast<float>(layout.offset);
    const float wrapped = d - std::floor(d / period) * period;
    const float view_width = period / static_cast<float>(layout.view_count);
    const uint32_t period_view = std::min(
        static_cast<uint32_t>(std::floor(wrapped / view_width)),
        layout.view_count - 1u);
    return layout.view_count - 1u - period_view;
}

[[nodiscard]] double candidate_score(
    const PhaseCandidate &candidate,
    const PhaseBandwidth &bandwidth) noexcept {
    const double dx = candidate.x_error_lattice / bandwidth.sigma_x_lattice;
    const double dy = candidate.y_error_lattice / bandwidth.sigma_y_lattice;
    const double du = candidate.u_error_physical / bandwidth.sigma_u_physical;
    return dx * dx + dy * dy + du * du;
}

[[nodiscard]] bool candidate_less(
    const PhaseCandidate &lhs,
    const PhaseCandidate &rhs,
    const PhaseBandwidth &bandwidth) noexcept {
    const double lhs_score = candidate_score(lhs, bandwidth);
    const double rhs_score = candidate_score(rhs, bandwidth);
    if (lhs_score != rhs_score) {
        return lhs_score < rhs_score;
    }
    if (lhs.phase.subpixel.encoded_y != rhs.phase.subpixel.encoded_y) {
        return lhs.phase.subpixel.encoded_y < rhs.phase.subpixel.encoded_y;
    }
    return lhs.phase.subpixel.encoded_x < rhs.phase.subpixel.encoded_x;
}

} // namespace

bool PhaseTransform::valid_rigid(double tolerance) const noexcept {
    if (!(tolerance > 0.0) || !finite(axis_x) || !finite(axis_y) ||
        !finite(axis_z) || !finite(translation)) {
        return false;
    }
    const bool unit_axes =
        std::abs(length(axis_x) - 1.0) <= tolerance &&
        std::abs(length(axis_y) - 1.0) <= tolerance &&
        std::abs(length(axis_z) - 1.0) <= tolerance;
    const bool orthogonal =
        std::abs(dot(axis_x, axis_y)) <= tolerance &&
        std::abs(dot(axis_x, axis_z)) <= tolerance &&
        std::abs(dot(axis_y, axis_z)) <= tolerance;
    const double determinant = dot(axis_x, cross(axis_y, axis_z));
    return unit_axes && orthogonal &&
        std::abs(std::abs(determinant) - 1.0) <= tolerance;
}

uint32_t PhaseLayout::encoded_width() const noexcept {
    return output_width <= std::numeric_limits<uint32_t>::max() / 3u
        ? output_width * 3u
        : 0u;
}

double PhaseLayout::focal_lattice_width() const noexcept {
    const uint32_t width = encoded_width();
    return width > 0u ? focal_width / static_cast<double>(width) : 0.0;
}

double PhaseLayout::focal_lattice_height() const noexcept {
    return output_height > 0u
        ? focal_height / static_cast<double>(output_height)
        : 0.0;
}

bool PhaseLayout::valid() const noexcept {
    const double values[] = {
        pitch,
        slant,
        offset,
        focal_distance,
        focal_width,
        focal_height,
        array_angle_degrees};
    for (double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return output_width > 0u && output_height > 0u && view_count > 0u &&
        encoded_width() > 0u && pitch > 1.0e-6 && focal_distance > 1.0e-6 &&
        focal_width > 1.0e-6 && focal_height > 1.0e-6 &&
        std::abs(array_angle_degrees) < 179.0 &&
        std::isfinite(std::tan(slant)) && local_to_world.valid_rigid();
}

bool PhaseBandwidth::valid() const noexcept {
    return std::isfinite(sigma_x_lattice) &&
        std::isfinite(sigma_y_lattice) &&
        std::isfinite(sigma_u_physical) && sigma_x_lattice > 0.0 &&
        sigma_y_lattice > 0.0 && sigma_u_physical > 0.0;
}

PhaseLayout make_phase_layout(
    const LenticularParams &lenticular,
    const LightFieldGeometryParams &geometry,
    PhaseTransform local_to_world) noexcept {
    return {
        lenticular.width,
        lenticular.height,
        lenticular.view_count,
        lenticular.pitch,
        lenticular.slant,
        lenticular.offset,
        geometry.focal_distance,
        geometry.focal_width,
        geometry.focal_height,
        geometry.array_angle_degrees,
        local_to_world};
}

std::optional<PhaseCoordinate> phi(
    const PhaseLayout &layout,
    uint32_t encoded_x,
    uint32_t encoded_y) noexcept {
    if (!layout.valid() || encoded_x >= layout.encoded_width() ||
        encoded_y >= layout.output_height) {
        return std::nullopt;
    }

    PhaseCoordinate coordinate{};
    coordinate.subpixel.encoded_x = encoded_x;
    coordinate.subpixel.encoded_y = encoded_y;
    coordinate.subpixel.pixel_x = encoded_x / 3u;
    coordinate.subpixel.channel = encoded_x % 3u;
    coordinate.subpixel.focal_lattice_x =
        static_cast<double>(encoded_x) + 0.5;
    coordinate.subpixel.focal_lattice_y =
        static_cast<double>(encoded_y) + 0.5;
    coordinate.view_id = view_id(layout, encoded_x, encoded_y);

    const double lattice_width = layout.focal_lattice_width();
    const double lattice_height = layout.focal_lattice_height();
    coordinate.x = {
        -layout.focal_width * 0.5 +
            coordinate.subpixel.focal_lattice_x * lattice_width,
        layout.focal_height * 0.5 -
            coordinate.subpixel.focal_lattice_y * lattice_height,
        layout.focal_distance};

    const double view_u = layout.view_count > 1u
        ? static_cast<double>(coordinate.view_id) /
            static_cast<double>(layout.view_count - 1u)
        : 0.5;
    const double view_angle = (view_u - 0.5) *
        layout.array_angle_degrees * (kPi / 180.0);
    coordinate.u = {
        layout.focal_distance * std::tan(view_angle),
        0.0,
        0.0};
    coordinate.world_origin = transform_point_impl(
        layout.local_to_world, coordinate.u);
    const PhasePoint3 focal_world = transform_point_impl(
        layout.local_to_world, coordinate.x);
    coordinate.world_direction = normalized(subtract(
        focal_world, coordinate.world_origin));
    return coordinate;
}

std::optional<PlaneIntersection> intersect_plane(
    PhasePoint3 origin,
    PhasePoint3 direction,
    double plane_z,
    double parallel_epsilon) noexcept {
    if (!finite(origin) || !finite(direction) || !std::isfinite(plane_z) ||
        !std::isfinite(parallel_epsilon) || !(parallel_epsilon > 0.0) ||
        std::abs(direction.z) <= parallel_epsilon) {
        return std::nullopt;
    }
    const double t = (plane_z - origin.z) / direction.z;
    if (!std::isfinite(t)) {
        return std::nullopt;
    }
    return PlaneIntersection{t, add(origin, multiply(direction, t))};
}

PhaseTransform look_at_phase_transform(
    PhasePoint3 eye,
    PhasePoint3 target,
    PhasePoint3 up) noexcept {
    const PhasePoint3 forward = normalized(subtract(target, eye));
    const PhasePoint3 right = normalized(cross(forward, up));
    const PhasePoint3 corrected_up = normalized(cross(right, forward));
    return {right, corrected_up, forward, eye};
}

PhasePoint3 transform_point_to_world(
    const PhaseTransform &transform,
    PhasePoint3 point) noexcept {
    return transform_point_impl(transform, point);
}

PhasePoint3 transform_vector_to_world(
    const PhaseTransform &transform,
    PhasePoint3 vector) noexcept {
    return add(
        multiply(transform.axis_x, vector.x),
        add(
            multiply(transform.axis_y, vector.y),
            multiply(transform.axis_z, vector.z)));
}

PhasePoint3 transform_point_to_local(
    const PhaseTransform &transform,
    PhasePoint3 point) noexcept {
    const PhasePoint3 translated = subtract(point, transform.translation);
    return {
        dot(translated, transform.axis_x),
        dot(translated, transform.axis_y),
        dot(translated, transform.axis_z)};
}

PhasePoint3 transform_vector_to_local(
    const PhaseTransform &transform,
    PhasePoint3 vector) noexcept {
    return {
        dot(vector, transform.axis_x),
        dot(vector, transform.axis_y),
        dot(vector, transform.axis_z)};
}

PhaseCandidateSet inverse_candidates(
    const PhaseLayout &layout,
    const PhaseCoordinate &target,
    uint32_t radius,
    const PhaseBandwidth &bandwidth) noexcept {
    PhaseCandidateSet result{};
    if (!layout.valid() || !bandwidth.valid() || radius > 3u ||
        !finite(target.x) || !finite(target.u)) {
        return result;
    }

    const double projected_x =
        (target.x.x + layout.focal_width * 0.5) /
            layout.focal_lattice_width() - 0.5;
    const double projected_y =
        (layout.focal_height * 0.5 - target.x.y) /
            layout.focal_lattice_height() - 0.5;
    if (!std::isfinite(projected_x) || !std::isfinite(projected_y)) {
        return result;
    }
    const int64_t center_x = static_cast<int64_t>(std::llround(projected_x));
    const int64_t center_y = static_cast<int64_t>(std::llround(projected_y));
    const int64_t signed_radius = static_cast<int64_t>(radius);

    for (int64_t dy = -signed_radius; dy <= signed_radius; ++dy) {
        for (int64_t dx = -signed_radius; dx <= signed_radius; ++dx) {
            const int64_t encoded_x = center_x + dx;
            const int64_t encoded_y = center_y + dy;
            if (encoded_x < 0 || encoded_y < 0 ||
                encoded_x >= static_cast<int64_t>(layout.encoded_width()) ||
                encoded_y >= static_cast<int64_t>(layout.output_height)) {
                continue;
            }
            const auto candidate_phase = phi(
                layout,
                static_cast<uint32_t>(encoded_x),
                static_cast<uint32_t>(encoded_y));
            if (!candidate_phase || result.count >= result.capacity) {
                continue;
            }
            PhaseCandidate candidate{};
            candidate.phase = *candidate_phase;
            candidate.x_error_lattice =
                (candidate.phase.x.x - target.x.x) /
                layout.focal_lattice_width();
            candidate.y_error_lattice =
                (candidate.phase.x.y - target.x.y) /
                layout.focal_lattice_height();
            candidate.u_error_physical = candidate.phase.u.x - target.u.x;
            const double score = candidate_score(candidate, bandwidth);
            candidate.bandwidth_weight = std::exp(-0.5 * score);

            size_t insertion = result.count;
            while (insertion > 0u && candidate_less(
                    candidate, result.items[insertion - 1u], bandwidth)) {
                result.items[insertion] = result.items[insertion - 1u];
                --insertion;
            }
            result.items[insertion] = candidate;
            ++result.count;
        }
    }
    return result;
}

PhaseCandidateSet inverse_same_channel_candidates(
    const PhaseLayout &layout,
    const PhaseCoordinate &target,
    uint32_t radius,
    const PhaseBandwidth &bandwidth,
    uint32_t channel) noexcept {
    PhaseCandidateSet result{};
    if (!layout.valid() || !bandwidth.valid() || radius > 3u || channel > 2u ||
        !finite(target.x) || !finite(target.u)) {
        return result;
    }

    const double projected_encoded_x =
        (target.x.x + layout.focal_width * 0.5) /
            layout.focal_lattice_width() - 0.5;
    const double projected_y =
        (layout.focal_height * 0.5 - target.x.y) /
            layout.focal_lattice_height() - 0.5;
    const double projected_pixel_x =
        (projected_encoded_x - static_cast<double>(channel)) / 3.0;
    if (!std::isfinite(projected_pixel_x) || !std::isfinite(projected_y)) {
        return result;
    }

    const int64_t center_pixel_x =
        static_cast<int64_t>(std::llround(projected_pixel_x));
    const int64_t center_y = static_cast<int64_t>(std::llround(projected_y));
    const int64_t signed_radius = static_cast<int64_t>(radius);
    for (int64_t dy = -signed_radius; dy <= signed_radius; ++dy) {
        for (int64_t dx = -signed_radius; dx <= signed_radius; ++dx) {
            const int64_t pixel_x = center_pixel_x + dx;
            const int64_t encoded_y = center_y + dy;
            const int64_t encoded_x = pixel_x * 3 + channel;
            if (pixel_x < 0 || encoded_y < 0 ||
                encoded_x >= static_cast<int64_t>(layout.encoded_width()) ||
                encoded_y >= static_cast<int64_t>(layout.output_height)) {
                continue;
            }
            const auto candidate_phase = phi(
                layout,
                static_cast<uint32_t>(encoded_x),
                static_cast<uint32_t>(encoded_y));
            if (!candidate_phase || result.count >= result.capacity) {
                continue;
            }
            PhaseCandidate candidate{};
            candidate.phase = *candidate_phase;
            candidate.x_error_lattice =
                (candidate.phase.x.x - target.x.x) /
                layout.focal_lattice_width();
            candidate.y_error_lattice =
                (candidate.phase.x.y - target.x.y) /
                layout.focal_lattice_height();
            candidate.u_error_physical = candidate.phase.u.x - target.u.x;
            const double score = candidate_score(candidate, bandwidth);
            candidate.bandwidth_weight = std::exp(-0.5 * score);

            size_t insertion = result.count;
            while (insertion > 0u && candidate_less(
                    candidate, result.items[insertion - 1u], bandwidth)) {
                result.items[insertion] = result.items[insertion - 1u];
                --insertion;
            }
            result.items[insertion] = candidate;
            ++result.count;
        }
    }
    return result;
}

} // namespace lfs::lightfield
