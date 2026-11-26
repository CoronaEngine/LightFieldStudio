#pragma once

#include "lightfield_framebuffer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace lfs::lightfield {

struct PhasePoint3 {
    double x{};
    double y{};
    double z{};
};

struct PhaseTransform {
    PhasePoint3 axis_x{1.0, 0.0, 0.0};
    PhasePoint3 axis_y{0.0, 1.0, 0.0};
    PhasePoint3 axis_z{0.0, 0.0, 1.0};
    PhasePoint3 translation{};

    [[nodiscard]] bool valid_rigid(double tolerance = 1.0e-6) const noexcept;
};

struct PhaseLayout {
    uint32_t output_width{};
    uint32_t output_height{};
    uint32_t view_count{};
    double pitch{};
    double slant{};
    double offset{};
    double focal_distance{};
    double focal_width{};
    double focal_height{};
    double array_angle_degrees{};
    PhaseTransform local_to_world{};

    [[nodiscard]] uint32_t encoded_width() const noexcept;
    [[nodiscard]] double focal_lattice_width() const noexcept;
    [[nodiscard]] double focal_lattice_height() const noexcept;
    [[nodiscard]] bool valid() const noexcept;
};

struct PhysicalSubpixel {
    uint32_t encoded_x{};
    uint32_t encoded_y{};
    uint32_t pixel_x{};
    uint32_t channel{};
    double focal_lattice_x{};
    double focal_lattice_y{};
};

struct PhaseCoordinate {
    PhysicalSubpixel subpixel{};
    uint32_t view_id{};
    PhasePoint3 x{};
    PhasePoint3 u{};
    PhasePoint3 world_origin{};
    PhasePoint3 world_direction{0.0, 0.0, 1.0};
};

struct PlaneIntersection {
    double t{};
    PhasePoint3 point{};
};

struct PhaseBandwidth {
    double sigma_x_lattice{1.0};
    double sigma_y_lattice{1.0};
    double sigma_u_physical{1.0};

    [[nodiscard]] bool valid() const noexcept;
};

struct PhaseCandidate {
    PhaseCoordinate phase{};
    double x_error_lattice{};
    double y_error_lattice{};
    double u_error_physical{};
    double bandwidth_weight{};
};

struct PhaseCandidateSet {
    static constexpr size_t capacity = 49u;

    std::array<PhaseCandidate, capacity> items{};
    size_t count{};

    [[nodiscard]] const PhaseCandidate *begin() const noexcept {
        return items.data();
    }
    [[nodiscard]] const PhaseCandidate *end() const noexcept {
        return items.data() + count;
    }
};

[[nodiscard]] PhaseLayout make_phase_layout(
    const LenticularParams &lenticular,
    const LightFieldGeometryParams &geometry,
    PhaseTransform local_to_world = {}) noexcept;

[[nodiscard]] std::optional<PhaseCoordinate> phi(
    const PhaseLayout &layout,
    uint32_t encoded_x,
    uint32_t encoded_y) noexcept;

[[nodiscard]] std::optional<PlaneIntersection> intersect_plane(
    PhasePoint3 origin,
    PhasePoint3 direction,
    double plane_z,
    double parallel_epsilon = 1.0e-9) noexcept;

[[nodiscard]] PhaseTransform look_at_phase_transform(
    PhasePoint3 eye,
    PhasePoint3 target,
    PhasePoint3 up) noexcept;

[[nodiscard]] PhasePoint3 transform_point_to_world(
    const PhaseTransform &transform,
    PhasePoint3 point) noexcept;

[[nodiscard]] PhasePoint3 transform_vector_to_world(
    const PhaseTransform &transform,
    PhasePoint3 vector) noexcept;

[[nodiscard]] PhasePoint3 transform_point_to_local(
    const PhaseTransform &transform,
    PhasePoint3 point) noexcept;

[[nodiscard]] PhasePoint3 transform_vector_to_local(
    const PhaseTransform &transform,
    PhasePoint3 vector) noexcept;

[[nodiscard]] PhaseCandidateSet inverse_candidates(
    const PhaseLayout &layout,
    const PhaseCoordinate &target,
    uint32_t radius,
    const PhaseBandwidth &bandwidth) noexcept;

[[nodiscard]] PhaseCandidateSet inverse_same_channel_candidates(
    const PhaseLayout &layout,
    const PhaseCoordinate &target,
    uint32_t radius,
    const PhaseBandwidth &bandwidth,
    uint32_t channel) noexcept;

} // namespace lfs::lightfield
