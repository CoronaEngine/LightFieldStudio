#pragma once

#include "ssat/phase1/phase_mapping.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include "ssat/phase1/calibrated_lattice.h"

namespace lfs::lightfield {

[[nodiscard]] inline bool same_phase_layout(
    const PhaseLayout &lhs,
    const PhaseLayout &rhs) noexcept {
    const auto close = [](double a, double b) noexcept {
        return std::abs(a - b) <= 1.0e-9;
    };
    const auto close_point = [&](PhasePoint3 a,
                                 PhasePoint3 b) noexcept {
        return close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z);
    };
    return lhs.output_width == rhs.output_width &&
        lhs.output_height == rhs.output_height &&
        lhs.view_count == rhs.view_count &&
        close(lhs.pitch, rhs.pitch) && close(lhs.slant, rhs.slant) &&
        close(lhs.offset, rhs.offset) &&
        close(lhs.focal_distance, rhs.focal_distance) &&
        close(lhs.focal_width, rhs.focal_width) &&
        close(lhs.focal_height, rhs.focal_height) &&
        close(lhs.array_angle_degrees, rhs.array_angle_degrees) &&
        close_point(lhs.local_to_world.axis_x, rhs.local_to_world.axis_x) &&
        close_point(lhs.local_to_world.axis_y, rhs.local_to_world.axis_y) &&
        close_point(lhs.local_to_world.axis_z, rhs.local_to_world.axis_z) &&
        close_point(lhs.local_to_world.translation,
                    rhs.local_to_world.translation);
}

struct PhaseFrameState {
    uint64_t frame_index{};
    uint64_t layout_generation{};
    uint64_t layout_hash{};
    PhaseLayout current{};
    PhaseLayout previous{};
    uint64_t static_view_id_address{};
    uint64_t static_inverse_ray_length_address{};
    uint64_t static_view_x_address{};
    uint64_t static_focal_x_address{};
    uint64_t static_focal_y_address{};
    uint64_t static_ray_count{};
    uint64_t static_focal_x_count{};
    uint64_t static_focal_y_count{};
    uint32_t static_view_count{};
    bool previous_valid{};
    // Bound by Phase1, called only by spatial prepare on cold layout demand.
    std::function<std::shared_ptr<const CalibratedLattice>()> calibrated_lattice;


    [[nodiscard]] bool repeated_camera() const noexcept {
        return previous_valid && same_phase_layout(current, previous);
    }
    [[nodiscard]] bool valid() const noexcept;
};

class PhaseFrameStateConsumer {
public:
    virtual ~PhaseFrameStateConsumer() noexcept = default;

    [[nodiscard]] virtual bool set_phase_frame_state(
        const PhaseFrameState &state) noexcept = 0;
    virtual void invalidate_phase_frame_state() noexcept = 0;
};

} // namespace lfs::lightfield
