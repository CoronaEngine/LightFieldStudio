#include "ssat/phase1/phase_frame_state.h"

namespace lfs::lightfield {

bool PhaseFrameState::valid() const noexcept {
    if (layout_generation == 0u || layout_hash == 0u || !current.valid()) {
        return false;
    }
    if (current.output_width == 0u || current.output_height == 0u) {
        return false;
    }
    if (static_view_id_address == 0u ||
        static_inverse_ray_length_address == 0u ||
        static_view_x_address == 0u ||
        static_focal_x_address == 0u ||
        static_focal_y_address == 0u ||
        static_ray_count != static_cast<uint64_t>(
            current.output_width) * current.output_height * 3u ||
        static_focal_x_count != static_cast<uint64_t>(
            current.output_width) * 3u ||
        static_focal_y_count != static_cast<uint64_t>(
            current.output_height) ||
        static_view_count != current.view_count) {
        return false;
    }
    return !previous_valid ||
        (previous.valid() &&
         previous.output_width == current.output_width &&
         previous.output_height == current.output_height &&
         previous.view_count == current.view_count &&
         previous.pitch == current.pitch &&
         previous.slant == current.slant &&
         previous.offset == current.offset &&
         previous.focal_distance == current.focal_distance &&
         previous.focal_width == current.focal_width &&
         previous.focal_height == current.focal_height &&
         previous.array_angle_degrees == current.array_angle_degrees);
}

} // namespace lfs::lightfield
