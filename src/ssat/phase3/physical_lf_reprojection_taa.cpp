#include "ssat/phase3/physical_lf_reprojection_taa.h"
#include "ssat/phase3/native/early_measurement_runtime.h"

#include "ssat/phase3/history_lifecycle.h"
#include "ssat/phase3/material_normal_history.h"
#include "ssat/resident_storage.h"

#include "math/basic_types.h"
#include "math/transform.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace lfs::denoise {
using namespace ocarina;
namespace phase3_native = physical_lf::native;
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr uint32_t kSurfaceValidBit = 1u << 0u;
constexpr uint32_t kSurfaceClassShift = 2u;
constexpr uint32_t kSurfaceClassMask = 0x1fu;
constexpr uint32_t kSurfaceDiffuse = 1u << 0u;
constexpr uint32_t kSurfaceGlossy = 1u << 1u;
constexpr uint32_t kSurfaceNearSpecular = 1u << 2u;
constexpr uint32_t kSurfaceTransmission = 1u << 3u;
constexpr uint32_t kSurfaceEmissive = 1u << 4u;
constexpr uint32_t kResponsiveSurfaceMask =
    kSurfaceGlossy | kSurfaceNearSpecular |
    kSurfaceTransmission | kSurfaceEmissive;

struct SupportProfile {
    uint32_t radius{};
    uint32_t max_taps{};
    float sigma_x_lattice{};
    float sigma_u_views{};
    float position_tolerance_lattice{};
    float minimum_normal_similarity{};
    float maximum_albedo_relative_error{};
    float maximum_ray_angular_error{};
};

[[nodiscard]] SupportProfile support_profile(Lane lane) noexcept {
    return lane == Lane::Quality
        ? SupportProfile{3u, 4u, 6.0f, 1.0f, 1.5f, 0.90f, 0.25f, 0.05f}
        : SupportProfile{2u, 4u, 4.5f, 0.75f, 1.5f, 0.92f, 0.20f, 0.035f};
}

template<typename T>
[[nodiscard]] const BufferView<T> *required_view(
    const backend::FrameResources &resources,
    backend::Semantic semantic) noexcept {
    return resources.view<T>(semantic, backend::TemporalSlot::Current, 0u);
}

template<typename T>
[[nodiscard]] uint64_t native_address(const BufferView<T> &view) noexcept {
    return static_cast<uint64_t>(view.handle()) +
        static_cast<uint64_t>(view.offset_in_byte());
}

[[nodiscard]] float4x4 phase_transform_matrix(
    const lightfield::PhaseTransform &transform) noexcept {
    return make_float4x4(
        static_cast<float>(transform.axis_x.x),
        static_cast<float>(transform.axis_x.y),
        static_cast<float>(transform.axis_x.z), 0.0f,
        static_cast<float>(transform.axis_y.x),
        static_cast<float>(transform.axis_y.y),
        static_cast<float>(transform.axis_y.z), 0.0f,
        static_cast<float>(transform.axis_z.x),
        static_cast<float>(transform.axis_z.y),
        static_cast<float>(transform.axis_z.z), 0.0f,
        static_cast<float>(transform.translation.x),
        static_cast<float>(transform.translation.y),
        static_cast<float>(transform.translation.z), 1.0f);
}

[[nodiscard]] phase3_native::Matrix4x4 native_matrix(
    const float4x4 &matrix) noexcept {
    phase3_native::Matrix4x4 result{};
    for (size_t column = 0u; column < 4u; ++column) {
        result.columns[column] = {
            matrix[column][0u], matrix[column][1u],
            matrix[column][2u], matrix[column][3u]};
    }
    return result;
}


void populate_view_positions(phase3_native::Phase3Params &params) noexcept {
    const uint32_t view_count = std::max(params.view_count, 1u);
    const uint32_t inline_count = std::min(
        view_count, phase3_native::kInlinePhase3ViewCount);
    for (uint32_t view = 0u; view < inline_count; ++view) {
        const float unit = view_count > 1u
            ? static_cast<float>(view) / static_cast<float>(view_count - 1u)
            : 0.5f;
        const float angle =
            (unit - 0.5f) * params.array_angle_radians;
        params.view_positions[view] =
            params.focal_distance * std::tan(angle);
    }
    for (uint32_t view = 0u; view < inline_count; ++view) {
        const uint32_t neighbor = view + 1u < inline_count
            ? view + 1u : (view > 0u ? view - 1u : view);
        const float spacing = std::abs(
            params.view_positions[neighbor] - params.view_positions[view]);
        params.inverse_view_spacings[view] = spacing > 1.0e-6f
            ? 1.0f / spacing : 1.0e6f;
    }
}

[[nodiscard]] std::pair<std::vector<float>, std::vector<float>>
make_extended_view_luts(
    uint32_t view_count,
    float focal_distance,
    float array_angle_radians) {
    view_count = std::max(view_count, 1u);
    std::vector<float> positions(view_count);
    std::vector<float> inverse_spacings(view_count);
    for (uint32_t view = 0u; view < view_count; ++view) {
        const float unit = view_count > 1u
            ? static_cast<float>(view) / static_cast<float>(view_count - 1u)
            : 0.5f;
        positions[view] = focal_distance * std::tan(
            (unit - 0.5f) * array_angle_radians);
    }
    for (uint32_t view = 0u; view < view_count; ++view) {
        const uint32_t neighbor = view + 1u < view_count
            ? view + 1u : (view > 0u ? view - 1u : view);
        const float spacing = std::abs(
            positions[neighbor] - positions[view]);
        inverse_spacings[view] = spacing > 1.0e-6f
            ? 1.0f / spacing : 1.0e6f;
    }
    return {std::move(positions), std::move(inverse_spacings)};
}

struct ReducedBasisFactors {
    std::int32_t b0x{1};
    std::int32_t b0y{};
    std::int32_t b1x{};
    std::int32_t b1y{1};
    bool valid{};
};

struct AffineMetricValue {
    double constant{};
    double slope{};

    [[nodiscard]] double at(double parameter) const noexcept {
        return constant + slope * parameter;
    }
};

[[nodiscard]] AffineMetricValue metric_inner_coefficients(
    std::int32_t ax,
    std::int32_t ay,
    std::int32_t bx,
    std::int32_t by,
    double slant) noexcept {
    return {
        static_cast<double>(ax) * bx + static_cast<double>(ay) * by,
        (static_cast<double>(ax) + slant * ay) *
            (static_cast<double>(bx) + slant * by)};
}

[[nodiscard]] std::int32_t nearest_lattice_integer(double value) noexcept {
    const double lower = std::floor(value);
    return static_cast<std::int32_t>(
        value - lower > 0.5 ? lower + 1.0 : lower);
}

[[nodiscard]] ReducedBasisFactors prevalidated_rgb_basis(
    const phase3_native::Phase3Params &params,
    std::span<const float> extended_inverse_spacings) noexcept {
    ReducedBasisFactors result{};
    const double grid_slant = 3.0 * params.slant_tangent;
    if (!(params.inverse_sigma_x_lattice > 0.0f) ||
        !(params.inverse_sigma_u_views > 0.0f) ||
        !(params.focal_distance > 0.0f) || params.view_count == 0u) {
        return result;
    }
    double minimum_inverse_spacing =
        std::numeric_limits<double>::infinity();
    double maximum_inverse_spacing{};
    for (std::uint32_t view = 0u; view < params.view_count; ++view) {
        const double spacing = !extended_inverse_spacings.empty()
            ? extended_inverse_spacings[view]
            : params.inverse_view_spacings[view];
        if (!std::isfinite(spacing) || !(spacing > 0.0)) return result;
        minimum_inverse_spacing = std::min(minimum_inverse_spacing, spacing);
        maximum_inverse_spacing = std::max(maximum_inverse_spacing, spacing);
    }
    constexpr double phase_padding = 0.0002;
    const double angle0 = params.phase_angle_bias - phase_padding*params.phase_to_angle_scale;
    const double angle1 = params.phase_angle_bias +
        (static_cast<double>(params.pitch)+phase_padding) * params.phase_to_angle_scale;
    const double minimum_absolute_angle =
        ((angle0 <= 0.0 && angle1 >= 0.0) ||
         (angle1 <= 0.0 && angle0 >= 0.0))
        ? 0.0
        : std::min(std::abs(angle0), std::abs(angle1));
    const double maximum_absolute_angle =
        std::max(std::abs(angle0), std::abs(angle1));
    if (maximum_absolute_angle >= 0.5 * kPi - 1.0e-6) return result;
    const double tangent_minimum = std::tan(minimum_absolute_angle);
    const double tangent_maximum = std::tan(maximum_absolute_angle);
    const double derivative_scale = std::abs(
        static_cast<double>(params.focal_distance) *
        params.phase_to_angle_scale);
    const double derivative_minimum = derivative_scale *
        (1.0 + tangent_minimum * tangent_minimum);
    const double derivative_maximum = derivative_scale *
        (1.0 + tangent_maximum * tangent_maximum);
    const double sigma_ratio = static_cast<double>(
        params.inverse_sigma_u_views / params.inverse_sigma_x_lattice);
    double parameter_low = derivative_minimum *
        minimum_inverse_spacing * sigma_ratio;
    double parameter_high = derivative_maximum *
        maximum_inverse_spacing * sigma_ratio;
    parameter_low *= parameter_low;
    parameter_high *= parameter_high;
    if (!std::isfinite(parameter_low) || !std::isfinite(parameter_high) ||
        parameter_low < 0.0 || parameter_high < parameter_low) {
        return result;
    }

    {
        // For RGB the metric is j^2 I + vv^T. Bound det and trace over the
        // complete calibrated angle and spacing interval, including FP32 error.
        const double j2=double(params.inverse_sigma_x_lattice)*params.inverse_sigma_x_lattice;
        const double a=1.+grid_slant*grid_slant;
        const double determinant_lower=j2*j2*(1.+a*parameter_low);
        const double trace_upper=j2*(2.+a*parameter_high);
        const double bound=std::max(trace_upper*trace_upper,1.);
        if(!std::isfinite(bound) || bound>=std::numeric_limits<float>::max() ||
           determinant_lower<=(1.e-8+8.*std::numeric_limits<float>::epsilon())*bound)return {};
    }
    constexpr double kDecisionMargin = 5.0e-4;
    for (int iteration = 0; iteration < 4; ++iteration) {
        AffineMetricValue norm0 = metric_inner_coefficients(
            result.b0x, result.b0y, result.b0x, result.b0y,
            grid_slant);
        AffineMetricValue norm1 = metric_inner_coefficients(
            result.b1x, result.b1y, result.b1x, result.b1y,
            grid_slant);
        const double difference_low =
            norm1.at(parameter_low) - norm0.at(parameter_low);
        const double difference_high =
            norm1.at(parameter_high) - norm0.at(parameter_high);
        const double norm_margin = 5.e-5 * std::max({1.0, norm0.at(parameter_low),
            norm1.at(parameter_low), norm0.at(parameter_high), norm1.at(parameter_high)});
        if (!std::isfinite(difference_low) ||
            !std::isfinite(difference_high) ||
            std::abs(difference_low) <= norm_margin ||
            std::abs(difference_high) <= norm_margin ||
            std::signbit(difference_low) != std::signbit(difference_high)) {
            return {};
        }
        if (difference_low < 0.0) {
            std::swap(result.b0x, result.b1x);
            std::swap(result.b0y, result.b1y);
            norm0 = metric_inner_coefficients(
                result.b0x, result.b0y, result.b0x, result.b0y,
                grid_slant);
        }
        const AffineMetricValue numerator = metric_inner_coefficients(
            result.b0x, result.b0y, result.b1x, result.b1y,
            grid_slant);
        const double denominator_low = norm0.at(parameter_low);
        const double denominator_high = norm0.at(parameter_high);
        if (!(denominator_low > 1.0e-12) ||
            !(denominator_high > 1.0e-12)) {
            return {};
        }
        const double projection_low =
            numerator.at(parameter_low) / denominator_low;
        const double projection_high =
            numerator.at(parameter_high) / denominator_high;
        if (!std::isfinite(projection_low) ||
            !std::isfinite(projection_high) ||
            std::abs(projection_low) > 4096.0 ||
            std::abs(projection_high) > 4096.0) {
            return {};
        }
        const std::int32_t multiple_low =
            nearest_lattice_integer(projection_low);
        const std::int32_t multiple_high =
            nearest_lattice_integer(projection_high);
        const double projection_minimum =
            std::min(projection_low, projection_high);
        const double projection_maximum =
            std::max(projection_low, projection_high);
        if (multiple_low != multiple_high ||
            projection_minimum <=
                static_cast<double>(multiple_low) - 0.5 + kDecisionMargin ||
            projection_maximum >=
                static_cast<double>(multiple_low) + 0.5 - kDecisionMargin) {
            return {};
        }
        const std::int64_t next_x = static_cast<std::int64_t>(result.b1x) -
            static_cast<std::int64_t>(multiple_low) * result.b0x;
        const std::int64_t next_y = static_cast<std::int64_t>(result.b1y) -
            static_cast<std::int64_t>(multiple_low) * result.b0y;
        if (next_x < std::numeric_limits<std::int32_t>::min() ||
            next_x > std::numeric_limits<std::int32_t>::max() ||
            next_y < std::numeric_limits<std::int32_t>::min() ||
            next_y > std::numeric_limits<std::int32_t>::max()) {
            return {};
        }
        result.b1x = static_cast<std::int32_t>(next_x);
        result.b1y = static_cast<std::int32_t>(next_y);
    }
    const std::int64_t determinant =
        static_cast<std::int64_t>(result.b0x) * result.b1y -
        static_cast<std::int64_t>(result.b0y) * result.b1x;
    {
        const double w=3.*params.output_width,h=params.output_height;
        const double a=std::abs(double(result.b0x)),b=std::abs(double(result.b1x));
        const double c=std::abs(double(result.b0y)),d=std::abs(double(result.b1y));
        const double u=d*w+b*h+4.,v=c*w+a*h+4.;
        // The four certified addresses use a common integer origin. Above
        // the bound the ordinary wide-integer projection remains available.
        if(std::max(a*u+b*v+a+b,c*u+d*v+c+d)>=double(1u<<30u))return {};
    }
    result.valid = determinant == 1 || determinant == -1;
    return result;
}

[[nodiscard]] backend::Error missing(std::string message) {
    return {backend::ErrorCode::ResourceMissing, std::move(message)};
}

[[nodiscard]] backend::Result<backend::ResourceRequirementSet>
physical_requirements() {
    backend::ResourceRequirementSet requirements;
    const std::pair<backend::Semantic, backend::ValueType> current[] = {
        {backend::Semantic::Radiance, backend::ValueType::Float4},
        {backend::Semantic::PackedSurfaceObservation,
         backend::ValueType::PackedSurfaceObservation32},
    };
    for (const auto &[semantic, type] : current) {
        auto added = requirements.add({
            semantic, backend::TemporalSlot::Current, 0u, type, false});
        if (!added.ok())
            return backend::Result<backend::ResourceRequirementSet>::failure(added.error());
    }
    requirements.require_rigid_instance_motion();
    // RGB history is queried in physical LF ray space. No sampled secondary
    // path is required to produce that four-dimensional correspondence.
    return backend::Result<backend::ResourceRequirementSet>::success(std::move(requirements));
}

} // namespace

struct PhysicalLfReprojectionTaa::Impl {
    PhysicalLfTaaConfig config{};
    PhysicalLfTaaTelemetry telemetry{};
    Device *device{};
    backend::Extent2D sample_extent{};
    backend::Extent2D output_extent{};
    uint64_t layout_generation{};
    uint64_t layout_hash{};
    lightfield::PhaseFrameState phase_state{};
    bool motion_committed{}, motion_current{};
    std::uint64_t previous_measurement_geometry{},current_regular_measurement_geometry{};
    std::uint64_t current_regular_material_normals{};
    std::array<Buffer<uint32_t>,2u> panel_samples{};
    std::uint64_t current_panel_samples{};

    HistoryLifecycle history_lifecycle{};
    MaterialNormalHistory material_normals{};
    bool phase_state_bound{};
    std::optional<ReducedBasisFactors> rgb_basis;

    std::array<Buffer<phase3_native::TemporalSample>, 2u> history{};
    std::array<Buffer<uint4>,2u> history_tail{};
    bool split_history{}, single_tail{}, native_trace_history{};
    std::optional<mr_native::Params> pending_history_commit;
    Buffer<uint4> committed_geometry{};
    Buffer<float> extended_view_positions{};
    Buffer<float> extended_inverse_view_spacings{};
    std::vector<float> view_positions_host{};
    std::vector<float> inverse_view_spacings_host{};
    Buffer<uint32_t> frame_status{};
    BufferView<phase3_native::TemporalSample> presentation_source{};
    uint32_t presentation_history_index{
        std::numeric_limits<uint32_t>::max()};

    mr_native::Runtime measurement_runtime{};
    std::array<Buffer<half4>, 2> measurement{};

    ResolvedCurrentView resolved_current{};
    // Phase2 lends its early ES observation to the RGB history update.

    explicit Impl(PhysicalLfTaaConfig input) : config(std::move(input)) {
        const SupportProfile profile = support_profile(config.lane);
        telemetry.support_radius = profile.radius;
        telemetry.support_max_taps = profile.max_taps;
    }

    void invalidate_history() noexcept {
        current_regular_material_normals=0u;
        current_panel_samples=0u;
        material_normals.reset();
        pending_history_commit.reset();
        motion_committed=motion_current=false;
        history_lifecycle.reset();

        phase_state_bound = false;
        telemetry.phase_state_valid = false;
        telemetry.history_valid_before = false;
        telemetry.temporal_used = false;
        presentation_source = {};

        presentation_history_index = std::numeric_limits<uint32_t>::max();
        resolved_current = {};

        measurement_runtime.clear();
    }

    [[nodiscard]] backend::Error allocate(
        const backend::FrameContext &context) noexcept {
        if (context.device == nullptr || context.stream == nullptr ||
            context.sample_domain != backend::SampleDomain::EncodedSubpixelGrid ||
            !context.sample_extent.valid() || !context.output_extent.valid() ||
            context.sample_extent.width != context.output_extent.width * 3u ||
            context.sample_extent.height != context.output_extent.height ||
            config.lenticular.width != context.output_extent.width ||
            config.lenticular.height != context.output_extent.height ||
            config.lenticular.view_count == 0u ||
            config.lenticular.view_count > phase3_native::kMaximumPhase3ViewCount ||
            config.lenticular.pitch <= 0.0 ||
            config.geometry.focal_distance <= 0.0) {
            return {backend::ErrorCode::ResourceMismatch,
                    "Physical LF TAA requires a calibrated 3W by H encoded layout."};
        }
        const bool changed = device != context.device ||
            sample_extent != context.sample_extent ||
            output_extent != context.output_extent ||
            layout_generation != context.layout_generation ||
            layout_hash != context.layout_hash;
        if (!changed && telemetry.prepared) return {};

        panel_samples={};
        device = context.device;
        sample_extent = context.sample_extent;
        output_extent = context.output_extent;
        layout_generation = context.layout_generation;
        layout_hash = context.layout_hash;
        rgb_basis.reset();
        invalidate_history();
        try {
            const size_t count = static_cast<size_t>(sample_extent.area());
            split_history=resident::dedicated(device);
            single_tail=split_history;
            native_trace_history=single_tail && resident::native_packed_owner(device);
            if(native_trace_history)committed_geometry=device->create_buffer<uint4>(1,"lfs.history.first_surface_head");
            else if(split_history && !single_tail)committed_geometry=device->create_buffer<uint4>(count,"lfs.history.committed_surface");
            else committed_geometry={};
            for (size_t index = 0u; index < history.size(); ++index) {
                if(native_trace_history){
                    history[index]={};
                    if(index==0u)history_tail[index]=device->create_buffer<uint4>(count,"lfs.history.last_surface_tail");
                    else history_tail[index]={};
                    continue;
                }
                if(single_tail) {
                    history_tail[index]={};
                    if(index==0u)history[index]=device->create_buffer<phase3_native::TemporalSample>(count,"lfs.history.last_complete_surface");
                    else history[index]={};
                    continue;
                }
                if(split_history) {
                    history[index]={};
                    history_tail[index]=device->create_buffer<uint4>(count,"lfs.history.surface_tail."+std::to_string(index));
                    continue;
                }
                history_tail[index]={};
                history[index] =
                    device->create_buffer<phase3_native::TemporalSample>(
                        count, "lfs.denoise.physical_lf_taa.history." +
                                   std::to_string(index));
            }
            for (size_t i=0; i<2; ++i){
                if(native_trace_history && i==1u)measurement[i]={};
                else measurement[i]=device->create_buffer<half4>(count,"lfs.early_measurement."+std::to_string(i));
            }

            frame_status = device->create_buffer<uint32_t>(
                1u, "lfs.denoise.physical_lf_taa.frame_status");

            extended_view_positions = {};
            extended_inverse_view_spacings = {};
            std::tie(view_positions_host, inverse_view_spacings_host) =
                make_extended_view_luts(
                    config.lenticular.view_count,
                    static_cast<float>(config.geometry.focal_distance),
                    static_cast<float>(
                        config.geometry.array_angle_degrees * kPi / 180.0));
            if (config.lenticular.view_count >
                phase3_native::kInlinePhase3ViewCount) {
                extended_view_positions = device->create_buffer<float>(
                    view_positions_host.size(),
                    "lfs.denoise.physical_lf_taa.extended_view_positions");
                extended_inverse_view_spacings = device->create_buffer<float>(
                    inverse_view_spacings_host.size(),
                    "lfs.denoise.physical_lf_taa.extended_inverse_view_spacings");
                *context.stream
                    << extended_view_positions.upload(
                           view_positions_host.data())
                    << extended_inverse_view_spacings.upload(
                           inverse_view_spacings_host.data())
                    << synchronize() << commit();
            }

        } catch (const std::exception &exception) {
            return {backend::ErrorCode::OutOfMemory, exception.what()};
        } catch (...) {
            return {backend::ErrorCode::OutOfMemory,
                    "Physical LF TAA GPU allocation failed."};
        }
        {
            if (auto error=measurement_runtime.initialize(*device,config.lane==Lane::Quality)) return error;
        }

        telemetry.sample_extent = sample_extent;
        telemetry.output_extent = output_extent;
        telemetry.history_bytes = static_cast<uint64_t>(sample_extent.area()) *
            (native_trace_history?sizeof(uint4):(single_tail?sizeof(phase3_native::TemporalSample):history.size()*(split_history?sizeof(uint4):sizeof(phase3_native::TemporalSample)))) +
            sample_extent.area()*(native_trace_history?1u:2u)*sizeof(half4) + committed_geometry.size()*sizeof(uint4);

        telemetry.canonical_bytes = 0u;
        ++telemetry.allocation_generation;
        telemetry.allocation_count = (single_tail?4u:(split_history?6u:5u)) + (extended_view_positions.size()!=0u?2u:0u);
        if(!split_history && (sample_extent.width==4608u || sample_extent.width==11520u)) {
            telemetry.history_bytes+=2ull*sample_extent.area()*sizeof(uint32_t);
            telemetry.allocation_count+=2u;
        }
        telemetry.prepared = true;

        telemetry.output_ready = false;

        return {};
    }
};

PhysicalLfReprojectionTaa::PhysicalLfReprojectionTaa(
    PhysicalLfTaaConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
    module_telemetry_.name = "PhysicalLfTemporalCommonCore";

}
PhysicalLfReprojectionTaa::PhysicalLfReprojectionTaa(
    PhysicalLfReprojectionTaa &&) noexcept = default;
PhysicalLfReprojectionTaa &PhysicalLfReprojectionTaa::operator=(
    PhysicalLfReprojectionTaa &&) noexcept = default;
PhysicalLfReprojectionTaa::~PhysicalLfReprojectionTaa() noexcept = default;

std::uint64_t PhysicalLfReprojectionTaa::previous_measurement_geometry() const noexcept {return impl_->previous_measurement_geometry;}
std::uint64_t PhysicalLfReprojectionTaa::current_regular_measurement_geometry() const noexcept {return impl_->current_regular_measurement_geometry;}
std::uint64_t PhysicalLfReprojectionTaa::current_regular_material_normals() const noexcept {return impl_->current_regular_material_normals;}
std::uint64_t PhysicalLfReprojectionTaa::current_panel_samples() const noexcept {return impl_->current_panel_samples;}

TemporalMethod PhysicalLfReprojectionTaa::method() const noexcept {
    return TemporalMethod::PhysicalLfReprojection;
}
SignalDomain PhysicalLfReprojectionTaa::input_domain() const noexcept {
    return SignalDomain::PhysicalRgbVariance;
}
SignalDomain PhysicalLfReprojectionTaa::output_domain() const noexcept {

    return SignalDomain::EncodedCarrierScalar;
}
const ModuleTelemetry &PhysicalLfReprojectionTaa::module_telemetry() const noexcept {
    return module_telemetry_;
}
const PhysicalLfTaaTelemetry &PhysicalLfReprojectionTaa::telemetry() const noexcept {
    return impl_->telemetry;
}

bool PhysicalLfReprojectionTaa::set_resolved_current_view(const ResolvedCurrentView &view) noexcept {
    if (!view.valid() || view.domain != SignalDomain::DemodulatedRgbVariance ||
        view.encoding != NativeCurrentEncoding::LinearHalf4 || view.packed_signal_stride_bytes != sizeof(ocarina::half4)) {
        impl_->resolved_current = {}; return false;
    }
    if(impl_->config.lane==Lane::Quality &&
        (view.frame_status_encoding!=FrameStatusEncoding::StatusAndReconstructionCoverageUInt32 ||
         view.extent.width%32u!=0u || view.frame_status.size()!=size_t(1u+reconstruction_coverage_words(view.extent.width,view.extent.height))*sizeof(uint32_t))){
        impl_->resolved_current={};return false;
    }
    impl_->resolved_current = view; return true;
}

void PhysicalLfReprojectionTaa::invalidate_resolved_current_view() noexcept {
    impl_->resolved_current = {};

}

bool PhysicalLfReprojectionTaa::set_phase_frame_state(
    const lightfield::PhaseFrameState &state) noexcept {
    if (!state.valid() || !impl_->telemetry.prepared ||
        state.layout_generation != impl_->layout_generation ||
        state.layout_hash != impl_->layout_hash ||
        state.current.output_width != impl_->output_extent.width ||
        state.current.output_height != impl_->output_extent.height ||
        state.previous_valid != impl_->history_lifecycle.valid()) {
        impl_->phase_state_bound = false;
        impl_->telemetry.phase_state_valid = false;
        return false;
    }
    impl_->motion_current=state.previous_valid && (impl_->motion_committed || !state.repeated_camera());
    impl_->phase_state = state;
    impl_->phase_state_bound = true;
    impl_->telemetry.phase_state_valid = true;
    return true;
}

void PhysicalLfReprojectionTaa::invalidate_phase_frame_state() noexcept {
    impl_->phase_state_bound = false;
    impl_->telemetry.phase_state_valid = false;
    impl_->resolved_current = {};

}

backend::Error PhysicalLfReprojectionTaa::reset(
    backend::ResetReason reason) noexcept {
    if (reason == backend::ResetReason::None) {
        return {backend::ErrorCode::InvalidArgument,
                "Physical LF TAA reset reason must not be None."};
    }
    impl_->invalidate_history();

    ++impl_->telemetry.reset_count;
    impl_->telemetry.output_ready = false;
    return {};
}

backend::Error PhysicalLfReprojectionTaa::prepare(
    const backend::FrameContext &context) noexcept {
    const backend::Error error = impl_->allocate(context);
    if (!error) {
        module_telemetry_.owned_device_bytes =
            impl_->telemetry.history_bytes + impl_->telemetry.canonical_bytes + sizeof(std::uint32_t) +
            static_cast<uint64_t>(impl_->extended_view_positions.size() +
                                  impl_->extended_inverse_view_spacings.size()) *
                sizeof(float);
        module_telemetry_.allocation_count =
            impl_->telemetry.allocation_count;
    }
    return error;
}

backend::Error PhysicalLfReprojectionTaa::enqueue_commit(
    const backend::FrameContext &context,
    CommandBatch &commands) noexcept {
    if(!impl_->pending_history_commit)return {};
    if(context.device!=impl_->device || context.layout_generation!=impl_->layout_generation)
        return {backend::ErrorCode::ResourceMismatch,"History commit belongs to a different layout."};
    auto error=impl_->measurement_runtime.commit_history(*impl_->pending_history_commit,commands);
    impl_->pending_history_commit.reset();
    return error;
}

backend::Error PhysicalLfReprojectionTaa::validate_submission(
    const backend::FrameContext &) noexcept {

    return impl_->measurement_runtime.validate();
}

void PhysicalLfReprojectionTaa::finalize_success(
    const backend::FrameContext &) noexcept {
    impl_->material_normals.commit();
    impl_->motion_committed=impl_->motion_current;
    impl_->history_lifecycle.commit_success();


    impl_->measurement_runtime.clear();
}

void PhysicalLfReprojectionTaa::abort_submission(
    const backend::FrameContext &) noexcept {
    impl_->current_regular_material_normals=impl_->current_panel_samples=0u;
    impl_->material_normals.abort();
    if(impl_->split_history){
        impl_->invalidate_history();
    } else {
        impl_->history_lifecycle.cancel_pending();
    }

    impl_->measurement_runtime.clear();
    impl_->telemetry.output_ready = false;
}

backend::Result<backend::ResourceRequirementSet>
PhysicalLfReprojectionTaa::resource_requirements(
    const backend::FrameContext &) noexcept {
    return physical_requirements();
}

backend::Result<backend::ProcessOutput> PhysicalLfReprojectionTaa::process(
    const backend::FrameContext &context,
    const backend::FrameResources &resources,
    CommandBatch &commands) noexcept {
        impl_->current_regular_material_normals=impl_->current_panel_samples=0u;
        impl_->history_lifecycle.begin_frame();
    impl_->pending_history_commit.reset();

    impl_->measurement_runtime.clear();
    const bool use_history=impl_->history_lifecycle.valid();
    const ResolvedCurrentView &temporal_current = impl_->resolved_current;
    const bool use_compact_phase2 = temporal_current.valid();

    backend::Result<backend::ResourceRequirementSet> required =
        physical_requirements();
    if (!required.ok()) {
        return backend::Result<backend::ProcessOutput>::failure(required.error());
    }
    backend::Result<void> valid = resources.validate(required.value());
    if (!valid.ok()) {
        return backend::Result<backend::ProcessOutput>::failure(valid.error());
    }
    if (!impl_->telemetry.prepared || context.device != impl_->device ||
        context.sample_extent != impl_->sample_extent ||
        context.output_extent != impl_->output_extent ||
        context.sample_domain != backend::SampleDomain::EncodedSubpixelGrid ||
        context.layout_generation != impl_->layout_generation ||
        context.layout_hash != impl_->layout_hash ||
        !impl_->phase_state_bound || !impl_->phase_state.valid() ||
        impl_->phase_state.frame_index != context.frame_index ||
        impl_->phase_state.previous_valid != use_history) {
        return backend::Result<backend::ProcessOutput>::failure({
            backend::ErrorCode::InvalidState,
            "Physical LF TAA frame state does not match the current layout/history."});
    }
    if (use_compact_phase2 &&
        (impl_->resolved_current.extent != context.sample_extent ||
         impl_->resolved_current.frame_index != context.frame_index ||
         impl_->resolved_current.layout_generation != context.layout_generation ||
         impl_->resolved_current.layout_hash != context.layout_hash)) {
        return backend::Result<backend::ProcessOutput>::failure({
            backend::ErrorCode::ResourceMismatch,
            "Physical LF TAA native current does not match this frame."});
    }

    const auto *radiance = required_view<float4>(resources, backend::Semantic::Radiance);
    const auto *packed_observation = required_view<backend::PackedSurfaceObservationV1>(
        resources, backend::Semantic::PackedSurfaceObservation);
    const backend::RigidInstanceMotionTable *rigid = resources.rigid_instance_motion();
    const bool guide_missing = !packed_observation ||
        packed_observation->size() != context.sample_extent.area();
    if (!radiance || guide_missing || !rigid ||
        (use_history &&
         !impl_->history_lifecycle.source_generation_valid(
             rigid->history_valid, rigid->history_generation))) {
        return backend::Result<backend::ProcessOutput>::failure(
            missing("Physical LF TAA current guides or rigid motion history are incomplete."));
    }

    commands << impl_->frame_status.reset();
    const uint32_t read_index = impl_->history_lifecycle.read_index();
    const uint32_t write_index = impl_->history_lifecycle.write_index();
    const SupportProfile profile = support_profile(impl_->config.lane);

    phase3_native::Phase3Params params{};

    params.current_phase2 = use_compact_phase2
        ? temporal_current.packed_signal_address : 0u;

    params.current_packed_observation = packed_observation
        ? native_address(*packed_observation) : 0u;


    params.current_denoiser_path_guide = 0u;



    params.response_direction_glossy_only =
        impl_->config.response_direction_glossy_only ? 1u : 0u;

    resident::TraceWorkspaces native_workspace{};
    if(impl_->native_trace_history){
        native_workspace=resident::trace_workspaces(context.device,context.sample_extent.area(),params.current_packed_observation);
        if(!native_workspace)return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::ResourceMismatch,"The original native packed surface and post-trace resources do not match."});
        params.native_first_history_head=native_address(impl_->committed_geometry.view());
        params.history_geometry_write=native_workspace.first_hit;
        params.history_geometry_read=use_history?native_workspace.second_hit:0u;
        params.history_write=native_address(impl_->history_tail[0].view());
        params.history_read=use_history?params.history_write:0u;
    } else if(impl_->single_tail) {
        params.history_write=native_address(impl_->history[0].view());
        params.history_read=use_history?params.history_write:0u;
    } else if(impl_->split_history) {
        params.history_geometry_write=native_address(impl_->committed_geometry.view());
        params.history_geometry_read=use_history?params.history_geometry_write:0;
        params.history_read=use_history?native_address(impl_->history_tail[read_index].view()):0;
        params.history_write=native_address(impl_->history_tail[write_index].view());
    } else {
        params.history_read=use_history?native_address(impl_->history[read_index].view()):0;
        params.history_write=native_address(impl_->history[write_index].view());
    }
    impl_->previous_measurement_geometry=!impl_->native_trace_history && (!impl_->split_history || impl_->single_tail)?params.history_read:0u;
    impl_->current_regular_measurement_geometry=(!impl_->split_history && !impl_->single_tail)?params.history_write:0u;
    params.rigid_instance_motion = native_address(rigid->records);

    params.frame_status = native_address(impl_->frame_status.view());
    const lightfield::PhaseLayout &current_layout = impl_->phase_state.current;
    const lightfield::PhaseLayout &previous_layout = use_history
        ? impl_->phase_state.previous : impl_->phase_state.current;
    params.current_local_to_world = native_matrix(
        phase_transform_matrix(current_layout.local_to_world));
    params.previous_world_to_local = native_matrix(inverse(
        phase_transform_matrix(previous_layout.local_to_world)));
    params.output_width = impl_->output_extent.width;
    params.output_height = impl_->output_extent.height;
    params.view_count = std::max(impl_->config.lenticular.view_count, 1u);
    params.extended_view_positions =
        impl_->extended_view_positions.size() != 0u
        ? native_address(impl_->extended_view_positions.view()) : 0u;
    params.extended_inverse_view_spacings =
        impl_->extended_inverse_view_spacings.size() != 0u
        ? native_address(impl_->extended_inverse_view_spacings.view()) : 0u;
    params.extended_view_count = static_cast<uint32_t>(
        impl_->extended_view_positions.size());
    params.rigid_instance_count = static_cast<uint32_t>(rigid->records.size());
    params.rigid_instance_stride_bytes = rigid->record_stride_bytes;
    params.history_valid = use_history ? 1u : 0u;

    params.static_layout = use_history && same_phase_layout(
        current_layout, previous_layout) ? 1u : 0u;

    params.pitch = static_cast<float>(impl_->config.lenticular.pitch);
    params.inverse_pitch = 1.0f / params.pitch;
    params.slant_tangent = std::tan(
        static_cast<float>(impl_->config.lenticular.slant));
    params.offset = static_cast<float>(impl_->config.lenticular.offset);
    params.focal_distance = static_cast<float>(
        impl_->config.geometry.focal_distance);
    params.focal_width = static_cast<float>(impl_->config.geometry.focal_width);
    params.focal_height = static_cast<float>(impl_->config.geometry.focal_height);
    params.array_angle_radians = static_cast<float>(
        impl_->config.geometry.array_angle_degrees * kPi / 180.0);
    params.focal_lattice_width = params.focal_width /
        static_cast<float>(impl_->sample_extent.width);
    params.focal_lattice_height = params.focal_height /
        static_cast<float>(impl_->sample_extent.height);
    params.sigma_x_lattice = profile.sigma_x_lattice;
    params.sigma_u_views = profile.sigma_u_views;
    params.position_tolerance_lattice = profile.position_tolerance_lattice;
    params.minimum_normal_similarity = profile.minimum_normal_similarity;
    params.maximum_albedo_relative_error = profile.maximum_albedo_relative_error;
    params.maximum_ray_angular_error = profile.maximum_ray_angular_error;
    params.inverse_focal_distance = 1.0f / params.focal_distance;
    params.inverse_focal_lattice_width = 1.0f / params.focal_lattice_width;
    params.inverse_focal_lattice_height = 1.0f / params.focal_lattice_height;
    params.inverse_sigma_x_lattice = 1.0f / params.sigma_x_lattice;
    params.inverse_sigma_u_views = 1.0f / params.sigma_u_views;
    params.phase_view_scale = static_cast<float>(params.view_count) *
        params.inverse_pitch;
    params.pitch_per_view = params.pitch / static_cast<float>(params.view_count);
    params.inverse_array_angle_radians =
        std::abs(params.array_angle_radians) > 1.0e-7f
        ? 1.0f / params.array_angle_radians : 0.0f;
    params.inverse_maximum_ray_angular_error =
        1.0f / params.maximum_ray_angular_error;

    params.inverse_normal_error_range =
        1.0f / (1.0f - params.minimum_normal_similarity);
    if (params.view_count > 1u) {
        const float inverse_view_span =
            1.0f / static_cast<float>(params.view_count - 1u);
        params.phase_to_angle_scale = -params.array_angle_radians *
            static_cast<float>(params.view_count) * params.inverse_pitch *
            inverse_view_span;
        params.phase_angle_bias = 0.5f * params.array_angle_radians *
            static_cast<float>(params.view_count) * inverse_view_span;
    }
    params.base_position_tolerance = std::max(
        params.focal_lattice_width, params.focal_lattice_height) *
        params.position_tolerance_lattice;
    populate_view_positions(params);

    {
        // M consumes and overwrites only its own early ES color. Query geometry
        // is read from packed observations; previous histories are separate.

        params.frame_status = impl_->resolved_current.frame_status_address;
        if (params.frame_status == 0u)
            return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::ResourceMismatch,
                "Early measurement requires LinearHalf4 and shared ES error status."});
        mr_native::Params m{};m.geometry=params;
        const bool regular_layout = !impl_->split_history;
        if (regular_layout) {
            // A shared basis accelerates regular RGB queries. Wide-angle panels
            // can require a different basis at each projected view.
            if (!impl_->rgb_basis) {
                const auto extended = params.view_count > phase3_native::kInlinePhase3ViewCount
                    ? std::span<const float>{impl_->inverse_view_spacings_host}
                    : std::span<const float>{};
                impl_->rgb_basis = prevalidated_rgb_basis(params, extended);
            }
            const auto &rgb = *impl_->rgb_basis;
            m.rgb_b0x = rgb.b0x;
            m.rgb_b0y = rgb.b0y;
            m.rgb_b1x = rgb.b1x;
            m.rgb_b1y = rgb.b1y;
            m.rgb_basis_valid = rgb.valid;
            m.regular_r2_coverage = impl_->resolved_current.regular_r2_coverage;
            m.deterministic_motion_environment = impl_->motion_current && context.deterministic_environment;
        }
        if(impl_->single_tail){
            const std::uint64_t count=context.sample_extent.area();
            const auto bank=impl_->native_trace_history?native_workspace.canonical:resident::geometry_bank(context.device,1u,count);
            const std::uint64_t capacity=count*sizeof(uint4);
            const std::uint64_t canonical=context.output_extent.area()*sizeof(float4);
            const std::uint64_t metadata=count*sizeof(uint2);
            if(!bank || canonical>capacity || metadata>capacity-canonical ||
               bank>std::numeric_limits<std::uint64_t>::max()-canonical || (bank+canonical)%alignof(uint2))
                return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::ResourceMismatch,"Transient history would overlap presented pixels or exceed primary-hit storage."});
            m.history_metadata_staging=bank+canonical;
            if (temporal_current.held_angular_residuals) {
                const auto expected = m.history_metadata_staging + metadata;
                if (impl_->config.lane != Lane::Quality || temporal_current.held_angular_residuals != expected ||
                    count * sizeof(std::uint16_t) > capacity - canonical - metadata) {
                    return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::ResourceMismatch,
                        "The repeated angular estimate must follow current metadata in the exclusive hit bank."});
                }
                m.held_angular_residuals = expected;
            }
            if (temporal_current.current_angular_residuals) {
                if (!impl_->native_trace_history || temporal_current.current_angular_residuals!=m.history_metadata_staging+metadata)
                    return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::ResourceMismatch,"The angular result does not belong to the native frame workspace."});
                m.current_angular_residuals=temporal_current.current_angular_residuals;
            }
        }
        if (!impl_->split_history && !impl_->motion_current && impl_->phase_state.repeated_camera()) {
            if (!radiance || radiance->size() != context.sample_extent.area())
                return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::ResourceMismatch,"Stationary reconstruction requires full raw radiance."});
            m.stationary_radiance = native_address(*radiance);
        }
        if(impl_->native_trace_history){
            m.committed_measurement=native_address(impl_->measurement[0].view());
            m.measurement_read=use_history?m.committed_measurement:0u;
            m.measurement_write=native_workspace.first_hit;
            m.native_second_hit=native_workspace.second_hit;
        } else {
            m.measurement_read=use_history?native_address(impl_->measurement[read_index].view()):0u;
            m.measurement_write=native_address(impl_->measurement[write_index].view());
        }

        const bool moving_panel=regular_layout && impl_->motion_current &&
            (context.sample_extent.width==4608u || context.sample_extent.width==11520u);
        if(moving_panel) {
            const size_t count=context.sample_extent.area();
            if(impl_->panel_samples[0].size()!=count || impl_->panel_samples[1].size()!=count) {
                try {
                    std::array<Buffer<uint32_t>,2u> fresh;
                    fresh[0]=context.device->create_buffer<uint32_t>(count,"lfs.panel_exposure.a");
                    fresh[1]=context.device->create_buffer<uint32_t>(count,"lfs.panel_exposure.b");
                    impl_->panel_samples=std::move(fresh);
                    impl_->telemetry.history_bytes+=count*2u*sizeof(uint32_t);
                    impl_->telemetry.allocation_count+=2u;
                } catch(const std::exception &e) {
                    return backend::Result<backend::ProcessOutput>::failure({backend::ErrorCode::OutOfMemory,e.what()});
                }
            }
            m.current_panel_samples=native_address(impl_->panel_samples[write_index].view());
            m.previous_panel_samples=use_history && impl_->motion_committed?
                native_address(impl_->panel_samples[read_index].view()):0u;
            impl_->current_panel_samples=m.current_panel_samples;
        }
        if(moving_panel && impl_->config.lane==Lane::Quality) {
            auto normals=impl_->material_normals.stage(context,*radiance,*packed_observation,
                impl_->phase_state.repeated_camera(),use_history,m.stationary_radiance==0u,commands);
            if(!normals.ok())return backend::Result<backend::ProcessOutput>::failure(normals.error());
            m.current_material_normals=normals.value().current;
            m.previous_material_normals=normals.value().previous;
            m.vertex_frame=normals.value().vertex_frame;
              impl_->current_regular_material_normals=m.current_material_normals;
        }
        if(auto error=impl_->measurement_runtime.dispatch(m, commands, regular_layout))
            return backend::Result<backend::ProcessOutput>::failure(error);
        if(impl_->split_history)impl_->pending_history_commit=m;
        impl_->history_lifecycle.mark_pending(rigid->history_generation);
        impl_->telemetry.frame_index=context.frame_index;
        impl_->telemetry.history_valid_before=use_history;
        impl_->telemetry.temporal_used=use_history;
        impl_->telemetry.output_ready=true;
        module_telemetry_.name="PhysicalLfEarlyMeasurement";
        module_telemetry_.cold_bypass=false;
        module_telemetry_.dispatch_count=impl_->split_history ? 2u : 1u;
        // One device-only commit publishes the surface and optional complete historical tail.
        module_telemetry_.copy_count=impl_->split_history ? 1u : 0u;
        module_telemetry_.read_bytes=context.sample_extent.area()*(8u+32u+(impl_->single_tail?32u:(impl_->split_history?16u:0u))+(use_history?4u*40u:0u));
        module_telemetry_.write_bytes=context.sample_extent.area()*(32u+8u+8u+(impl_->single_tail?8u:0u));
        if(impl_->config.lane==Lane::Quality)module_telemetry_.write_bytes+=(context.sample_extent.area()/32u)*sizeof(uint32_t); // Conservative one atomic OR per warp; actual zero for all-direct warps.

        module_telemetry_.output_ready=true;
        return backend::Result<backend::ProcessOutput>::success({{},use_history,true});
    }
}

backend::Result<backend::PresentOutput> PhysicalLfReprojectionTaa::present(
    const backend::FrameContext &context,
    const backend::FrameResources &,
    const backend::ProcessOutput &output,
    CommandBatch &commands) noexcept {
    if (!impl_->telemetry.output_ready || !output.presentation_only ||
        output.radiance.size() != 0u ||
        impl_->presentation_source.size() != impl_->sample_extent.area() ||
        context.output_extent != impl_->output_extent) {
        return backend::Result<backend::PresentOutput>::failure({
            backend::ErrorCode::InvalidState,
            "Physical LF TAA output is unavailable for composition."});
    }
    return backend::Result<backend::PresentOutput>::failure({
        backend::ErrorCode::InvalidState,
        "Physical LF TAA composition is owned by DenoisePipeline."});
}

} // namespace lfs::denoise
