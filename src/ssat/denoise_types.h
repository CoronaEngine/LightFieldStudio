#pragma once
#include "ssat/native/reconstruction_coverage.h"

#include "backend/frame_contract.h"

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace lfs::denoise {

enum class Lane : uint32_t {
    Quality = 0u,
    Performance = 1u,
};

enum class SignalDomain : uint32_t {
    RawPhysicalRgbSecondMoment = 0u,
    DemodulatedRgbVariance = 1u,
    PhysicalRgbVariance = 2u,
    EncodedCarrierScalar = 3u,
};

enum class NativeCurrentEncoding : uint32_t {
    None = 0u,
    LinearHalf4 = 2u,
};

enum class CompositionEncoding : uint32_t {
    None = 0u,
    DemodulatedLinearHalf4 = 3u,
    CanonicalDisplayFloat4 = 4u,
};

enum class SpatialMethod : uint32_t {
    Off = 0u,
    EncodedAtrous = 1u,
};

enum class TemporalMethod : uint32_t {
    Off = 0u,
    PhysicalLfReprojection = 2u,
};

// Borrowed frame-local spatial observation. Ownership stays with Phase2.
struct ResolvedCurrentView {
    SignalDomain domain{SignalDomain::DemodulatedRgbVariance};
    NativeCurrentEncoding encoding{NativeCurrentEncoding::None};
    backend::Extent2D extent{};
    uint64_t packed_signal_address{};
    uint32_t packed_signal_stride_bytes{};
    ocarina::BufferView<uint8_t> packed_signal{};
    uint64_t frame_index{},layout_generation{},layout_hash{};
    uint64_t frame_status_address{};
    // The spatial stage supplies a previous angular correction only on matching held-camera frames.
    uint64_t held_angular_residuals{};
    uint64_t current_angular_residuals{};
    ocarina::BufferView<uint8_t> frame_status{};
    FrameStatusEncoding frame_status_encoding{FrameStatusEncoding::ScalarUInt32};
    RegularR2Coverage regular_r2_coverage{};
    [[nodiscard]] bool valid() const noexcept {
        return extent.valid() && encoding==NativeCurrentEncoding::LinearHalf4 &&
            packed_signal_address!=0 && packed_signal_stride_bytes==sizeof(ocarina::half4) &&
            packed_signal.size()>=extent.area()*packed_signal_stride_bytes &&
            layout_generation!=0 && layout_hash!=0 &&
            ((frame_status_address==0 && frame_status.size()==0) ||
             (frame_status_address!=0 && frame_status.size()>=sizeof(uint32_t)));
    }
};

struct CompositionView {
    CompositionEncoding encoding{CompositionEncoding::None};
    backend::Extent2D sample_extent{};
    uint64_t packed_source_address{};
    uint32_t packed_source_stride_bytes{};
    ocarina::BufferView<uint8_t> packed_source{};
    uint64_t frame_status_address{};
    ocarina::BufferView<uint8_t> frame_status{};
    uint64_t frame_index{},layout_generation{},layout_hash{};
    [[nodiscard]] bool valid() const noexcept {
        const bool canonical=encoding==CompositionEncoding::CanonicalDisplayFloat4;
        const bool format=canonical?sample_extent.width%3u==0u&&packed_source_stride_bytes==sizeof(ocarina::float4):encoding==CompositionEncoding::DemodulatedLinearHalf4&&packed_source_stride_bytes==sizeof(ocarina::half4);
        return format&&sample_extent.valid()&&packed_source_address!=0&&packed_source.size()>=(canonical?sample_extent.area()/3u:sample_extent.area())*packed_source_stride_bytes&&layout_generation!=0&&layout_hash!=0;
    }
};

struct ModuleTelemetry {
    std::string_view name{};
    double direct_ms{};
    uint64_t owned_device_bytes{};
    uint64_t read_bytes{};
    uint64_t write_bytes{};
    uint32_t allocation_count{};
    uint32_t dispatch_count{};
    uint32_t copy_count{};
    uint32_t sync_count{};
    uint64_t accepted_history{};
    uint64_t rejected_history{};
    bool cold_bypass{};
    bool output_ready{};
};

[[nodiscard]] constexpr std::string_view name(Lane lane) noexcept {
    return lane == Lane::Quality ? "quality" : "performance";
}

} // namespace lfs::denoise
