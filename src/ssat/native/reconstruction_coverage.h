#pragma once
#include <cstdint>
namespace lfs::denoise {
enum class FrameStatusEncoding : uint32_t {ScalarUInt32, StatusAndReconstructionCoverageUInt32, StatusAndGridTileCoverageUInt32};
// The leading word is the frame status. Each later word describes 128x32 centers.
// Regular changing-camera Q uses bits 0..15 for R2 and bits 16..31 for R4.
// Other Q dispatches retain their original sixteen R4 parity bits.
enum class RegularR2Coverage : std::uint32_t { Original, AllSources, DeterministicEnvironment };
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr unsigned regular_r2_bit(unsigned x, unsigned y) {
    // Four 64x16 output blocks, each with the four R2 interleave parities.
    return ((y / 16u) & 1u) * 8u + ((x / 64u) & 1u) * 4u + (y & 1u) * 2u + (x & 1u);
}
// Quality coverage is produced by M on the same stream. Normal Performance
// records one bit per 32x16 output tile in bits 0..7 of the same word extent.
// These bits are produced by fused ES1 from the actual first-hit partition.
constexpr uint32_t reconstruction_coverage_columns(uint32_t w){return (w+127u)/128u;}
constexpr uint32_t reconstruction_coverage_words(uint32_t w,uint32_t h){return reconstruction_coverage_columns(w)*((h+31u)/32u);}
}
