#pragma once
#include <cuda_runtime.h>

namespace lfs::denoise::native {
// The canonical composer transfer, shared by standalone and fused execution.
// The physical input has already crossed the TemporalSample32 half boundary.
__device__ __forceinline__ float canonical_physical_transfer(float value) {
    const float exposed = 1.0f - __expf(-value);
    return exposed <= 0.0031308f ? 12.92f * exposed
        : 1.055f * __powf(exposed, 1.0f / 2.4f) - 0.055f;
}
}
