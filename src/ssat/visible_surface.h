#pragma once

#include "backend/frame_contract.h"
#include <memory>

namespace lfs::denoise::resident {
class VisibleSurface {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    VisibleSurface();
    ~VisibleSurface();
    backend::Error prepare(const backend::FrameContext&) noexcept;
    backend::Result<ocarina::BufferView<backend::PackedSurfaceObservationV1>>
    encode(const backend::FrameContext&,ocarina::BufferView<ocarina::float4>,ocarina::CommandBatch&) noexcept;
};
}
