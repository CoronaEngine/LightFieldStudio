#pragma once
#include "backend/frame_contract.h"
#include "native/material_vertex_frame.h"
#include <cstdint>
#include <memory>

namespace lfs::denoise {
struct MaterialNormalViews {
    std::uint64_t current{}, previous{};
    mr_native::MaterialVertexFrame vertex_frame{};
};
class MaterialNormalHistory {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    MaterialNormalHistory();
    ~MaterialNormalHistory();
    backend::Result<MaterialNormalViews> stage(const backend::FrameContext&,
        ocarina::BufferView<ocarina::float4>,ocarina::BufferView<backend::PackedSurfaceObservationV1>,
        bool same_pose,bool history,bool may_fuse,ocarina::CommandBatch&) noexcept;
    void commit() noexcept;
    void abort() noexcept;
    void reset() noexcept;
};
} // namespace lfs::denoise
