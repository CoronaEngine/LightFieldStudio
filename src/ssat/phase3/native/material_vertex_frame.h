#pragma once
#include <cstdint>

namespace lfs::denoise::mr_native {
struct MaterialVertexMesh {
    std::uint64_t vertices{}, triangles{};
    std::uint32_t vertex_count{}, triangle_count{}, material{}, padding{};
};
struct MaterialVertexFrame {
    std::uint64_t hits{}, meshes{}, transforms{};
    std::uint32_t instance_count{}, padding{};
};
}
