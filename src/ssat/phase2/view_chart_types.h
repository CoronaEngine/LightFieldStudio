#pragma once
#include "ssat/phase1/calibrated_lattice.h"
#include <array>
#include <cstdint>
#include <vector>

namespace lfs::denoise::view_chart {
using Lattice=lightfield::CalibratedLattice;
struct Block {
    uint32_t first{}, count{}, source_first{}, source_count{}, view{}, pad[3]{};
};
static_assert(sizeof(Block)==32);
struct Layout {
    uint32_t width{}, height{}, views{}, footprint{};
    std::array<float,4> bounds{};
    std::vector<uint32_t> ids, offsets, centers, sources, slots;
    std::vector<Block> blocks;
};
}
