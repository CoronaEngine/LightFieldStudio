#pragma once
#include <cstdint>
#include <vector>

namespace lfs::lightfield {
// CPU snapshot of the actual GPU ViewId table and its physical focal lattice.
// Available on explicit cold demand only; never rebuilt/read back on a warm frame.
struct CalibratedLattice {
    uint32_t width{}, height{}, views{};
    std::vector<float> x, y;
    std::vector<uint8_t> view;
};
}
