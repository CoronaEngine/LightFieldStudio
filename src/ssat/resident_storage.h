#pragma once
#include "backend/frame_contract.h"
namespace lfs::denoise::resident {
bool dedicated(ocarina::Device*) noexcept;
std::uint64_t geometry_bank(ocarina::Device*,std::uint64_t,std::size_t) noexcept;
std::uint64_t inactive_geometry_bank(ocarina::Device*,std::size_t) noexcept;
struct TraceWorkspaces {
    std::uint64_t spatial{},canonical{},first_hit{},second_hit{};
    explicit operator bool() const noexcept { return spatial && canonical && first_hit && second_hit; }
};
bool native_packed_owner(ocarina::Device*) noexcept;
TraceWorkspaces trace_workspaces(ocarina::Device*,std::size_t,std::uint64_t packed_surface) noexcept;
}
