#pragma once
#include "ssat/phase1/calibrated_lattice.h"
#include "ssat/runtime_paths.h"
#ifndef XXH_INLINE_ALL
#define XXH_INLINE_ALL
#endif
#include <xxhash.h>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <span>

namespace lfs::denoise::chart_cache_io {

// Preserve the fingerprint already used by compact resident1 caches.
inline constexpr std::uint64_t lattice_seed=0x3153484346534cull;

inline std::uint64_t fingerprint(const lightfield::CalibratedLattice &l) noexcept {
    auto h=XXH64(l.view.data(),l.view.size(),lattice_seed);
    h=XXH64(l.x.data(),l.x.size()*sizeof(float),h);
    h=XXH64(l.y.data(),l.y.size()*sizeof(float),h);
    const std::array<std::uint32_t,3> dimensions{l.width,l.height,l.views};
    return XXH64(dimensions.data(),sizeof(dimensions),h);
}

inline std::filesystem::path directory() {
    return user_data_directory()/"calibrated-cache";
}

// Readers see either a complete previous file or a complete replacement. An
// unwritable cache must not prevent preparation of the in-memory layout.
template<class Write>
void store_atomically(const std::filesystem::path &target,Write write) noexcept {
    std::filesystem::path temporary;
    try {
        std::filesystem::create_directories(target.parent_path());
        static std::atomic_uint64_t sequence{};
        temporary=target;
        temporary+=".tmp-"+std::to_string(GetCurrentProcessId())+"-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+
            std::to_string(sequence.fetch_add(1,std::memory_order_relaxed));
        std::ofstream file(temporary,std::ios::binary|std::ios::trunc);
        if(file)write(file);
        file.close();
        if(file) {
            std::error_code error;
            std::filesystem::rename(temporary,target,error);
            if(!error)return;
        }
    } catch(...) {}
    if(!temporary.empty()) {
        std::error_code ignored;
        std::filesystem::remove(temporary,ignored);
    }
}
}
