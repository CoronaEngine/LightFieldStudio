#pragma once

#include "ssat/phase2/chart_cache_io.h"
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace lfs::denoise::resident_chart::disk {
struct Header {
    std::uint64_t magic{}, lattice{}, ids{}, offsets{}, checksum{};
    std::uint32_t width{},height{},views{},version{};
};
static constexpr std::uint64_t magic=chart_cache_io::lattice_seed;
inline std::uint64_t fingerprint(const Lattice& l) noexcept {
    return chart_cache_io::fingerprint(l);
}
inline std::uint64_t payload_hash(const Layout& c) noexcept {
    auto h=XXH64(c.ids.data(),c.ids.size()*sizeof(uint32_t),magic);
    return XXH64(c.offsets.data(),c.offsets.size()*sizeof(uint32_t),h);
}
inline std::shared_ptr<const Layout> load(const std::filesystem::path& path,const Lattice& l,std::uint64_t key) {
    std::error_code error;const auto bytes=std::filesystem::file_size(path,error);
    if(error || bytes<sizeof(Header))return {};
    std::ifstream in(path,std::ios::binary);Header h{};
    if(!in.read(reinterpret_cast<char*>(&h),sizeof(h)))return {};
    if(h.version!=1 || h.magic!=magic || h.lattice!=key || h.width!=l.width || h.height!=l.height || h.views!=l.views ||
       h.ids!=l.view.size() || !h.offsets || h.offsets%34u ||
       h.offsets>std::min<std::uint64_t>(std::uint64_t(l.view.size())*34u,1ull<<27) ||
       bytes!=sizeof(Header)+(h.ids+h.offsets)*sizeof(uint32_t))return {};
    auto c=std::make_shared<Layout>();c->width=l.width;c->height=l.height;c->views=l.views;c->footprint=25;
    c->bounds={l.x.front(),l.x.back(),l.y.front(),l.y.back()};c->ids.resize(h.ids);c->offsets.resize(h.offsets);
    if(!in.read(reinterpret_cast<char*>(c->ids.data()),std::streamsize(h.ids*sizeof(uint32_t))) ||
       !in.read(reinterpret_cast<char*>(c->offsets.data()),std::streamsize(h.offsets*sizeof(uint32_t))) ||
       payload_hash(*c)!=h.checksum)return {};
    return c;
}
inline void store(const std::filesystem::path& path,const Layout& c,std::uint64_t key) noexcept {
    chart_cache_io::store_atomically(path,[&](std::ofstream& f) {
        Header h{magic,key,c.ids.size(),c.offsets.size(),payload_hash(c),c.width,c.height,c.views,1};
        f.write(reinterpret_cast<const char*>(&h),sizeof(h));
        f.write(reinterpret_cast<const char*>(c.ids.data()),std::streamsize(h.ids*sizeof(uint32_t)));
        f.write(reinterpret_cast<const char*>(c.offsets.data()),std::streamsize(h.offsets*sizeof(uint32_t)));
    });
}
}
