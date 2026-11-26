#pragma once
#include "ssat/phase2/chart_cache_io.h"
#include "ssat/phase2/view_chart_types.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>

namespace lfs::denoise::view_chart::disk {

inline constexpr uint64_t wire=0x315247544843464cull;
struct Header {
    uint64_t magic{},lattice{},checksum{};
    std::array<uint64_t,6> counts{};
    uint32_t width{},height{},views{},footprint{},version{};
    std::array<float,4> bounds{};
    uint32_t reserved{};
};
static_assert(sizeof(Header)==112 && std::is_trivially_copyable_v<Block>);

inline std::filesystem::path file(uint64_t key,uint32_t footprint) {
    return chart_cache_io::directory()/("regular1-"+std::to_string(key)+"-"+std::to_string(footprint)+".bin");
}

inline auto fields(const Layout &c) {
    return std::array{std::as_bytes(std::span(c.ids)),std::as_bytes(std::span(c.offsets)),
        std::as_bytes(std::span(c.centers)),std::as_bytes(std::span(c.blocks)),
        std::as_bytes(std::span(c.sources)),std::as_bytes(std::span(c.slots))};
}

inline uint64_t checksum(const Layout &c) noexcept {
    uint64_t h=wire;
    for(auto bytes:fields(c))h=XXH64(bytes.data(),bytes.size(),h);
    return h;
}

inline auto lattice_fields(const Lattice &l) {
    return std::array{std::as_bytes(std::span(l.view)),std::as_bytes(std::span(l.x)),std::as_bytes(std::span(l.y))};
}

inline bool exact_input(std::ifstream &stream,const Lattice &l) {
    std::array<std::byte,64*1024> buffer;
    for(auto data:lattice_fields(l)) {
        while(!data.empty()) {
            const auto size=std::min(data.size(),buffer.size());
            if(!stream.read(reinterpret_cast<char*>(buffer.data()),std::streamsize(size)) ||
               std::memcmp(buffer.data(),data.data(),size)!=0)return false;
            data=data.subspan(size);
        }
    }
    return true;
}

inline void store(const std::filesystem::path &path,const Lattice &l,uint64_t key,const Layout &c) noexcept {
    chart_cache_io::store_atomically(path,[&](std::ofstream &out){
        Header header{wire,key,checksum(c),
            {c.ids.size(),c.offsets.size(),c.centers.size(),c.blocks.size(),c.sources.size(),c.slots.size()},
            c.width,c.height,c.views,c.footprint,1u,c.bounds,0u};
        out.write(reinterpret_cast<const char*>(&header),sizeof(header));
        for(auto b:lattice_fields(l))out.write(reinterpret_cast<const char*>(b.data()),std::streamsize(b.size()));
        for(auto b:fields(c))out.write(reinterpret_cast<const char*>(b.data()),std::streamsize(b.size()));
    });
}

template<class T>
inline bool read_vector(std::ifstream &in,std::vector<T> &v,uint64_t count) {
    v.resize(static_cast<size_t>(count));
    return bool(in.read(reinterpret_cast<char*>(v.data()),std::streamsize(count*sizeof(T))));
}

inline std::shared_ptr<const Layout> load(const std::filesystem::path &path,const Lattice &l,
                                          uint64_t key,uint32_t footprint) {
    const uint64_t n=uint64_t(l.width)*l.height;
    if(l.width<2u || l.height<2u || !l.views || l.views>256u || n>UINT32_MAX ||
       l.view.size()!=n || l.x.size()!=l.width || l.y.size()!=l.height || (footprint!=9u && footprint!=25u))return {};
    std::error_code error;
    const auto bytes=std::filesystem::file_size(path,error);
    if(error || bytes<sizeof(Header))return {};
    std::ifstream in(path,std::ios::binary);
    Header h{};
    if(!in.read(reinterpret_cast<char*>(&h),sizeof(h)))return {};
    const std::array<float,4> bounds{l.x.front(),l.x.back(),l.y.front(),l.y.back()};
    if(h.magic!=wire || h.version!=1u || h.lattice!=key || h.footprint!=footprint ||
       h.width!=l.width || h.height!=l.height || h.views!=l.views || h.reserved!=0u ||
       std::memcmp(h.bounds.data(),bounds.data(),sizeof(bounds)))return {};
    const auto [ids,offsets,centers,blocks,sources,slots]=h.counts;
    const uint64_t planes=(9u*footprint+9u+31u)/32u;
    if(ids!=n || centers!=n || slots!=planes*n || !offsets || offsets%34u || offsets>34*n ||
       blocks<(n+255)/256 || blocks>n || sources<n || sources>UINT32_MAX ||
       sources>n*footprint || sources>blocks*512)return {};
    const uint64_t input_bytes=n+(l.x.size()+l.y.size())*sizeof(float);
    const uint64_t field_bytes=(ids+offsets+centers+sources+slots)*sizeof(uint32_t)+blocks*sizeof(Block);
    if(bytes!=sizeof(h)+input_bytes+field_bytes || !exact_input(in,l))return {};
    auto c=std::make_shared<Layout>();
    c->width=l.width;c->height=l.height;c->views=l.views;c->footprint=footprint;c->bounds=bounds;
    if(!read_vector(in,c->ids,ids) || !read_vector(in,c->offsets,offsets) || !read_vector(in,c->centers,centers) ||
       !read_vector(in,c->blocks,blocks) || !read_vector(in,c->sources,sources) || !read_vector(in,c->slots,slots) ||
       checksum(*c)!=h.checksum)return {};
    return c;
}
}
