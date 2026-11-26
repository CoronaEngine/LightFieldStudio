#pragma once
#include "ssat/phase2/calibrated_view_chart.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lfs::denoise::resident_chart {

// The input ViewIds are copied from the actual Phase1 table. This module never
// evaluates or approximates the display interlacing formula.
using Lattice = lightfield::CalibratedLattice;
using Layout=view_chart::Layout;
using Block=view_chart::Block;
namespace detail {
using Pattern = std::array<uint32_t, 34>;
struct PatternHash {
    size_t operator()(const Pattern &p) const noexcept {
        uint64_t h=1469598103934665603ull;
        for (auto v:p) {h^=v; h*=1099511628211ull;}
        return static_cast<size_t>(h);
    }
};
inline uint32_t spread(uint32_t x) {
    x&=65535u; x=(x|(x<<8u))&0x00ff00ffu;
    x=(x|(x<<4u))&0x0f0f0f0fu; x=(x|(x<<2u))&0x33333333u;
    return (x|(x<<1u))&0x55555555u;
}
inline uint32_t morton(double x, double y) {
    if (!(x>=0.0&&x<=65535.0&&y>=0.0&&y<=65535.0))
        throw std::runtime_error("Calibrated chart Morton coordinate is out of range.");
    return spread(uint32_t(std::nearbyint(x))) | (spread(uint32_t(std::nearbyint(y)))<<1u);
}
struct View {
    std::vector<uint32_t> pool, ids, centers, sources, slots;
    std::vector<Pattern> patterns;
    std::vector<Block> blocks;
};

inline View build_view(const Lattice &l, uint32_t v, std::vector<uint32_t> pool, uint32_t footprint) {
    View out; out.pool=std::move(pool); const size_t count=out.pool.size();
    if(count<25u) throw std::runtime_error("A calibrated view has fewer than 25 sites.");
    std::vector<std::vector<uint32_t>> rows(l.height);
    for(auto i:out.pool)rows[i/l.width].push_back(i);
    std::array<uint32_t,25> nn{};
    out.ids.resize(count);
    std::unordered_map<Pattern,uint32_t,PatternHash> dictionary;
    const double dx=(double(l.x.back())-l.x.front())/(l.width-1u);
    const double dy=(double(l.y.back())-l.y.front())/(l.height-1u);
    const double ratio=std::abs(dy/dx);
    using Candidate=std::pair<double,uint32_t>;
    for(size_t rank=0;rank<count;++rank) {
        const uint32_t i=out.pool[rank],cx=i%l.width,cy=i/l.width;
        const double px=l.x[cx],py=l.y[cy];
        std::priority_queue<Candidate> heap;
        int low=int(cy),high=int(cy)+1;
        while(low>=0||high<int(l.height)) {
            const double dlo=low>=0?std::pow(double(l.y[low])-py,2):std::numeric_limits<double>::infinity();
            const double dhi=high<int(l.height)?std::pow(double(l.y[high])-py,2):std::numeric_limits<double>::infinity();
            const bool take_low=dlo<=dhi; const double y2=take_low?dlo:dhi;
            if(heap.size()==25u&&y2>heap.top().first)break;
            const auto &row=rows[take_low?low--:high++];
            auto cut=std::lower_bound(row.begin(),row.end(),cx,[&](uint32_t j,uint32_t x){return j%l.width<x;});
            int a=int(cut-row.begin())-1,b=a+1;
            while(a>=0||b<int(row.size())) {
                const double da=a>=0?std::pow(double(l.x[row[a]%l.width])-px,2):std::numeric_limits<double>::infinity();
                const double db=b<int(row.size())?std::pow(double(l.x[row[b]%l.width])-px,2):std::numeric_limits<double>::infinity();
                const bool left=da<=db; const double d2=y2+(left?da:db);
                if(heap.size()==25u&&d2>heap.top().first)break;
                const Candidate z{d2,row[left?a--:b++]};
                if(heap.size()<25u)heap.push(z);
                else if(z<heap.top()){heap.pop();heap.push(z);}
            }
        }
        if(heap.size()!=25u)throw std::runtime_error("Incomplete calibrated nearest-site query.");
        for(auto &j:nn){j=heap.top().second;heap.pop();}
        std::sort(nn.begin(),nn.end());
        if(!std::binary_search(nn.begin(),nn.end(),i))throw std::runtime_error("Chart query lost its center.");
        Pattern pattern{};
        // Preserve the retained terminal order and its center/neighbor mass.
        auto nine=nn;
        std::sort(nine.begin(),nine.end(),[&](uint32_t a,uint32_t b){
            const double ax=(int(a%l.width)-int(cx))*dx,ay=(int(a/l.width)-int(cy))*dy;
            const double bx=(int(b%l.width)-int(cx))*dx,by=(int(b/l.width)-int(cy))*dy;
            const double ad=ax*ax+ay*ay,bd=bx*bx+by*by;return ad<bd||(ad==bd&&a<b);
        });
        for(size_t k=0;k<9u;++k){const auto j=nine[k];pattern[25u+k]=uint16_t(int16_t(int(j%l.width)-int(cx)))|(uint32_t(uint16_t(int16_t(int(j/l.width)-int(cy))))<<16u);}
        // The terminal singular reflector uses the first nine calibrated sources.
        // Keep the ascending source order within each partition.
        std::array<uint32_t,9> inner;
        std::copy_n(nine.begin(),inner.size(),inner.begin());
        std::sort(inner.begin(),inner.end());
        std::array<uint32_t,25> ordered;
        std::copy(inner.begin(),inner.end(),ordered.begin());
        std::set_difference(nn.begin(),nn.end(),inner.begin(),inner.end(),ordered.begin()+inner.size());
        for(size_t k=0;k<ordered.size();++k){
            const auto j=ordered[k];
            const int x=int(j%l.width)-int(cx),y=int(j/l.width)-int(cy);
            if(x<-32768||x>32767||y<-32768||y>32767)throw std::runtime_error("Chart offset exceeds its signed wire.");
            pattern[k]=uint16_t(int16_t(x)) | (uint32_t(uint16_t(int16_t(y)))<<16u);
        }
        auto [it,inserted]=dictionary.emplace(pattern,uint32_t(out.patterns.size()));
        if(inserted)out.patterns.push_back(pattern);out.ids[rank]=it->second;
    }
    return out;
}
} // namespace detail

inline std::shared_ptr<const Layout> build(const Lattice &l,uint32_t footprint) {
    const uint64_t n=uint64_t(l.width)*l.height;
    if(footprint!=9u && footprint!=25u)throw std::runtime_error("Unsupported view footprint.");

    if(l.width<2u||l.height<2u||!l.views||l.views>256u||n>UINT32_MAX||l.view.size()!=n||l.x.size()!=l.width||l.y.size()!=l.height)
        throw std::runtime_error("Invalid calibrated lattice extent.");
    for(size_t i=0;i<l.x.size();++i)if(!std::isfinite(l.x[i])||(i&&l.x[i]<=l.x[i-1]))throw std::runtime_error("Focal X is not finite and strictly increasing.");
    for(size_t i=0;i<l.y.size();++i)if(!std::isfinite(l.y[i])||(i&&l.y[i]>=l.y[i-1]))throw std::runtime_error("Focal Y is not finite and strictly decreasing.");
    std::vector<std::vector<uint32_t>> pools(l.views);
    for(uint32_t i=0;i<n;++i){if(l.view[i]>=l.views)throw std::runtime_error("Actual Phase1 ViewId is out of range.");pools[l.view[i]].push_back(i);}
    auto out=std::make_shared<Layout>();out->width=l.width;out->height=l.height;out->views=l.views;out->footprint=footprint;
    out->bounds={l.x.front(),l.x.back(),l.y.front(),l.y.back()};out->ids.resize(size_t(n));
    std::unordered_map<detail::Pattern,uint32_t,detail::PatternHash> dictionary;
    for(uint32_t batch=0;batch<l.views;batch+=4u){
        std::vector<std::future<detail::View>> tasks;
        for(uint32_t v=batch;v<std::min(batch+4u,l.views);++v)tasks.push_back(std::async(std::launch::async,[&,v]{return detail::build_view(l,v,std::move(pools[v]),footprint);}));
        for(auto &task:tasks){auto view=task.get();const size_t count=view.pool.size();
            std::vector<uint32_t> remap(view.patterns.size());
            for(size_t k=0;k<view.patterns.size();++k){auto [it,inserted]=dictionary.emplace(view.patterns[k],uint32_t(out->offsets.size()/34u));if(inserted)out->offsets.insert(out->offsets.end(),view.patterns[k].begin(),view.patterns[k].end());remap[k]=it->second;}
            for(size_t k=0;k<count;++k)out->ids[view.pool[k]]=remap[view.ids[k]];
        }
    }
    return out;
}
 } // namespace lfs::denoise::resident_chart
#include "ssat/phase2/resident_chart_cache.h"
namespace lfs::denoise::resident_chart {
// One bounded immutable CPU layout survives processor recreation. Verify the
// complete actual lattice, not a hash alone. No normal-frame lookup or transfer.
inline std::shared_ptr<const Layout> cached(const Lattice &l,uint32_t footprint) {
    static std::mutex mutex;
    static std::shared_ptr<const Layout> layout;
    static Lattice input;
    std::lock_guard lock(mutex);
    if(layout&&input.width==l.width&&input.height==l.height&&input.views==l.views&&
        input.x==l.x&&input.y==l.y&&input.view==l.view)return layout;
    const auto hash=disk::fingerprint(l);
    const auto file=chart_cache_io::directory()/("resident1-"+std::to_string(hash)+".bin");
    auto next=disk::load(file,l,hash);
    if(!next) {
        std::fprintf(stderr,"Building the calibrated neighborhood for %u x %u sites.\n",l.width,l.height);
        next=build(l,footprint);disk::store(file,*next,hash);
        std::fprintf(stderr,"Calibrated neighborhood ready.\n");
    }
    input=l;layout=std::move(next);return layout;
}
} // namespace lfs::denoise::resident_chart
