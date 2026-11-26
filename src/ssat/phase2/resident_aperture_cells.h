#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lfs::denoise::resident_chart {
struct ApertureCells {
    std::uint32_t side{}, columns{}, rows{}, count{}, bits{};
    std::vector<std::uint32_t> words;
};
inline ApertureCells aperture_cells(const std::vector<std::uint8_t>& actual_views,
                                   std::uint32_t width,std::uint32_t height,std::uint32_t views) {
    if(!width || !height || !views || views>256u || views>actual_views.size() || actual_views.size()!=std::uint64_t(width)*height)
        throw std::invalid_argument("Invalid calibrated aperture grid.");
    ApertureCells out;
    out.side=std::uint32_t(std::ceil(std::sqrt(float(views)/3.f)));
    auto count_at=[&](unsigned side) {
        return std::uint64_t((width+3u*side-1)/(3u*side))*((height+side-1)/side)*views;
    };
    while(count_at(out.side)>actual_views.size())++out.side;
    const auto cell_width=3u*out.side;
    out.columns=(width+cell_width-1)/cell_width;
    out.rows=(height+out.side-1)/out.side;
    const auto count=count_at(out.side);
    if(count>UINT32_MAX || std::uint64_t(cell_width)*out.side>65535u)
        throw std::invalid_argument("Aperture cell representation exceeds its address range.");
    out.count=std::uint32_t(count);
    out.bits=cell_width*out.side<=255u?8u:16u;
    const auto per_word=32u/out.bits,mask=(1u<<out.bits)-1u;
    out.words.assign((count+per_word-1)/per_word,~0u);
    auto distance=[&](unsigned offset) {
        const int dx=int(2u*(offset%cell_width)+1u)-int(cell_width);
        const int dy=int(2u*(offset/cell_width)+1u)-int(out.side);
        return dx*dx+9*dy*dy;
    };
    for(unsigned y=0;y<height;++y) {
        const unsigned ty=y/out.side,local_y=(y%out.side)*cell_width;
        for(unsigned x=0;x<width;++x) {
            const auto view=actual_views[std::size_t(y)*width+x];
            if(view>=views)throw std::runtime_error("Aperture lookup received an out-of-range Phase1 ViewId.");
            const auto cell=(std::size_t(view)*out.rows+ty)*out.columns+x/cell_width;
            const auto shift=unsigned(cell%per_word)*out.bits,local=local_y+x%cell_width;
            auto& word=out.words[cell/per_word];
            const auto last=(word>>shift)&mask;
            if(last==mask || distance(local)<distance(last))word=(word&~(mask<<shift))|(local<<shift);
        }
    }
    return out;
}
struct TemplateIds {
    std::vector<std::uint32_t> words;
    std::uint32_t bits{32};
};
inline TemplateIds pack_ids(const std::vector<std::uint32_t>& ids,std::size_t template_count) {
    TemplateIds out;
    if(template_count>65536u)return out;
    out.bits=16u;out.words.resize((ids.size()+1u)/2u,0u);
    for(std::size_t i=0;i<ids.size();++i) {
        if(ids[i]>=template_count)throw std::runtime_error("Invalid calibrated template address.");
        out.words[i/2u]|=ids[i]<<unsigned((i%2u)*16u);
    }
    return out;
}
}
