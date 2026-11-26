// A homogeneous support has true surfaces of the same instance, and all of its
// actual exposure guides requested the linear measurement. The normal and
// albedo may still vary across this support, including at texture boundaries.
template<int Width, int Height>
__device__ __forceinline__ bool measure_uniform_linear_tile(
    const EncodedAtrousNativeParams &params, Half4 *raw, Half4 *filtered,
    SharedGuide *guides, Vec3 *normals) {
    constexpr int sw=38, count=38*22;
    const unsigned lane=threadIdx.y*32u+threadIdx.x;
    const auto reference=load_tile_guide(guides,normals,3*sw+3);
    bool same_support=true;
    for(unsigned i=lane;i<count;i+=512u){
        const auto g=load_tile_guide(guides,normals,int(i));
        same_support &= guide_validity(g)==1u && guide_path_kind(g)==0u && g.instance==reference.instance;
    }
    if(!__syncthreads_and(same_support))return false;
    if(one_ray_plane<sw,22,512>(guides,normals,3*sw+3,int(lane))) {
        plane_exposure<Width,Height>(params,raw,filtered,guides,normals);return true;
    }
    const int base_x=int(blockIdx.x)*32-3,base_y=int(blockIdx.y)*16-3;
    // Both banks always contain the same rounded half colors at their phase
    // boundary. A luminance is stored only in the other bank's free interval.
    float *raw_luma=reinterpret_cast<float*>(filtered);
    float *filtered_luma=reinterpret_cast<float*>(raw);
#ifndef LFS_ENCODED_PLAIN_CURRENT
    for(unsigned i=lane;i<count;i+=512u){
        const auto v=unpack(raw[i]);
        raw_luma[i]=safe_luminance({v.x,v.y,v.z});
    }
    __syncthreads();
#endif
    Half4 pending[2]{};
    #pragma unroll
    for(int pass=0;pass<2;++pass){
        const unsigned i=lane+512u*pass;
        if(i>=36u*20u)continue;
        const int x=1+int(i%36u),y=1+int(i/36u);
        pending[pass]=prefilter_shared_site<Width,Height,sw,true,true>(
            params,raw,guides,normals,x,y,base_x+x,base_y+y,raw_luma);
    }
    __syncthreads();
    #pragma unroll
    for(int pass=0;pass<2;++pass){
        const unsigned i=lane+512u*pass;
        if(i>=36u*20u)continue;
        const unsigned local=(1u+i/36u)*sw+1u+i%36u;
        const auto v=unpack(pending[pass]);
        filtered[local]=pending[pass];
        filtered_luma[local]=luminance({v.x,v.y,v.z});
    }
    __syncthreads();
    const int x=base_x+3+int(threadIdx.x),y=base_y+3+int(threadIdx.y);
    auto measured=atrous_shared_site<Width,sw,false,true,false,true>(
        params,filtered,guides,normals,3+int(threadIdx.x),3+int(threadIdx.y),x,y,1,1,
        filtered_luma);
    // The common-path certificate includes the exact Exposure guide. The
    // normal producer records the same negative sign for each linear sample.
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    measured.w|=0x8000u;
#endif
    store_signal(reinterpret_cast<Half4*>(params.scratch_b),y*Width+x,measured);
    return true;
}
