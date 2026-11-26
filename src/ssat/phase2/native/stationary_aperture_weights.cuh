// The projected cell support depends on the primary geometry and calibration.
// Store it separately from the radiance, which is updated every sample.
namespace {
template<int Width,int Height>
__device__ void cache_aperture_weights(const EncodedAtrousNativeParams &p) {
    if(!*reinterpret_cast<const unsigned*>(p.weights_refresh))return;
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p.count)return;
    auto *weights=reinterpret_cast<unsigned*>(p.stationary_weights);
    const auto *obs=reinterpret_cast<const AtlasObservation*>(p.packed_observation);
    const auto a=obs[i];const bool surface=(a.metadata&3u)==1u;
    const unsigned ch=i%3u;
    const auto albedo=a.linear_albedo_rgb_diffuse_factor;
    const float av=from_half(ch==0?albedo.x:ch==1?albedo.y:albedo.z);
    const unsigned mod=surface?unsigned(to_half(sqrtf(fmaf(av,av,.01f)))):unsigned(to_half(1.f));
    const float scaled_depth=a.hit_distance*reinterpret_cast<const float*>(p.static_inverse_ray_length)[i];
    if(!surface || !(scaled_depth>0.f)){
        const bool direct=p.deterministic_environment && (a.metadata&3u)==2u;
        weights[i]=mod | (unsigned(direct)<<31u);
        return;
    }
    const float parallax=(1.f-1.f/scaled_depth)*p.atlas_inverse_dx;
    weights[p.count+i]=__float_as_uint(parallax);
    const auto *positions=reinterpret_cast<const float*>(p.static_view_x);
    const int view=reinterpret_cast<const unsigned char*>(p.static_view_id)[i];
    const float origin=positions[view],inv_side=1.f/float(p.atlas_side);
    const unsigned x=i%Width,y=i/Width;
    const int cols=(Width+3*p.atlas_side-1)/(3*p.atlas_side),rows=(Height+p.atlas_side-1)/p.atlas_side;
    const int cy=int(floorf((y+.5f)*inv_side-.5f));
    const unsigned cls=(a.metadata>>2u)&127u;
    const int span=(cls==kSurfaceGlossy || cls==kSurfaceNearSpecular)?2:1;
    const float rough=float((a.metadata>>9u)&255u)*(1.f/255.f);
    const float sigma=fmaxf(rough*rough*(p.static_view_count-1)*p.atlas_inverse_span,1.f);
    const float inverse_sigma=cls==kSurfaceDiffuse?0.f:.5f/(sigma*sigma);
    const auto normal=unpack_normal(a.packed_geometric_normal_oct16);
    const auto *lookup=reinterpret_cast<const unsigned*>(p.atlas);
    unsigned sum=0;
    for(int dv=-span;dv<=span;++dv){
        const int v=view+dv;unsigned pack4=0u;
        if(v>=0 && v<int(p.static_view_count)) {
            const float px=fmaf(positions[v]-origin,parallax,float(x));
            if(px>=-3.f*p.atlas_side && px<float(Width)+3.f*p.atlas_side) {
                const int cx=int(floorf((px+.5f)*inv_side/3.f-.5f));
                const float angular=__expf(-dv*dv*inverse_sigma);
                #pragma unroll
                for(unsigned k=0;k<4;++k){
                    const int tx=cx+int(k&1u),ty=cy+int(k>>1u);
                    if(unsigned(tx)>=unsigned(cols) || unsigned(ty)>=unsigned(rows))continue;
                    const unsigned j=lookup[(v*rows+ty)*cols+tx];if(j>=p.count)continue;
                    const float geo=atlas_geometry(a,obs[j],normal);
                    const float dx=(float(j%Width)-px)*(inv_side/3.f),dy=(float(j/Width)-float(y))*inv_side;
                    const unsigned q=__float2uint_rn(255.f*angular*geo*__expf(-2.f*(dx*dx+dy*dy)));
                    pack4|=q<<(k*8u);sum+=q;
                }
            }
        }
        weights[size_t(4+dv)*p.count+i]=pack4;
    }
    weights[i]=mod | (sum<<16u) | (unsigned(cls==kSurfaceNearSpecular)<<29u) | (unsigned(span==2)<<30u);
}

template<int Width,int Height>
__device__ void apply_aperture_weights(const EncodedAtrousNativeParams &p){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=p.count)return;
    const auto *weights=reinterpret_cast<const unsigned*>(p.stationary_weights);
    const unsigned meta=weights[i],ch=i%3u,mass=(meta>>16u)&8191u;
    float value;
    if(meta>>31u){
        const auto raw=reinterpret_cast<const float4*>(p.radiance)[i];
        value=ch==0?raw.x:ch==1?raw.y:raw.z;
    }else if(mass==0u){
        const auto h=reinterpret_cast<const Half4*>(p.scratch_a)[i];
        value=from_half(ch==0?h.x:ch==1?h.y:h.z)*from_half(uint16_t(meta));
    }else{
        const float scale=__uint_as_float(weights[p.count+i]);
        const auto *positions=reinterpret_cast<const float*>(p.static_view_x);
        const unsigned view=reinterpret_cast<const unsigned char*>(p.static_view_id)[i];
        const float origin=positions[view],inv_side=1.f/p.atlas_side;
        const int cols=(Width+3*p.atlas_side-1)/(3*p.atlas_side),rows=(Height+p.atlas_side-1)/p.atlas_side;
        const int cy=int(floorf((i/Width+.5f)*inv_side-.5f));
        const auto *filtered=reinterpret_cast<const Half4*>(p.scratch_b);
        const int span=(meta&(1u<<30u))?2:1;
        const bool singular=(meta&(1u<<29u)) && p.atlas_moments;
        float total=0.f, square=0.f, noise=0.f;
        for(int dv=-span;dv<=span;++dv){
            const unsigned taps=weights[size_t(4+dv)*p.count+i];
            if(taps==0u)continue;
            const int v=int(view)+dv;
            const float px=fmaf(positions[v]-origin,scale,float(i%Width));
            const int cx=int(floorf((px+.5f)*inv_side/3.f-.5f));
            #pragma unroll
            for(unsigned k=0;k<4;++k){
                const unsigned q=(taps>>(k*8u))&255u;if(!q)continue;
                const int index=(v*rows+cy+int(k>>1u))*cols+cx+int(k&1u);
                const auto h=filtered[index];
                total=fmaf(float(q),from_half(ch==0?h.x:ch==1?h.y:h.z),total);
                if(singular) {
                    const auto *moments=reinterpret_cast<const Half4*>(p.atlas_moments)+2ull*index;
                    const auto qr=moments[0],qn=moments[1];
                    const float rms=from_half(ch==0?qr.x:ch==1?qr.y:qr.z);
                    const float sigma=from_half(ch==0?qn.x:ch==1?qn.y:qn.z);
                    square=fmaf(float(q),rms*rms,square);
                    noise=fmaf(float(q),sigma*sigma,noise);
                }
            }
        }
        const float inv=1.f/float(mass);
        float signal=total*inv;
        if(singular) {
            const float local_noise=noise*inv;
            const float structure=fmaxf(0.f,fmaf(-signal,signal,square*inv)-local_noise);
            const float gain=structure/fmaxf(structure+local_noise,1.e-12f);
            const auto center=reinterpret_cast<const Half4*>(p.scratch_a)[i];
            const float current=from_half(ch==0?center.x:ch==1?center.y:center.z);
            signal=fmaf(gain,current-signal,signal);
        }
        value=from_half(to_half(signal))*from_half(uint16_t(meta));
    }
    if(!isfinite(value)){fail(p);return;}
    auto *out=reinterpret_cast<float*>(p.canonical);
    const unsigned slot=(i/3u)*4u+ch;
    if(p.hold_state) {
        const auto md=reinterpret_cast<const AtlasObservation*>(p.packed_observation)[i].metadata;
        if(!register_motion_surface(md))return;
        out[slot]=present_fixed_transition(p,i,fmaxf(0.f,value),out[slot]);
    } else out[slot]=denoise_native::canonical_physical_transfer(fmaxf(0.f,value));
    if(ch==2u)out[(i/3u)*4u+3u]=1.f;
}
}
#define LFS_APERTURE_CACHE(Name,Width,Height) \
extern "C" __global__ void lfs_build_static_weights_##Name(EncodedAtrousNativeParams p){cache_aperture_weights<Width,Height>(p);} \
extern "C" __global__ void lfs_cached_static_color_##Name(EncodedAtrousNativeParams p){apply_aperture_weights<Width,Height>(p);}
LFS_APERTURE_CACHE(2k,4608,2048)
LFS_APERTURE_CACHE(4k,11520,2160)
LFS_APERTURE_CACHE(8k,23040,4320)
#undef LFS_APERTURE_CACHE
