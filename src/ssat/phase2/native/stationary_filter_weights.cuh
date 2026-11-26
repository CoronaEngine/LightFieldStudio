// A stationary camera and static rigid transforms leave this spatial operator
// unchanged. Store its nine normalized weights and the destination modulation.

namespace {
enum : unsigned { kRawBackground=0, kGridSurface=1, kViewSurface=2, kOther=3 };
template<int Width,int Height>
__device__ void build_stationary_weights(const EncodedAtrousNativeParams &p) {
    if(!*reinterpret_cast<const unsigned*>(p.weights_refresh))return;
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p.count)return;
    const auto *obs=reinterpret_cast<const AtlasObservation*>(p.packed_observation);
    const auto a=obs[i];
    const unsigned md=a.metadata, cls=(md>>2u)&127u;
    const bool surface=(md&3u)==1u;
    const bool chart=surface && (cls==kSurfaceGlossy || cls==kSurfaceNearSpecular);
    const unsigned domain=surface?(chart?kViewSurface:kGridSurface):
        (p.deterministic_environment && (md&3u)==2u?kRawBackground:kOther);
    uint4 code{0,0,0,0};
    if(surface){
        const unsigned ch=i%3u;
        const auto albedo=a.linear_albedo_rgb_diffuse_factor;
        const float av=from_half(ch==0?albedo.x:ch==1?albedo.y:albedo.z);
        const auto normal=unpack_normal(a.packed_geometric_normal_oct16);
        const unsigned *offset=nullptr;
        if(chart){
            const unsigned id=chart_template_id(p,i);
            offset=reinterpret_cast<const unsigned*>(p.view_chart_offsets)+size_t(id)*34+25;
        }
        float weights[9],mass=0.f;
        #pragma unroll
        for(int k=0;k<9;++k){
            int dx=k%3-1,dy=k/3-1;float radial;
            if(chart){
                const unsigned b=offset[k];dx=int(static_cast<short>(b));dy=int(static_cast<short>(b>>16));
                radial=__expf(-(dx*dx/9.f+dy*dy)*2.0943951f/p.static_view_count);
            } else {
                radial=(dx==0?2.f:1.f)*(dy==0?2.f:1.f);
            }
            const int x=int(i%Width)+dx,y=int(i/Width)+dy;
            weights[k]=unsigned(x)<Width && unsigned(y)<Height?radial*atlas_geometry(a,obs[y*Width+x],normal):0.f;
            mass+=weights[k];
        }
        unsigned qsum=0;
        #pragma unroll
        for(int k=0;k<9;++k) {
            const unsigned q=__float2uint_rn(weights[k]*(255.f/mass));
            qsum+=q;
            if(k<4)code.x|=q<<(k*8);else if(k<8)code.y|=q<<((k-4)*8);else code.z=q;
        }
        code.w=unsigned(to_half(sqrtf(fmaf(av,av,.01f)))) | (qsum<<16u);
    }
    code.w|=domain<<28u;
    reinterpret_cast<uint4*>(p.stationary_weights)[i]=code;
}
template<int Width>
__device__ void present_stationary_weights(const EncodedAtrousNativeParams &p) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p.count)return;
    const uint4 code=reinterpret_cast<const uint4*>(p.stationary_weights)[i];
    const unsigned domain=code.w>>28u,ch=i%3u;
    const auto *signal=reinterpret_cast<const Half4*>(p.scratch_a);
    float value;
    if(domain==kRawBackground) {
        const float4 rgb=reinterpret_cast<const float4*>(p.radiance)[i];
        value=ch==0u?rgb.x:ch==1u?rgb.y:rgb.z;
    } else if(domain==kOther) {
        const Half4 h=signal[i];value=from_half(ch==0u?h.x:ch==1u?h.y:h.z);
    } else {
        const bool chart=domain==kViewSurface;
        const unsigned *offset=nullptr;
        if(chart){
            const unsigned id=chart_template_id(p,i);
            offset=reinterpret_cast<const unsigned*>(p.view_chart_offsets)+size_t(id)*34+25;
        }
        float sum=0.f;
        #pragma unroll
        for(int k=0;k<9;++k){
            const unsigned q=k<4?(code.x>>(k*8))&255u:k<8?(code.y>>((k-4)*8))&255u:code.z;
            if(q==0)continue;
            int dx=k%3-1,dy=k/3-1;
            if(chart){const unsigned b=offset[k];dx=int(static_cast<short>(b));dy=int(static_cast<short>(b>>16));}
            const Half4 h=signal[unsigned(int(i)+dy*Width+dx)];
            sum=fmaf(float(q),from_half(ch==0u?h.x:ch==1u?h.y:h.z),sum);
        }
        const unsigned qsum=(code.w>>16u)&511u;
        value=from_half(to_half(sum/float(qsum)))*from_half(uint16_t(code.w));
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
#define LFS_STATIC_WEIGHTS(Name, Width, Height) \
extern "C" __global__ void lfs_build_static_weights_##Name(EncodedAtrousNativeParams p){build_stationary_weights<Width,Height>(p);} \
extern "C" __global__ void lfs_cached_static_color_##Name(EncodedAtrousNativeParams p){present_stationary_weights<Width>(p);}
LFS_STATIC_WEIGHTS(2k,4608,2048)
LFS_STATIC_WEIGHTS(4k,11520,2160)
LFS_STATIC_WEIGHTS(8k,23040,4320)
#undef LFS_STATIC_WEIGHTS
