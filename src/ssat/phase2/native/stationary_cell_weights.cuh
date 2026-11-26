// Each calibrated view cell has a fixed 25-site spatial stencil while the
// camera and rigid geometry are stationary. The signal and noise stay live.
namespace {
template<int Width>
__device__ void build_cell_coefficients(const EncodedAtrousNativeParams &p) {
    if(!*reinterpret_cast<const unsigned*>(p.weights_refresh))return;
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=p.atlas_cells)return;
    auto *coeff=reinterpret_cast<unsigned*>(p.cell_coefficients);
    const unsigned i=reinterpret_cast<const unsigned*>(p.atlas)[cell];
    if(i>=p.count){coeff[size_t(6)*p.atlas_cells+cell]=0;return;}
    const auto *obs=reinterpret_cast<const AtlasObservation*>(p.packed_observation);
    const auto a=obs[i];
    if((a.metadata&3u)!=1u){coeff[size_t(6)*p.atlas_cells+cell]=0;return;}
    const auto n=unpack_normal(a.packed_geometric_normal_oct16);
    const unsigned pattern=chart_template_id(p,i);
    const auto *off=reinterpret_cast<const unsigned*>(p.view_chart_offsets)+size_t(pattern)*34;
    const float inv_radius=18.8495559f/(25.f*p.static_view_count);
    unsigned sum=0, word=0, last=0;
    for(int k=0;k<25;++k) {
        const unsigned b=off[k];
        const int dx=int(static_cast<short>(b)),dy=int(static_cast<short>(b>>16));
        const unsigned j=unsigned(int(i)+dy*Width+dx);
        const float geo=atlas_geometry(a,obs[j],n);
        const unsigned q=__float2uint_rn(255.f*geo*__expf(-(dx*dx/9.f+dy*dy)*inv_radius));
        sum+=q;
        if(k==24)last=q;
        else {
            word|=q<<((k&3)*8u);
            if((k&3)==3){
                coeff[size_t(k>>2)*p.atlas_cells+cell]=word;
                word=0;
            }
        }
    }
    const bool singular=((a.metadata>>2u)&127u)==kSurfaceNearSpecular;
    coeff[size_t(6)*p.atlas_cells+cell]=last|(sum<<8u)|(unsigned(singular)<<24);
}

template<int Width>
__device__ void apply_cell_coefficients(const EncodedAtrousNativeParams &p) {
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=p.atlas_cells)return;
    const auto *coeff=reinterpret_cast<const unsigned*>(p.cell_coefficients);
    const unsigned last=coeff[size_t(6)*p.atlas_cells+cell], mass=(last>>8u)&8191u;
    if(!mass)return;
    const unsigned i=reinterpret_cast<const unsigned*>(p.atlas)[cell];
    const unsigned pattern=chart_template_id(p,i);
    const auto *off=reinterpret_cast<const unsigned*>(p.view_chart_offsets)+size_t(pattern)*34;
    const auto *signal=reinterpret_cast<const Half4*>(p.scratch_a);
    const bool singular=(last&(1u<<24))!=0u;
    Vec3 total{0.f,0.f,0.f},square{0.f,0.f,0.f},noise{0.f,0.f,0.f};
    unsigned word=0;
    for(int k=0;k<25;++k){
        if((k&3)==0)word=k==24?last:coeff[size_t(k>>2)*p.atlas_cells+cell];
        const unsigned q=(word>>((k&3)*8u))&255u;
        if(!q)continue;
        const unsigned b=off[k];
        const int dx=int(static_cast<short>(b)),dy=int(static_cast<short>(b>>16));
        const auto m=unpack(signal[unsigned(int(i)+dy*Width+dx)]);
        const Vec3 rgb{m.x,m.y,m.z};
        total=total+rgb*float(q);
        if(singular){
            const Vec3 rgb2{rgb.x*rgb.x,rgb.y*rgb.y,rgb.z*rgb.z};
            const float l=luminance(rgb);
            square=square+rgb2*float(q);
            noise=noise+rgb2*(float(q)*fabsf(m.w)/fmaxf(l*l,1.e-12f));
        }
    }
    const float inv=1.f/float(mass);
    reinterpret_cast<Half4*>(p.scratch_b)[cell]=pack({total.x*inv,total.y*inv,total.z*inv,0.f});
    if(singular) {
        auto *mom=reinterpret_cast<Half4*>(p.atlas_moments)+2ull*cell;
        mom[0]=pack({sqrtf(square.x*inv),sqrtf(square.y*inv),sqrtf(square.z*inv),0.f});
        mom[1]=pack({sqrtf(noise.x*inv),sqrtf(noise.y*inv),sqrtf(noise.z*inv),0.f});
    }
}
}
#define LFS_CELL_CACHE(Name, Width) \
extern "C" __global__ void lfs_build_cell_weights_##Name(EncodedAtrousNativeParams p){build_cell_coefficients<Width>(p);} \
extern "C" __global__ void lfs_apply_cell_weights_##Name(EncodedAtrousNativeParams p){apply_cell_coefficients<Width>(p);}
LFS_CELL_CACHE(2k,4608)
LFS_CELL_CACHE(4k,11520)
LFS_CELL_CACHE(8k,23040)
#undef LFS_CELL_CACHE
