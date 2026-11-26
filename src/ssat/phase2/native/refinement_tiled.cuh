// Shared-tile spatial reconstruction with the original 25-source Gaussian estimate.
namespace {
#include "refinement_guide.cuh"
template<int Width>
__device__ __forceinline__ RefinementGuide refinement_guide(const EncodedAtrousNativeParams &p,int x,int y) {
    const unsigned i=y*Width+x;const auto s=load_surface_input(p,i);RefinementGuide g{};
    g.valid=s.validity;g.instance=s.validity==1u?s.instance:~0u;g.flags=s.surface_flags;
    g.a=pack({s.albedo.x,s.albedo.y,s.albedo.z,0.f});
    if(g.valid==1u) {
        const unsigned view=reinterpret_cast<const unsigned char*>(p.static_view_id)[i];
        if(view>=p.static_view_count){fail(p,kFrameStatusInvalidLayout);g.valid=kInvalidGuide;return g;}
        const float u=reinterpret_cast<const float*>(p.static_view_x)[view];
        const float inv=reinterpret_cast<const float*>(p.static_inverse_ray_length)[i];
        const Vec3 d{(reinterpret_cast<const float*>(p.static_focal_x)[x]-u)*inv,reinterpret_cast<const float*>(p.static_focal_y)[y]*inv,p.focal_distance*inv};
        g.p=Vec3{u,0.f,0.f}+d*s.hit;
        // The actual JIT rotates/normalizes original decoded Ng, without
        // requantizing that local normal back through an additional oct16.
        g.n=local_normal(p,s.world_normal);
        if(!finite3(g.p)||!finite3(g.n)){fail(p);g.valid=kInvalidGuide;}
    }
    return g;
}
__device__ __forceinline__ float selected(Vec3 v,unsigned c){return c==0u?v.x:(c==1u?v.y:v.z);}
#include "present_encoded_carrier.cuh"
#include "view_chart_refinement.cuh"
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
template<int Width,int Height>
__device__ float bilateral_diffuse_support(const EncodedAtrousNativeParams &,unsigned,float,float);
#endif


template<int Width,int Height,bool Directional=false>
__device__ __forceinline__ void refinement_dispatch(EncodedAtrousNativeParams p,Half4 *tile,RefinementGuide *guide) {
    constexpr int SW=36,SH=12;const int thread=int(threadIdx.y)*32+int(threadIdx.x);
    const int bx=int(blockIdx.x)*128+int(blockIdx.z%4),by=int(blockIdx.y)*32+int(blockIdx.z/4);
    const int cx=bx+int(threadIdx.x)*4,cy=by+int(threadIdx.y)*4;
    const unsigned group=unsigned(blockIdx.y)*unsigned((Width+127)/128)+blockIdx.x;
    const unsigned coverage_shift = Directional && p.register_motion && !p.hold_same ? 16u : 0u;
    if ((reinterpret_cast<const unsigned*>(p.frame_status)[1u+group] & (1u << (coverage_shift+blockIdx.z))) == 0u) return;
    bool needs_original=false;
    if(cx<Width&&cy<Height){const unsigned md=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[cy*Width+cx].metadata;const unsigned cls=(md>>denoise_native::kPackedSurfaceFlagsShift)&denoise_native::kPackedSurfaceFlagsMask;needs_original=!((md&3u)==1u&&(cls==kSurfaceGlossy||cls==kSurfaceNearSpecular));}
    if(!p.resident_storage && p.register_motion && p.deterministic_environment && cx<Width && cy<Height) {
        const unsigned i=cy*Width+cx;
        const auto md=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[i].metadata;
        if((md&3u)==ssat_native::kPackedSurfaceDomainEnvironment) {
            if(!(Directional && (Width==4608 || Width==11520) && p.register_motion==1u && !p.hold_same))
                present_encoded_carrier<false,Width>(p,i,md,pack({0.f,0.f,0.f,1.e-4f}));
            needs_original=false;
        }
    }
    if(coverage_shift && needs_original) {
        const unsigned i=cy*Width+cx;
        const auto md=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[i].metadata;
        if((md&511u)==(1u|(kSurfaceDiffuse<<2u)) &&
           from_half(reinterpret_cast<const Half4*>(p.field_measurements)[i].w)>=0.f &&
           reinterpret_cast<const denoise_native::TemporalSample32*>(p.regular_measurement_geometry)[i].sigma_albedo.x<to_half(1.f/3.f))
            needs_original=false;
    }
    const unsigned owners=__syncthreads_count(needs_original);
    if(!owners)return;
    constexpr bool regular=Directional && (Width==4608 || Width==11520);
    __shared__ unsigned owners_by_row[regular?8:1];
    __shared__ unsigned long long donor_mask[regular?12:1];
    __shared__ float inverse_luma[regular?432:1];
    const bool sparse=regular && !p.resident_storage && p.register_motion && !p.hold_same && owners<256u;
    if(sparse) {
        const unsigned row=__ballot_sync(0xffffffffu,needs_original);
        if(threadIdx.x==0u)owners_by_row[threadIdx.y]=row;
        __syncthreads();
        if(thread<12) {
            unsigned bits=0u;
            #pragma unroll
            for(int oy=0;oy<5;++oy)if(unsigned(thread-oy)<8u)bits|=owners_by_row[thread-oy];
            const unsigned long long b=bits;
            donor_mask[thread]=b|(b<<1u)|(b<<2u)|(b<<3u)|(b<<4u);
        }
        __syncthreads();
    }
    for(int k=thread;k<SW*SH;k+=256) {
        if(sparse && !(donor_mask[k/SW]&(1ull<<(k%SW))))continue;
        const int x=bx+(k%SW-2)*4,y=by+(k/SW-2)*4;
        if(x>=0&&x<Width&&y>=0&&y<Height){
            guide[k]=refinement_guide<Width>(p,x,y);const auto &g=guide[k];
            const bool direct=g.valid==1u&&(g.flags==kSurfaceGlossy||g.flags==kSurfaceNearSpecular);
            const bool unused=!p.resident_storage && p.register_motion && p.deterministic_environment && g.valid==0u;
            // R4 presents deterministic misses from original raw. A surface's
            // geometry filter cannot consume an environment signal.
            tile[k]=unused?Half4{}:load_signal(reinterpret_cast<const Half4*>(direct?p.scratch_b:p.scratch_a),y*Width+x);
            if constexpr(regular) {
                const auto v=unpack(tile[k]);
                inverse_luma[k]=regular_chart_reciprocal(fmaxf(luminance({v.x,v.y,v.z}),1.e-20f));
            }
        }
        else {tile[k]={};RefinementGuide g{};g.valid=kInvalidGuide;guide[k]=g;}
    }
    __syncthreads();
    const int x=bx+int(threadIdx.x)*4,y=by+int(threadIdx.y)*4;
    if(x>=Width||y>=Height||(!p.resident_storage && !needs_original))return;
    const int ci=(int(threadIdx.y)+2)*SW+int(threadIdx.x)+2;const auto cg=guide[ci];
    Vec4 cv=unpack(tile[ci]);const bool ck=cv.w>=0.f;cv.w=fabsf(cv.w);bool known=ck;
    const Vec3 c{cv.x,cv.y,cv.z},ca{from_half(cg.a.x),from_half(cg.a.y),from_half(cg.a.z)};
    if(cg.valid==1u&&(cg.flags==kSurfaceGlossy||cg.flags==kSurfaceNearSpecular)){
        return; // Exact reflective outputs are produced by the shared same-view dispatch.
    }
    const unsigned carrier=unsigned(x)%3u;const float lc=luminance(c),cc=selected(c,carrier),cs=cc/fmaxf(lc,1.e-20f);
    const float phi=32.f*sqrtf(fmaxf(cv.w,1.e-4f))+1.e-3f;
    const bool measure_plane=Directional && !p.resident_storage && p.register_motion &&
                             cg.valid==1u && cg.flags==kSurfaceDiffuse && cg.p.z>0.f;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    IrradianceFit diffuse_fit{};
    const float ac=selected(ca,carrier),bc=measure_plane?diffuse_response(ac):0.f;
    bool varied=false;
#endif
    Vec3 sum=make_vec3(0.f);float mass=0.f,var=0.f;
    for(int oy=-2;oy<=2;++oy) {
        #pragma unroll
        for(int ox=-2;ox<=2;++ox) {
            const int gi=ci+oy*SW+ox;const auto sg=guide[gi];
            if(sg.valid==kInvalidGuide)continue;
            const Vec4 raw=unpack(tile[gi]);const bool sk=raw.w>=0.f;const float v=fabsf(raw.w);const Vec3 z{raw.x,raw.y,raw.z};
            float geo=0.f,exponent=0.f;bool compatible=cg.valid==0u&&sg.valid==0u;Vec3 dp{};
            if(compatible)geo=1.f;
            if(cg.valid!=0u && sg.valid!=0u && cg.instance==sg.instance) {
                dp=sg.p-cg.p;geo=pow32(fmaxf(dot(cg.n,sg.n),0.f));
                const Vec3 sa{from_half(sg.a.x),from_half(sg.a.y),from_half(sg.a.z)};
                exponent=-fabsf(dot(dp,cg.n))*rsqrtf(fmaxf(dot(dp,dp),1.e-6f))-4.f*length(ca-sa);compatible=true;
            }
            if(!compatible)continue;
            const float l=luminance(z),component=selected(z,carrier);
            const float scale=regular?__fmul_rn(component,inverse_luma[gi]):component/fmaxf(l,1.e-20f),delta=cc-component;
            const float vv=cv.w*cs*cs+v*scale*scale;
            float residual=ck&&sk?.5f*delta*delta/fmaxf(vv,1.e-20f):fabsf(lc-l)/phi;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
            if(measure_plane && !p.hold_same && sg.valid==1u && ck && sk) {
                const float av=from_half(carrier==0u?sg.a.x:(carrier==1u?sg.a.y:sg.a.z));
                if(ac!=av) {
                    const float bs=diffuse_response(av);
                    const float d=fmaf(cc,bs,-component*bc);
                    const float vpair=fmaf(cv.w*cs*cs,bs*bs,v*scale*scale*bc*bc);
                    residual=.5f*d*d/fmaxf(vpair,1.e-20f);
                }
            }
#endif
            const int ax=ox<0?-ox:ox,ay=oy<0?-oy:oy;
            const float kx=ax==0?.375f:(ax==1?.25f:.0625f),ky=ay==0?.375f:(ay==1?.25f:.0625f);
            float w=kx*ky*geo*expf(exponent-residual);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
            if(measure_plane && !p.hold_same && sg.valid==1u) {
                const float av=from_half(carrier==0u?sg.a.x:(carrier==1u?sg.a.y:sg.a.z));
                if(ac!=av) {diffuse_fit.add(bc,diffuse_response(av),component,w,v);varied=true;}
            }
#endif
            known=known&&(sk||w==0.f);sum+=z*w;var+=v*w*w;mass+=w;

        }
    }
    const float inv=1.f/fmaxf(mass,1.e-4f);Vec3 result=sum*inv;
    float output_variance=fmaxf(var/fmaxf(mass*mass,1.e-4f),1.e-4f);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if(varied) {
        const float fitted=diffuse_fit.estimate(bc,selected(sum,carrier),mass,selected(result,carrier));
        if(carrier==0u)result.x=fitted;else if(carrier==1u)result.y=fitted;else result.z=fitted;
        output_variance=fmaxf(diffuse_fit.variance(bc,var,mass,output_variance),1.e-4f);
    }
#endif

#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if constexpr(Directional) {
        if(!p.resident_storage && p.register_motion && !p.hold_same &&
           cg.valid==1u && cg.flags==kSurfaceDiffuse) {
            // The R4 entry reads the R2 destination in scratch_a at tile[ci].
            // scratch_b still contains the earlier M, and is not an R2 observation.
            const Vec3 inner=c;
            const float r2=selected(inner,carrier),r4=selected(result,carrier);
            if(r4!=r2) {
                float restore=bilateral_diffuse_support<Width,Height>(p,y*Width+x,r2,r4);
                const auto m=unpack(reinterpret_cast<const Half4*>(p.field_measurements)[y*Width+x]);
                if(m.w>=0.f) {
                    const Vec3 current{m.x,m.y,m.z};
                    const float z=selected(current,carrier),chroma=z/fmaxf(luminance(current),1.e-20f);
                    const float vM=m.w*chroma*chroma,v2=cv.w*cs*cs;
                    // R2 contains the receiver M. Subtract the largest positive
                    // covariance term before interpreting their agreement as detail.
                    const float d=r4-r2;
                    const float evidence=fmaxf(fmaf(d,r4+r2-2.f*z,-2.f*sqrtf(vM*v2)),0.f);
                    restore=fmaxf(restore,evidence/fmaxf(evidence+v2,1.e-20f));
                }
                result+=(inner-result)*restore;
            }
        }
    }
#endif
    if(!finite3(result)||!isfinite(output_variance)){fail(p);return;}
    const unsigned index=y*Width+x;const auto encoded=pack({result.x,result.y,result.z,known?output_variance:-output_variance});
    if(!Directional && !p.resident_storage && p.register_motion && !p.hold_same && cg.valid==1u && cg.flags==kSurfaceDiffuse) {
        if(!p.canonical){fail(p,kFrameStatusInvalidLayout);return;}
        if(*reinterpret_cast<const unsigned*>(p.frame_status))return;
        // Compare the two actual spatial estimates on the original Half/color
        // boundary. Their propagated noise includes the ES estimate at sites
        // whose temporal variance is still marked unknown.
        const float alb=selected(ca,carrier);
        const float modulation=sqrtf(fmaf(alb,alb,.01f));
        const float energy_to_physical=p.final_energy*modulation;
        const float z=selected(result,carrier);
        const float chroma=z/fmaxf(luminance(result),1.e-20f);
        const float scale2=energy_to_physical*energy_to_physical;
        const float v2=cv.w*cs*cs*scale2;
        const float v4=output_variance*chroma*chroma*scale2;
        const float narrow=from_half(to_half(cc*p.final_energy))*modulation;
        const float wide=from_half(to_half(from_half(to_half(z))*p.final_energy))*modulation;
        const float difference=narrow-wide;
        const float structure=fmaxf(fmaf(difference,difference,-v2-v4),0.f);
        const float retain=structure/fmaxf(structure+v2,1.e-20f);
        const float physical=fmaf(retain,difference,wide);
        if(!isfinite(physical)){fail(p);return;}
        reinterpret_cast<float*>(p.canonical)[4u*(index/3u)+carrier]=physical;
        return;
    }
    const bool terminal_glossy=cg.valid==1u&&(cg.flags&kSurfaceGlossy)!=0u&&(cg.flags&(kSurfaceDiffuse|kSurfaceNearSpecular|kSurfaceTransmission))==0u;
    if(terminal_glossy)store_signal(reinterpret_cast<Half4*>(p.scratch_b),index,encoded);
    else present_encoded_carrier<false,Width>(p,index,reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[index].metadata,encoded);
}
}
#define LFS_REFINEMENT_ENTRY(Name,Width,Height) \
extern "C" __global__ void lfs_encoded_atrous_refinement_coverage_##Name(EncodedAtrousNativeParams p) { \
 __shared__ Half4 tile[36*12];__shared__ RefinementGuide guide[36*12]; \
 if(!valid_params<Width,Height>(p)){if(threadIdx.x==0&&threadIdx.y==0)fail(p,kFrameStatusInvalidLayout);return;} \
 refinement_dispatch<Width,Height>(p,tile,guide); }
LFS_REFINEMENT_ENTRY(2k,4608,2048)
LFS_REFINEMENT_ENTRY(4k,11520,2160)
LFS_REFINEMENT_ENTRY(8k,23040,4320)
#undef LFS_REFINEMENT_ENTRY

#define LFS_DIRECTIONAL_REFINEMENT_ENTRY(Name,Width,Height) \
extern "C" __global__ void lfs_regular_directional_refinement_##Name(EncodedAtrousNativeParams p) { \
 __shared__ Half4 tile[36*12];__shared__ RefinementGuide guide[36*12]; \
 if(!valid_params<Width,Height>(p)){if(threadIdx.x==0&&threadIdx.y==0)fail(p,kFrameStatusInvalidLayout);return;} \
 refinement_dispatch<Width,Height,true>(p,tile,guide); }
LFS_DIRECTIONAL_REFINEMENT_ENTRY(2k,4608,2048)
LFS_DIRECTIONAL_REFINEMENT_ENTRY(4k,11520,2160)
LFS_DIRECTIONAL_REFINEMENT_ENTRY(8k,23040,4320)
#undef LFS_DIRECTIONAL_REFINEMENT_ENTRY

#include "view_chart_cached.cuh"
