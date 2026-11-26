// Materialize exact geometry once per shared source, preserving the packed
// normal and RGB Half boundaries. The source layout is calibration-only.
namespace {
struct ViewChartBlock32 {unsigned first,count,source_first,source_count,view,pad0,pad1,pad2;};
static_assert(sizeof(ViewChartBlock32)==32u);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
#include "streamed_finite_chart.cuh"
#endif
template<int Width,int Height,bool Directional=false,bool MovingCache=false,bool CompactColors=false,bool MeasuredMoving=false>
__device__ __forceinline__ void view_chart_cached_dispatch(EncodedAtrousNativeParams p,Half4 *signals,ViewChartGuide36 *guides,Vec4 *source_colors=nullptr,float *source_inverse=nullptr,Vec3 *reflections=nullptr){
#ifdef LFS_ENCODED_PLAIN_CURRENT
 constexpr unsigned wire=0x56434233u,center_plane=2u,center_shift=17u;
#else
 constexpr unsigned wire=0x56434232u,center_plane=7u,center_shift=1u;
#endif
 if(!MovingCache && !p.view_chart_source_count){
   const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=p.count)return;
   const unsigned md=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[i].metadata;
   const unsigned cls=(md>>denoise_native::kPackedSurfaceFlagsShift)&denoise_native::kPackedSurfaceFlagsMask;
   if((md&3u)!=1u||(cls!=kSurfaceGlossy&&cls!=kSurfaceNearSpecular))return;
   const unsigned view=reinterpret_cast<const unsigned char*>(p.static_view_id)[i];
   if(view>=p.static_view_count){fail(p,kFrameStatusInvalidLayout);return;}
   const auto q=direct_chart_guide<Width>(p,i,view);
   const RefinementGuide cg{q.p,q.n,{q.a0,q.a1,q.a2,0u},q.instance,unsigned(q.metadata&3u),unsigned((q.metadata>>2u)&127u)};
   Vec4 v=unpack(load_signal(reinterpret_cast<const Half4*>(p.scratch_a),i));
   const bool known=v.w>=0.f;v.w=fabsf(v.w);
   view_chart_refinement<Width,Height,false,Directional>(p,int(i%Width),int(i/Width),cg,v,known);
   return;
 }
 const unsigned t=threadIdx.x;if(!p.view_chart_centers||!p.view_chart_blocks||!p.view_chart_sources||!p.view_chart_slots||p.view_chart_cache_wire!=wire||blockIdx.x>=p.view_chart_block_count||blockDim.x!=256u){if(t==0u)fail(p,kFrameStatusInvalidLayout);return;}
 const auto b=reinterpret_cast<const ViewChartBlock32*>(p.view_chart_blocks)[blockIdx.x];if(b.count==0u||b.count>256u||b.source_count==0u||b.source_count>512u||b.first>p.count-b.count||b.source_first>p.view_chart_source_count-b.source_count||b.view>=p.static_view_count){if(t==0u)fail(p,kFrameStatusInvalidLayout);return;}
 unsigned index=~0u;bool scope=false;if(t<b.count){index=reinterpret_cast<const unsigned*>(p.view_chart_centers)[b.first+t];if(index>=p.count)fail(p,kFrameStatusInvalidLayout);else{const unsigned md=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[index].metadata,cls=(md>>denoise_native::kPackedSurfaceFlagsShift)&denoise_native::kPackedSurfaceFlagsMask;scope=(md&3u)==1u&&(cls==kSurfaceGlossy||cls==kSurfaceNearSpecular);}}
 if(!__syncthreads_or(scope))return;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
 Vec3 comparison_incident{};
 if constexpr(CompactColors && MovingCache && (Width==4608 || Width==11520)) {
     const unsigned ref=reinterpret_cast<const unsigned*>(p.view_chart_centers)[b.first+b.count/2u];
     if(ref>=p.count){if(t==0u)fail(p,kFrameStatusInvalidLayout);return;}
     const float u=reinterpret_cast<const float*>(p.static_view_x)[b.view];
     const float iv=reinterpret_cast<const float*>(p.static_inverse_ray_length)[ref];
     comparison_incident={(reinterpret_cast<const float*>(p.static_focal_x)[ref%Width]-u)*iv,
                          reinterpret_cast<const float*>(p.static_focal_y)[ref/Width]*iv,p.focal_distance*iv};
 }
#endif
 for(unsigned k=t;k<b.source_count;k+=256u){
  const unsigned j=reinterpret_cast<const unsigned*>(p.view_chart_sources)[b.source_first+k];if(j>=p.count||reinterpret_cast<const unsigned char*>(p.static_view_id)[j]!=b.view){fail(p,kFrameStatusInvalidLayout);continue;}
  const auto o=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[j];const auto contract=ssat_native::validate_packed_surface_observation_v1(o);if(contract.status){fail(p);continue;}
  const unsigned valid=contract.domain==ssat_native::kPackedSurfaceDomainSurface?1u:0u,cls=valid?contract.surface_flags:0u;ViewChartGuide36 g{};g.instance=valid?unsigned(o.surface_key>>32u):~0u;g.a0=o.linear_albedo_rgb_diffuse_factor.x;g.a1=o.linear_albedo_rgb_diffuse_factor.y;g.a2=o.linear_albedo_rgb_diffuse_factor.z;
  unsigned interface_code=0u;
  const bool consumed=valid&&(cls==kSurfaceGlossy||cls==kSurfaceNearSpecular);
  if(consumed){
   const int x=int(j%Width),y=int(j/Width);const float u=reinterpret_cast<const float*>(p.static_view_x)[b.view],iv=reinterpret_cast<const float*>(p.static_inverse_ray_length)[j];const Vec3 d{(reinterpret_cast<const float*>(p.static_focal_x)[x]-u)*iv,reinterpret_cast<const float*>(p.static_focal_y)[y]*iv,p.focal_distance*iv};g.p=Vec3{u,0.f,0.f}+d*o.hit_distance;
   g.n=local_normal(p,unpack_normal(o.packed_geometric_normal_oct16));if(!finite3(g.p)||!finite3(g.n)){fail(p);continue;}
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
   if constexpr(MovingCache && CompactColors && (Width==4608 || Width==11520)) {
       if(cls==kSurfaceNearSpecular || cls==kSurfaceGlossy) {
           // The moving 2K finite chart uses the full measured RGB and
           // geometric guide. Only the narrow chart reads the optical axis,
           // and its donor gate requires the same surface flags.
           const bool radiometric_finite=Width==4608 && cls==kSurfaceGlossy &&
               p.register_motion==1u && !p.hold_same && p.regular_measurement_geometry && p.regular_repeat_state;
           const auto code=!radiometric_finite && p.regular_material_normals?
               reinterpret_cast<const unsigned*>(p.regular_material_normals)[j]:0xffffffffu;
           Vec3 axis{};
           if(code!=0xffffffffu) {
               const Vec3 ns=local_normal(p,unpack_normal(code));
               axis=d-ns*(2.f*dot(d,ns));
               const float front=__uint_as_float(reinterpret_cast<const unsigned*>(p.regular_material_normals)[size_t(p.count)+g.instance]);
               if(front<0.f)axis={};
               else if(front>0.f) {
                   const float eta=dot(d,g.n)<0.f?front:1.f/front;
                   const Vec3 face=dot(d,ns)<=0.f?ns:ns*(-1.f);
                   const float ci=fminf(1.f,fmaxf(0.f,-dot(d,face))),ratio=1.f/eta;
                   const float ct2=1.f-ratio*ratio*fmaxf(0.f,1.f-ci*ci);
                   // The same-view spatial footprint already includes the change
                   // in incoming direction between neighboring focal-plane sites.
                   // Compare only the additional differential due to the actual
                   // interface frames. Fresnel and branch validity use each real ray.
                   const Vec3 anchor=comparison_incident;
                   const float projection=dot(anchor,face);
                   axis=anchor-face*(2.f*projection);
                   if(ct2>0.f) {
                       const float ct=sqrtf(ct2);
                       const float s=(ci-eta*ct)/fmaxf(ci+eta*ct,1.e-20f);
                       const float t=(eta*ci-ct)/fmaxf(eta*ci+ct,1.e-20f);
                       const float fresnel=fminf(1.f,.5f*(s*s+t*t));
                       const bool transmit=fresnel<.5f;
                       if(transmit) {
                           const float ca=fminf(1.f,fmaxf(-projection,0.f));
                           const float v=1.f-ratio*ratio*fmaxf(0.f,1.f-ca*ca);
                           axis=projection<=0.f && v>0.f?anchor*ratio+face*fmaf(ratio,ca,-sqrtf(v)):Vec3{};
                       }
                       const unsigned minority=__float2uint_rn(126.f*fminf(fresnel,1.f-fresnel));
                       interface_code=unsigned(transmit)|(minority<<1u);
                   }
               }
           }
           g.n={chart_directions(g.n.x,axis.x),chart_directions(g.n.y,axis.y),chart_directions(g.n.z,axis.z)};
       }
   }
#endif
  }
  g.metadata=static_cast<unsigned short>(valid|(cls<<2u)|(interface_code<<9u));const auto signal=consumed?load_signal(reinterpret_cast<const Half4*>(p.scratch_a),j):Half4{};
  signals[k]=signal;guides[k]=g;
  if constexpr (MovingCache) {
      const Vec4 rgb=unpack(signal);
      const float l=luminance({rgb.x,rgb.y,rgb.z}),denom=fmaxf(l,1.e-20f);
      if constexpr (CompactColors) source_inverse[k]=regular_chart_reciprocal(denom);
      else source_colors[k]={rgb.x/denom,rgb.y/denom,rgb.z/denom,l};
  }
 }
 __syncthreads();if(!scope||index>=p.count)return;const unsigned slot=(reinterpret_cast<const unsigned*>(p.view_chart_slots)[size_t(center_plane)*p.count+b.first+t]>>center_shift)&511u;if(slot>=b.source_count||reinterpret_cast<const unsigned*>(p.view_chart_sources)[b.source_first+slot]!=index){fail(p,kFrameStatusInvalidLayout);return;}
 auto q=guides[slot];Vec3 cached_axis{};
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
 if constexpr(MovingCache && CompactColors && (Width==4608 || Width==11520)) {
     cached_axis=chart_direction<true>(q.n);
     q.n=chart_direction<false>(q.n);
 }
#endif
 const RefinementGuide cg{q.p,q.n,{q.a0,q.a1,q.a2,static_cast<unsigned short>(q.metadata>>9u)},q.instance,unsigned(q.metadata&3u),unsigned((q.metadata>>2u)&127u)};Vec4 c=unpack(signals[slot]);const bool known=c.w>=0.f;c.w=fabsf(c.w);
 if(cg.valid!=1u||(cg.flags!=kSurfaceGlossy&&cg.flags!=kSurfaceNearSpecular)){fail(p,kFrameStatusInvalidLayout);return;}
#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
 if constexpr(CompactColors) {
     const Vec3 receiver_axis=cached_axis;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
     if constexpr(MeasuredMoving) {
         static_assert(Directional && MovingCache && (Width==4608 || Width==11520));
         if(cg.flags==kSurfaceGlossy) {
             stream_finite_chart<Width>(p,index,b.first+t,slot,cg,c,known,signals,guides,source_inverse,cached_axis);
         } else {
             compact_view_chart_refinement<Width,Height,true,Directional,MovingCache,true,true>(
                 p,int(index%Width),int(index/Width),cg,c,known,signals,guides,b.first+t,
                 reflections,nullptr,source_inverse,source_inverse[slot],receiver_axis,slot);
         }
         return;
     } else
     if constexpr(MovingCache && (Width==4608 || Width==11520)) {
         if(p.register_motion==1u && !p.hold_same && cg.flags==kSurfaceGlossy &&
            p.regular_measurement_geometry && p.regular_repeat_state) {
             stream_finite_chart<Width>(p,index,b.first+t,slot,cg,c,known,signals,guides,source_inverse,cached_axis);
             return;
         }
         if(p.register_motion==1u && !p.hold_same && cg.flags==kSurfaceNearSpecular &&
            p.regular_measurement_geometry && p.regular_repeat_state) {
             compact_view_chart_refinement<Width,Height,true,Directional,MovingCache,true,true>(
                 p,int(index%Width),int(index/Width),cg,c,known,signals,guides,b.first+t,
                 reflections,nullptr,source_inverse,source_inverse[slot],receiver_axis,slot);
             return;
         }
     }
     if constexpr(!MeasuredMoving)
#endif
         compact_view_chart_refinement<Width,Height,true,Directional,MovingCache,true>(
             p,int(index%Width),int(index/Width),cg,c,known,signals,guides,b.first+t,
             reflections,nullptr,source_inverse,source_inverse[slot],receiver_axis,slot);
 }
 else
#endif
 view_chart_refinement<Width,Height,true,Directional,MovingCache>(p,int(index%Width),int(index/Width),cg,c,known,signals,guides,b.first+t,nullptr,source_colors);
}
}
#define LFS_VIEW_CHART_NORMAL_ENTRY(Name,Width,Height) \
extern "C" __global__ void lfs_encoded_atrous_view_chart_##Name(EncodedAtrousNativeParams p){ \
 __shared__ Half4 signal[512];__shared__ ViewChartGuide36 guide[512]; \
 if(!valid_params<Width,Height>(p)){if(threadIdx.x==0u)fail(p,kFrameStatusInvalidLayout);return;}view_chart_cached_dispatch<Width,Height>(p,signal,guide); }
LFS_VIEW_CHART_NORMAL_ENTRY(2k,4608,2048)
LFS_VIEW_CHART_NORMAL_ENTRY(4k,11520,2160)
LFS_VIEW_CHART_NORMAL_ENTRY(8k,23040,4320)
#undef LFS_VIEW_CHART_NORMAL_ENTRY

#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
#define LFS_DIRECTIONAL_CHART_ENTRY(Name,Width,Height) \
extern "C" __global__ void lfs_regular_directional_view_chart_##Name(EncodedAtrousNativeParams p){ \
 __shared__ Half4 signal[512];__shared__ ViewChartGuide36 guide[512]; \
 if(!valid_params<Width,Height>(p)){if(threadIdx.x==0u)fail(p,kFrameStatusInvalidLayout);return;}view_chart_cached_dispatch<Width,Height,true>(p,signal,guide); }
LFS_DIRECTIONAL_CHART_ENTRY(2k,4608,2048)
LFS_DIRECTIONAL_CHART_ENTRY(4k,11520,2160)
LFS_DIRECTIONAL_CHART_ENTRY(8k,23040,4320)
#undef LFS_DIRECTIONAL_CHART_ENTRY
#endif

#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
#define LFS_MOVING_DIRECTIONAL true
#define LFS_OPTICAL_CHART_SCRATCH(Width)
#define LFS_OPTICAL_CHART_BUFFER nullptr
#else
#define LFS_MOVING_DIRECTIONAL false
#define LFS_OPTICAL_CHART_SCRATCH(Width)
#define LFS_OPTICAL_CHART_BUFFER nullptr
#endif
#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
#define LFS_MOVING_CHART_ENTRY(Name,Width,Height) \
extern "C" __global__ __launch_bounds__(256,4) void lfs_regular_moving_view_chart_##Name(EncodedAtrousNativeParams p){ \
 __shared__ Half4 signal[512]; __shared__ ViewChartGuide36 guide[512]; __shared__ float inverse_luma[512]; LFS_OPTICAL_CHART_SCRATCH(Width) \
 if(!valid_params<Width,Height>(p)){if(threadIdx.x==0u)fail(p,kFrameStatusInvalidLayout);return;} \
 view_chart_cached_dispatch<Width,Height,LFS_MOVING_DIRECTIONAL,true,true>(p,signal,guide,nullptr,inverse_luma,LFS_OPTICAL_CHART_BUFFER); }
#else
#define LFS_MOVING_CHART_ENTRY(Name,Width,Height) \
extern "C" __global__ void lfs_regular_moving_view_chart_##Name(EncodedAtrousNativeParams p){ \
 __shared__ Half4 signal[512]; __shared__ ViewChartGuide36 guide[512]; __shared__ Vec4 colors[512]; \
 if(!valid_params<Width,Height>(p)){if(threadIdx.x==0u)fail(p,kFrameStatusInvalidLayout);return;} \
 view_chart_cached_dispatch<Width,Height,LFS_MOVING_DIRECTIONAL,true>(p,signal,guide,colors); }
#endif
LFS_MOVING_CHART_ENTRY(2k,4608,2048)
LFS_MOVING_CHART_ENTRY(4k,11520,2160)
LFS_MOVING_CHART_ENTRY(8k,23040,4320)

#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
#define LFS_MEASURED_CHART_ENTRY(Name,Width,Height) \
extern "C" __global__ __launch_bounds__(256,4) void lfs_regular_measured_chart_##Name(EncodedAtrousNativeParams p){ \
 __shared__ Half4 signal[512]; __shared__ ViewChartGuide36 guide[512]; __shared__ float inverse_luma[512]; \
 if(!valid_params<Width,Height>(p)||p.resident_storage||p.register_motion!=1u||p.hold_same|| \
    !p.regular_measurement_geometry||!p.regular_repeat_state){if(threadIdx.x==0u)fail(p,kFrameStatusInvalidLayout);return;} \
 view_chart_cached_dispatch<Width,Height,true,true,true,true>(p,signal,guide,nullptr,inverse_luma); }
LFS_MEASURED_CHART_ENTRY(2k,4608,2048)
LFS_MEASURED_CHART_ENTRY(4k,11520,2160)
#undef LFS_MEASURED_CHART_ENTRY
#endif
#undef LFS_MOVING_CHART_ENTRY
#undef LFS_MOVING_DIRECTIONAL
#undef LFS_OPTICAL_CHART_SCRATCH
#undef LFS_OPTICAL_CHART_BUFFER
#endif
