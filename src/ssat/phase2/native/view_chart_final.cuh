// Physical LF terminal reconstruction. Current presentation only; no history write.
namespace {
__device__ __forceinline__ bool field_same_physical_ray(const EncodedAtrousNativeParams &,unsigned,
        const denoise_native::PackedSurfaceObservationV1 &);
template<int Width,int Height>
__device__ __forceinline__ float resolve_held_apertures(const EncodedAtrousNativeParams &p,unsigned i,
        const denoise_native::PackedSurfaceObservationV1 &a,float wide) {
    using Observation=denoise_native::PackedSurfaceObservationV1;
    auto *s=reinterpret_cast<float*>(p.regular_repeat_state);
    const float n=s[2ull*p.count+i];
    const float baseline=denoise_native::canonical_physical_transfer(wide);
    auto *old_base=s+5ull*p.count,*old_output=s+6ull*p.count;
    if(!(n>1.f)) {old_base[i]=old_output[i]=baseline;return baseline;}
    const unsigned ch=i%3u,view=reinterpret_cast<const unsigned char*>(p.static_view_id)[i];
    const unsigned side=p.atlas_side,cols=(Width+3u*side-1u)/(3u*side),rows=(Height+side-1u)/side;
    const auto *sites=reinterpret_cast<const unsigned*>(p.atlas);
    const auto *obs=reinterpret_cast<const Observation*>(p.packed_observation);
    const auto *apertures=reinterpret_cast<const float*>(p.static_view_x);
    const float depth=a.hit_distance*reinterpret_cast<const float*>(p.static_inverse_ray_length)[i];
    float desired=baseline;
    if(depth>0.f && view>0u && view+1u<p.static_view_count) {
        const float shift=(1.f-1.f/depth)*p.atlas_inverse_dx;
        const float invside=1.f/side;
        const float area=(float(i/Width)+.5f)*invside-.5f;
        const int cy=__float2int_rd(area);
        const Vec3 normal=unpack_normal(a.packed_geometric_normal_oct16);
        float left=0.f,right=0.f,left_var=0.f,right_var=0.f;
        bool both=true;
        for(int dir=-1;dir<=1;dir+=2) {
            const int v=int(view)+dir;
            const float px=fmaf(apertures[v]-apertures[view],shift,float(i%Width));
            const int cx=__float2int_rd((px+.5f)*invside/3.f-.5f);
            float total=0.f,mass=0.f,var_sum=0.f;
            #pragma unroll
            for(unsigned t=0;t<4u;++t) {
                const int x=cx+int(t&1u),y=cy+int(t>>1u);
                if(unsigned(x)>=cols || unsigned(y)>=rows)continue;
                const unsigned j=sites[(unsigned(v)*rows+unsigned(y))*cols+unsigned(x)];
                if(j>=p.count)continue;
                const auto b=obs[j];
                if(b.surface_key!=a.surface_key || ((b.metadata^a.metadata)&0x1ffffu))continue;
                if(!(s[2ull*p.count+j]>1.f))continue;
                float geom=1.f;
                if(a.packed_geometric_normal_oct16!=b.packed_geometric_normal_oct16)
                    geom=pow32(fmaxf(dot(normal,unpack_normal(b.packed_geometric_normal_oct16)),0.f));
                if(!(geom>0.f))continue;
                const auto m=repeated_es_measurement(p,j);
                const float component=selected({m.x,m.y,m.z},ch);
                const float scale=component/fmaxf(luminance({m.x,m.y,m.z}),1.e-12f);
                const float dx=(float(j%Width)-px)*(invside/3.f),dy=(float(j/Width)-float(i/Width))*invside;
                const float w=geom*__expf(-2.f*fmaf(dx,dx,dy*dy));
                mass+=w;total=fmaf(w,component,total);
                var_sum=fmaf(w*w,fabsf(m.w)*scale*scale,var_sum);
            }
            if(!(mass>1.e-8f)){both=false;break;}
            if(dir<0){left=total/mass;left_var=var_sum/(mass*mass);}
            else {right=total/mass;right_var=var_sum/(mass*mass);}
        }
        if(both) {
            const auto albedo=a.linear_albedo_rgb_diffuse_factor;
            const float ab=from_half(ch==0u?albedo.x:(ch==1u?albedo.y:albedo.z));
            const float mod=sqrtf(fmaf(ab,ab,.01f));
            const float fraction=(apertures[view]-apertures[view-1u])/(apertures[view+1u]-apertures[view-1u]);
            const float angular=fmaf(fraction,right-left,left)*mod;
            const float var_angle=((1.f-fraction)*(1.f-fraction)*left_var+fraction*fraction*right_var)*mod*mod;
            const float same_direction=(left*mod-wide)*(right*mod-wide);
            const float center=s[i],var_center=s[p.count+i]/(n*(n-1.f));
            const float contrast=angular-wide,separation=center-wide;
            const float signed_support=fminf(1.f,fmaxf(0.f,fmaf(contrast,separation,var_center)/
                                              fmaxf(fmaf(contrast,contrast,var_center),1.e-20f)));
            const float corroborated=fmaxf(0.f,same_direction);
            const float gain=signed_support*corroborated/fmaxf(corroborated+var_angle,1.e-20f);
            desired=denoise_native::canonical_physical_transfer(fmaxf(0.f,fmaf(gain,contrast,wide)));
        }
    }
    const float radius=fabsf(baseline-old_base[i]);
    const float value=fminf(old_output[i]+radius,fmaxf(old_output[i]-radius,desired));
    old_output[i]=value;old_base[i]=baseline;
    return value;
}
template<int Width,int Height,bool MovingOnly=false,bool Resident=false>
__device__ __forceinline__ void view_chart_final_dispatch(EncodedAtrousNativeParams p){
 if constexpr(MovingOnly) {p.hold_same=0u;p.hold_previous=0u;}
 const unsigned i=blockIdx.x*256u+threadIdx.x;if(i>=p.count)return;
 if(!p.canonical){fail(p,kFrameStatusInvalidLayout);return;}
 const auto md=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[i].metadata;const unsigned cls=(md>>denoise_native::kPackedSurfaceFlagsShift)&denoise_native::kPackedSurfaceFlagsMask;
 constexpr bool record_atlas=MovingOnly && !Resident && (Width==4608 || Width==11520);
 constexpr bool acquire_sample=record_atlas && Width==11520;
 float raw_sample=0.f;
 if constexpr(record_atlas) {
     if((acquire_sample || p.atlas_moments) && register_motion_surface(md)) {
         const auto r=reinterpret_cast<const float4*>(p.radiance)[i];
         if constexpr(acquire_sample)raw_sample=selected({r.x,r.y,r.z},i%3u);
         if(p.atlas_moments) {
             const bool valid=finite3({r.x,r.y,r.z}) && fmaxf(fmaxf(r.x,r.y),r.z)<=65504.f;
             // The complete acquisition and the carrier come from one ray.
             reinterpret_cast<Half4*>(p.atlas_moments)[i]=valid?pack({r.x,r.y,r.z,1.f}):Half4{};
         }
     }
 }
 if constexpr(MovingOnly && !Resident && (Width==4608 || Width==11520)) {
     if(p.deterministic_environment && (md&3u)==ssat_native::kPackedSurfaceDomainEnvironment) {
         present_encoded_carrier<false,Width>(p,i,md,pack({0.f,0.f,0.f,1.e-4f}));
         return;
     }

 }
 const bool glossy=(md&3u)==1u&&(cls&kSurfaceGlossy)!=0u&&(cls&(kSurfaceDiffuse|kSurfaceNearSpecular|kSurfaceTransmission))==0u;
 if(!glossy) {
     if constexpr(Resident)return;
     if(p.register_motion && register_motion_surface(md) && !*reinterpret_cast<const unsigned*>(p.frame_status)) {
         const unsigned ch=i%3u, pixel=i/3u;
         auto *canonical=reinterpret_cast<float*>(p.canonical);
         float output=present_motion_canonical<Width,acquire_sample>(p,i,canonical[4u*pixel+ch],raw_sample);
         if constexpr(Width==4608 || Width==11520) {
             if((md&511u)==(1u|(kSurfaceNearSpecular<<2u)))output=combine_previous_footprint(p,i,output);
         }
         canonical[4u*pixel+ch]=output;
         if(ch==2u)canonical[4u*pixel+3u]=1.f;
     }
     return;
 }
 const unsigned channel=i%3u,pixel=i/3u;auto *canonical=reinterpret_cast<float*>(p.canonical);
 // The ordered frame-end guard owns failure fills; successful colors are written once.
 if(*reinterpret_cast<const unsigned*>(p.frame_status)!=0u)return;
 if(!p.view_chart_ids||!p.view_chart_offsets||p.view_chart_wire!=0x56434832u||!p.view_chart_templates||!(p.final_energy>0.f)||!isfinite(p.final_energy)){fail(p,kFrameStatusInvalidLayout);return;}
 const auto *src=reinterpret_cast<const Half4*>(cls==kSurfaceGlossy?p.scratch_b:p.scratch_a);const Vec4 cv=unpack(load_signal(src,i));float result=selected({cv.x,cv.y,cv.z},channel);
 if(!finite3({cv.x,cv.y,cv.z})||!isfinite(cv.w)){fail(p);return;}
 const bool repeated_lobe=!Resident && !MovingOnly && p.regular_repeat_state && p.register_motion && p.hold_same && p.hold_previous && cls==kSurfaceGlossy;
 if(MovingOnly && p.register_motion && cls==kSurfaceGlossy) {
     // These nine addresses come from the same calibrated template and the
     // coefficients were formed on the same real primary hits by Q25.
     const unsigned tid=(Resident?chart_template_id(p,i):reinterpret_cast<const unsigned*>(p.view_chart_ids)[i]);
     if(tid>=p.view_chart_templates){fail(p,kFrameStatusInvalidLayout);return;}
     const auto *sites=reinterpret_cast<const unsigned*>(p.view_chart_offsets)+34ull*p.view_chart_templates+9ull*tid;
     const auto *cache=reinterpret_cast<const unsigned*>(p.regular_repeat_state);
     // Retain the native center mass on the full calibrated aperture.
     // The measured geometry and exposure contrast gate the neighboring rays.
     constexpr float cw=.786986f*.786986f;
     constexpr float nw=(1.f-cw)/8.f;
     float sum=0.f,mass=0.f;
     const bool compact=!Resident && (Width==4608 || Width==11520) && p.view_chart_source_count!=0u;
     const unsigned low=compact?cache[i]:0u,high=compact?cache[p.count+i]:0u;
     unsigned last=0u;
     #pragma unroll
     for(unsigned k=0;k<9u;++k) {
         const unsigned bits=sites[k];
         float weight=cw;
         if(compact) {
             const unsigned byte=k<8u?((k<4u?low:high)>>(8u*(k&3u)))&255u:last;
             if(bits)weight=nw*(float(byte)*(1.f/255.f));else last=byte;
         } else {
             const unsigned packed=cache[size_t(k/2u)*p.count+i];
             if(bits)weight=nw*from_half(static_cast<unsigned short>(packed>>((k&1u)*16u)));
         }
         if(weight>0.f) {
             const int dx=int(static_cast<short>(bits)),dy=int(static_cast<short>(bits>>16u));
             const unsigned j=unsigned(int(i)+dx+dy*Width);
             float color;
             if constexpr(MovingOnly && !Resident && (Width==4608 || Width==11520))
                 color=bits?from_half(reinterpret_cast<const unsigned short*>(src)[4ull*j+channel]):result;
             else {
                 const auto value=unpack(load_signal(src,j));
                 color=selected({value.x,value.y,value.z},channel);
             }
             sum=fmaf(weight,color,sum);
             mass+=weight;
         }
     }
     result=sum/fmaxf(mass,1.e-4f);
 } else if((md&3u)==1u&&glossy){
  const int x=int(i%Width),y=int(i/Width);const auto cg=refinement_guide<Width>(p,x,y);const unsigned vi=reinterpret_cast<const unsigned char*>(p.static_view_id)[i],tid=(Resident?chart_template_id(p,i):reinterpret_cast<const unsigned*>(p.view_chart_ids)[i]);if(tid>=p.view_chart_templates||vi>=p.static_view_count){fail(p,kFrameStatusInvalidLayout);return;}
  const Vec3 ca{from_half(cg.a.x),from_half(cg.a.y),from_half(cg.a.z)};float sum=0.f,mass=0.f;
  constexpr float cw=.786986f*.786986f,nw=(1.f-cw)/8.f;
  #pragma unroll
  for(unsigned k=0u;k<9u;++k){
   const unsigned bits=reinterpret_cast<const unsigned*>(p.view_chart_offsets)[size_t(tid)*34u+25u+k];const int xx=x+int(static_cast<short>(bits)),yy=y+int(static_cast<short>(bits>>16u));if(xx<0||xx>=Width||yy<0||yy>=Height){fail(p,kFrameStatusInvalidLayout);return;}
   const unsigned j=unsigned(yy*Width+xx);if(reinterpret_cast<const unsigned char*>(p.static_view_id)[j]!=vi){fail(p,kFrameStatusInvalidLayout);return;}
   const bool center=j==i;const auto sg=refinement_guide<Width>(p,xx,yy);
   if(center||(sg.valid==1u&&sg.instance==cg.instance&&sg.flags==cg.flags)){
    float w=center?cw:nw;
    if(!center){const Vec3 dp=sg.p-cg.p,sa{from_half(sg.a.x),from_half(sg.a.y),from_half(sg.a.z)};w*=pow32(fmaxf(dot(cg.n,sg.n),0.f))*expf(-fabsf(dot(dp,cg.n))*rsqrtf(fmaxf(dot(dp,dp),1.e-6f))-4.f*length(ca-sa));}
    const Vec4 sv=unpack(load_signal(src,j));if(!finite3({sv.x,sv.y,sv.z})){fail(p);return;}sum+=selected({sv.x,sv.y,sv.z},channel)*w;mass+=w;
   }
  }
  result=sum/fmaxf(mass,1.e-4f);
 }
 result=result*p.final_energy;const float variance=fmaxf(fabsf(cv.w)*p.final_energy*p.final_energy,1.e-4f);
 if(!isfinite(result)||!isfinite(variance)){fail(p);return;}
 // Preserve the original intermediate Half rounding before modulation/transfer.
 float physical=from_half(to_half(result));physical=isfinite(physical)&&physical>=0.f?physical:0.f;
 const auto observation=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[i];
 if(p.deterministic_environment&&(md&3u)==ssat_native::kPackedSurfaceDomainEnvironment){
  const auto raw=reinterpret_cast<const denoise_native::Float4*>(p.radiance)[i];const float v=channel==0u?raw.x:(channel==1u?raw.y:raw.z);physical=isfinite(v)&&v>=0.f?v:0.f;
 }else if((md&3u)==ssat_native::kPackedSurfaceDomainSurface){
  const auto albedo=observation.linear_albedo_rgb_diffuse_factor;const float a=from_half(channel==0u?albedo.x:(channel==1u?albedo.y:albedo.z));physical*=sqrtf(fmaf(a,a,.01f));
 }
 // The measured finite-lobe estimate already contains the 25-site and
 // terminal nine-site refinement from this aperture.
 if constexpr(Resident) {
     canonical[pixel*4u+channel]=p.register_motion && register_motion_surface(md)?
         physical:denoise_native::canonical_physical_transfer(physical);
 } else {
 float output=denoise_native::canonical_physical_transfer(physical);
 if(repeated_lobe)output=resolve_held_apertures<Width,Height>(p,i,observation,physical);
 if constexpr(Width==4608 || Width==11520) {
     if(cls==kSurfaceGlossy)output=combine_previous_footprint(p,i,output);
 }
 canonical[pixel*4u+channel]=output;
 }
 if(channel==2u)canonical[pixel*4u+3u]=1.f;
}
}
#define LFS_VIEW_FINAL_ENTRY(Name,Width,Height) \
extern "C" __global__ void lfs_encoded_atrous_view_final_##Name(EncodedAtrousNativeParams p){if(!valid_params<Width,Height>(p)){if(threadIdx.x==0u)fail(p,kFrameStatusInvalidLayout);return;}view_chart_final_dispatch<Width,Height>(p);}
LFS_VIEW_FINAL_ENTRY(2k,4608,2048)
LFS_VIEW_FINAL_ENTRY(4k,11520,2160)
LFS_VIEW_FINAL_ENTRY(8k,23040,4320)
#undef LFS_VIEW_FINAL_ENTRY

#define LFS_MOVING_PANEL_ENTRY(Name,Width,Height) \
extern "C" __global__ void lfs_moving_panel_##Name(EncodedAtrousNativeParams p){ \
 if(!valid_params<Width,Height>(p)){if(threadIdx.x==0)fail(p,kFrameStatusInvalidLayout);return;} \
 view_chart_final_dispatch<Width,Height,true>(p); }
LFS_MOVING_PANEL_ENTRY(2k,4608,2048)
LFS_MOVING_PANEL_ENTRY(4k,11520,2160)
LFS_MOVING_PANEL_ENTRY(8k,23040,4320)
#undef LFS_MOVING_PANEL_ENTRY

#define LFS_RESIDENT_PANEL_ENTRY(Name,Width,Height) \
extern "C" __global__ void lfs_resident_panel_##Name(EncodedAtrousNativeParams p){ \
 if(!valid_params<Width,Height>(p)){if(threadIdx.x==0)fail(p,kFrameStatusInvalidLayout);return;} \
 view_chart_final_dispatch<Width,Height,false,true>(p); }
LFS_RESIDENT_PANEL_ENTRY(2k,4608,2048)
LFS_RESIDENT_PANEL_ENTRY(4k,11520,2160)
LFS_RESIDENT_PANEL_ENTRY(8k,23040,4320)
#undef LFS_RESIDENT_PANEL_ENTRY
