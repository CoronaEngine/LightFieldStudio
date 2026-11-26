// Shared final Half4/gain/modulation/transfer contract. Only the owning
// reconstruction partition writes this target's canonical component.
#ifdef LFS_ENCODED_PLAIN_CURRENT
template<int Width,bool Diffuse>
__device__ __forceinline__ float resolve_motion_field(const EncodedAtrousNativeParams &,unsigned,float);
#else
template<int Width,bool HasRawSample>
__device__ __forceinline__ float present_motion_canonical(const EncodedAtrousNativeParams &,unsigned,float,float);
template<int Width>
__device__ __forceinline__ float present_mature_diffuse(const EncodedAtrousNativeParams &,unsigned,float);
#endif
__device__ __forceinline__ float combine_previous_footprint(
        const EncodedAtrousNativeParams &p,unsigned i,float current) {
    if(!p.register_motion || p.resident_storage || !p.panel_samples)return current;
    auto *samples=reinterpret_cast<unsigned*>(p.panel_samples);
    const unsigned staged=samples[i],encoded_overlap=staged>>16u;
    if(encoded_overlap==0x7e00u)return current;
    samples[i]=unsigned(to_half(current))|0x8000u;
    const unsigned original=staged&65535u;
    const float overlap=from_half(encoded_overlap&0x7fffu);
    if(p.hold_same || !(original&0x8000u) || !(overlap>0.f))return current;
    const float previous=from_half(original&0x7fffu);
    const float gain=overlap/(1.f+overlap);
    if((encoded_overlap&0x8000u) && current<1.f && previous<1.f) {
        const float now=current<=.04045f?current/12.92f:__powf((current+.055f)/1.055f,2.4f);
        const float old=previous<=.04045f?previous/12.92f:__powf((previous+.055f)/1.055f,2.4f);
        const float a=-log1pf(-fminf(now,.99999994f)),b=-log1pf(-fminf(old,.99999994f));
        return denoise_native::canonical_physical_transfer(fmaxf(0.f,fmaf(gain,b-a,a)));
    }
    return fmaf(gain,previous-current,current);
}
template<bool KnownSurface = false,int Width = 0>
__device__ __forceinline__ void present_encoded_carrier(const EncodedAtrousNativeParams &p,unsigned i,unsigned md,Half4 encoded,float surface_albedo=0.f,float local_reference=0.f,bool repeated=false){
 if(!p.canonical){fail(p,kFrameStatusInvalidLayout);return;}
 const unsigned channel=i%3u,pixel=i/3u;auto *canonical=reinterpret_cast<float*>(p.canonical);
 // The ordered frame-end guard owns failure fills; successful colors are written once.
 if(*reinterpret_cast<const unsigned*>(p.frame_status)!=0u)return;
 const Vec4 cv=unpack(encoded);
 if(!finite3({cv.x,cv.y,cv.z})||!isfinite(cv.w)){fail(p);return;}
 const float result=selected({cv.x,cv.y,cv.z},channel)*p.final_energy;
 const float variance=fmaxf(fabsf(cv.w)*p.final_energy*p.final_energy,1.e-4f);
 if(!isfinite(result)||!isfinite(variance)){fail(p);return;}
 float physical=from_half(to_half(result));physical=isfinite(physical)&&physical>=0.f?physical:0.f;

 if(!KnownSurface&&p.deterministic_environment&&(md&3u)==ssat_native::kPackedSurfaceDomainEnvironment){
  const auto raw=reinterpret_cast<const denoise_native::Float4*>(p.radiance)[i];const float value=channel==0u?raw.x:(channel==1u?raw.y:raw.z);physical=isfinite(value)&&value>=0.f?value:0.f;
 }else if(KnownSurface || (md&3u)==ssat_native::kPackedSurfaceDomainSurface){
  float a=surface_albedo;
  if constexpr (!KnownSurface) {
      const auto albedo=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[i].linear_albedo_rgb_diffuse_factor;
      a=from_half(channel==0u?albedo.x:(channel==1u?albedo.y:albedo.z));
  }
  physical*=sqrtf(fmaf(a,a,.01f));
 }
 if(p.resident_storage) {
     canonical[pixel*4u+channel]=(p.register_motion && (KnownSurface || register_motion_surface(md)))?
         physical:denoise_native::canonical_physical_transfer(physical);
     if(channel==2u)canonical[pixel*4u+3u]=1.f;
     return;
 }
#ifdef LFS_ENCODED_PLAIN_CURRENT
 if constexpr(Width!=0) {
     // Repeated-ray state validates the previous primary class before reuse. A
     // broad lobe with no angular estimate neither reads nor needs that state.
     // The calibrated nine-site owner handles pure glossy and near-specular.
     // The grid owner handles every other class, including pure diffuse.
     constexpr unsigned field_class=KnownSurface?kSurfaceNearSpecular:kSurfaceDiffuse;
     if(p.register_motion && (md&511u)==(1u|(field_class<<2u)))
         physical=resolve_motion_field<Width,!KnownSurface>(p,i,physical);
 }
 float output=denoise_native::canonical_physical_transfer(physical);
 if constexpr(KnownSurface && Width!=0) {
     if(p.register_motion && (md&511u)==(1u|(kSurfaceNearSpecular<<2u))) {
         // The two Half boundaries and color modulation match the unnormalized
         // nine-site reconstruction on this same primary ray.
         const float inner=from_half(to_half(from_half(to_half(local_reference))*p.final_energy));
         const float reference=denoise_native::canonical_physical_transfer(inner*sqrtf(fmaf(surface_albedo,surface_albedo,.01f)));
         auto *history=reinterpret_cast<float*>(p.stationary_weights);
         if(repeated) {
             const float radius=fabsf(reference-history[i]);
             const float last=history[p.count+i];
             output=fminf(last+radius,fmaxf(last-radius,output));
         }
         history[i]=reference;history[p.count+i]=output;
     }
 }

 canonical[pixel*4u+channel]=output;
 if(channel==2u)canonical[pixel*4u+3u]=1.f;
#else
 if constexpr(Width!=0) {
     if(p.register_motion && (KnownSurface || register_motion_surface(md))) {
         // The final panel pass reads this component once after all spatial owners finish.
         canonical[pixel*4u+channel]=physical;
         if(channel==2u)canonical[pixel*4u+3u]=1.f;
         return;
     }
 }
 canonical[pixel*4u+channel]=denoise_native::canonical_physical_transfer(physical);
 if(channel==2u)canonical[pixel*4u+3u]=1.f;
#endif
}
