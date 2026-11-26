// Repeated, physically identical rays acquire independent raw observations.
// Only the angular correction is calibrated. The original ES/M spatial estimate stays live.
template<int Width>
__device__ __forceinline__ float present_mature_diffuse(const EncodedAtrousNativeParams &p,unsigned i,float base) {
    const unsigned ch=i%3u;
    const auto raw=reinterpret_cast<const float4*>(p.radiance)[i];
    const float sample=ch==0u?raw.x:(ch==1u?raw.y:raw.z);
    auto *moments=reinterpret_cast<half2*>(p.regular_repeat_state);
    moments[4ull*p.count+i]=__floats2half2_rn(fminf(sample,65504.f),fminf(base,65504.f));
    const float value=denoise_native::canonical_physical_transfer(resolve_motion_field<Width>(p,i,base));
    moments[6ull*p.count+i]=__floats2half2_rn(denoise_native::canonical_physical_transfer(base),value);
    return value;
}
template<int Width,bool HasRawSample>
__device__ __forceinline__ float present_motion_canonical(
        const EncodedAtrousNativeParams &p,unsigned i,float base,float sample) {
    const unsigned ch=i%3u;
    if constexpr(!HasRawSample) {
        const float4 raw=reinterpret_cast<const float4*>(p.radiance)[i];
        sample=ch==0u?raw.x:(ch==1u?raw.y:raw.z);
    }
    auto *history=reinterpret_cast<float*>(p.regular_repeat_state);
    auto *moments=reinterpret_cast<half2*>(p.regular_repeat_state);
    auto *counts=reinterpret_cast<unsigned*>(p.regular_repeat_state)+3ull*p.count;
    constexpr float half_max=65504.f;
    const auto a=reinterpret_cast<const AtlasObservation*>(p.packed_observation)[i];
    const bool diffuse=(a.metadata&511u)==(1u|(kSurfaceDiffuse<<2u));
    const float spatial=base;
    constexpr bool share=Width==11520;
    const float response=share?panel_material_response(a,ch):0.f;
    base=share?matching_spatial_base(p,i,a,spatial,response):matching_spatial_base(p,i,a,spatial);
    // The Q diffuse R2 bank remains live through presentation, including the
    // first held frame where the spatial owner begins emitting the wider R4.
    float r2_physical=0.f;
    if(diffuse) {
        const auto rb=load_signal(reinterpret_cast<const Half4*>(p.scratch_b),i);
        const float z=from_half(ch==0u?rb.x:(ch==1u?rb.y:rb.z));
        r2_physical=from_half(to_half(z*p.final_energy))*(share?response:panel_material_response(a,ch));
    }
    if(!p.hold_same || !p.hold_previous) {
        // One extra packed store on moving frames. The next repeat can use the
        // preceding independent raw and center without a second full image or a ray.
        moments[4ull*p.count+i]=__floats2half2_rn(fminf(sample,half_max),fminf(base,half_max));
        const float resolved=resolve_motion_field<Width,share>(p,i,spatial,&a,base,response);
        const float output=denoise_native::canonical_physical_transfer(resolved);
        if(diffuse) {
            const float reference=Width==11520 && resolved==r2_physical?
                output:denoise_native::canonical_physical_transfer(r2_physical);
            moments[6ull*p.count+i]=__floats2half2_rn(reference,output);
        }
        return output;
    }
    const bool same=field_same_physical_ray(p,i,a);
    const unsigned old_n=p.hold_same==1u?1u:counts[i]>>16u;
    const float last_angle=p.hold_same==1u?history[i]:0.f;
    const float2 prior_raw=__half22float2(moments[4ull*p.count+i]);
    const float result=resolve_motion_field<Width,share>(p,i,spatial,&a,base,response);
    const float base_c=denoise_native::canonical_physical_transfer(diffuse?r2_physical:base);
    if(!same || old_n==0u) {
        const float c=denoise_native::canonical_physical_transfer(result);
        moments[4ull*p.count+i]=__floats2half2_rn(fminf(sample,half_max),0.f);
        moments[5ull*p.count+i]=__floats2half2_rn(fminf(base,half_max),0.f);
        moments[6ull*p.count+i]=__floats2half2_rn(base_c,c);
        counts[i]=(counts[i]&65535u)|(1u<<16u);
        return c;
    }
    const bool first=p.hold_same==1u;
    const float2 prior_base=first?make_float2(prior_raw.y,0.f):__half22float2(moments[5ull*p.count+i]);
    const float2 last=first && !diffuse?make_float2(denoise_native::canonical_physical_transfer(prior_raw.y),
                      denoise_native::canonical_physical_transfer(fmaxf(0.f,prior_raw.y+last_angle))):
                      __half22float2(moments[6ull*p.count+i]);
    const unsigned n=min(old_n+1u,65535u);
    const float inv=1.f/float(n);
    const float raw_innovation=sample-prior_raw.x,base_innovation=base-prior_base.x;
    const float raw_mean=fmaf(inv,raw_innovation,prior_raw.x),base_mean=fmaf(inv,base_innovation,prior_base.x);
    const float raw_sigma=first?0.f:prior_raw.y;
    // Carry population standard deviations to retain the range of sparse bright paths in Half2.
    const float raw_var=(1.f-inv)*fmaf(inv,raw_innovation*raw_innovation,raw_sigma*raw_sigma);
    const float base_var=(1.f-inv)*fmaf(inv,base_innovation*base_innovation,prior_base.y*prior_base.y);
    const float variance=(raw_var+float(n)*base_var)/float(n-1u);
    const float field=result-base;
    const float gain=fminf(1.f,fmaxf(0.f,fmaf(field,raw_mean-base,variance)/fmaxf(fmaf(field,field,variance),1.e-20f)));
    const float allowed=fabsf(base_c-last.x);
    const float desired=denoise_native::canonical_physical_transfer(fmaxf(0.f,fmaf(gain,field,base)));
    const float output=fminf(last.y+allowed,fmaxf(last.y-allowed,desired));
    moments[4ull*p.count+i]=__floats2half2_rn(fminf(raw_mean,half_max),fminf(sqrtf(fmaxf(0.f,raw_var)),half_max));
    moments[5ull*p.count+i]=__floats2half2_rn(fminf(base_mean,half_max),fminf(sqrtf(fmaxf(0.f,base_var)),half_max));
    moments[6ull*p.count+i]=__floats2half2_rn(base_c,output);
    counts[i]=(counts[i]&65535u)|(n<<16u);
    return output;
}
