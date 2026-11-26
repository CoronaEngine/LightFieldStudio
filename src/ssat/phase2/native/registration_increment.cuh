__device__ __forceinline__ float registration_age(const EncodedAtrousNativeParams &p,const float *word) {
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if(p.regular_repeat_state)return float(*reinterpret_cast<const unsigned*>(word)&65535u);
#endif
    return *word;
}
__device__ __forceinline__ void set_registration_age(const EncodedAtrousNativeParams &p,float *word,float value) {
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if(p.regular_repeat_state){*reinterpret_cast<unsigned*>(word)=unsigned(fminf(value,65535.f));return;}
#endif
    *word=value;
}
__device__ __forceinline__ bool field_same_physical_ray(const EncodedAtrousNativeParams &p,unsigned i,const AtlasObservation &a) {
    const auto &g=reinterpret_cast<const denoise_native::TemporalSample32*>(p.hold_previous)[i];
    const auto albedo=a.linear_albedo_rgb_diffuse_factor,previous=g.signal_physical_mean_rms;
    const unsigned instance=unsigned(a.surface_key>>32u);
    const auto *rigid=reinterpret_cast<const denoise_native::RigidInstanceMotionRecord*>(p.rigid_records);
    return instance<p.rigid_count && (rigid[instance].metadata.x&3u)==3u &&
        g.surface_key==a.surface_key && g.depth==a.hit_distance && g.packed_normal==a.packed_geometric_normal_oct16 &&
        ((g.metadata>>2u)&31u)==((a.metadata>>2u)&31u) &&
        albedo.x==previous.x && albedo.y==previous.y && albedo.z==previous.z && albedo.w==previous.w;
}
// Use only the repeated physical ray's measured registration correction. The
// instantaneous spatial/presentation estimate remains live at every exposure.
__device__ __forceinline__ float update_registration_increment(const EncodedAtrousNativeParams &p,
        unsigned i,const AtlasObservation &a,float base,float result) {
    auto *history=reinterpret_cast<float*>(p.stationary_weights);
    const float correction=result-base;
    if(!p.hold_same || !p.hold_previous) {
        history[i]=correction;
        return result;
    }
    const bool same=field_same_physical_ray(p,i,a);
    if(!same) {
        history[i]=history[p.count+i]=correction;
        history[2ull*p.count+i]=0.f;set_registration_age(p,history+3ull*p.count+i,1.f);
        return result;
    }
    const float4 last=p.hold_same==1u?make_float4(history[i],history[i],0.f,1.f):
        make_float4(history[i],history[p.count+i],history[2ull*p.count+i],registration_age(p,history+3ull*p.count+i));
    #ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    const float count=fminf(last.w+1.f,p.regular_repeat_state?65535.f:65536.f);
#else
    const float count=fminf(last.w+1.f,65536.f);
#endif
    const float innovation=correction-last.y;
    const float squared=innovation*innovation;
    const float variance=last.w>1.f?fmaf(1.f/last.w,squared-last.z,last.z):squared;
    // At the first identical pair the observed innovation contains only one
    // degree of freedom. Subsequent pairs update their own innovation scale.
    float gain=1.f/count;
    if(last.w>1.f && last.z>1.e-16f && variance>1.e-16f)
        gain=last.z/(last.z+last.w*variance);
    const float mean=fmaf(gain,correction-last.x,last.x);
    history[i]=mean;history[p.count+i]=correction;
    history[2ull*p.count+i]=variance;set_registration_age(p,history+3ull*p.count+i,count);
    return fmaxf(0.f,base+mean);
}
