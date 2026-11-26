// A fixed ray starts a new raw observation series. The motion-conditioned
// ES/M series continues independently in the ordinary reconstruction banks.
namespace {
__device__ void update_fixed_observation(const EncodedAtrousNativeParams &p) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p.count)return;
    auto *fixed=reinterpret_cast<Half4*>(p.hold_state);
    auto *statistics=fixed+p.count;
    const auto a=reinterpret_cast<const AtlasObservation*>(p.packed_observation)[i];
    const auto prev=reinterpret_cast<const denoise_native::TemporalSample32*>(p.hold_previous)[i];
    const auto rgb=reinterpret_cast<const float4*>(p.radiance)[i];
    const auto albedo=a.linear_albedo_rgb_diffuse_factor;
    const bool surface=(a.metadata&3u)==1u;
    const auto mod=[&](unsigned short al) {const float v=from_half(al);return surface?sqrtf(fmaf(v,v,.01f)):1.f;};
    const Vec3 sample{fminf(fmaxf(rgb.x/mod(albedo.x),0.f),500.f),
                      fminf(fmaxf(rgb.y/mod(albedo.y),0.f),500.f),
                      fminf(fmaxf(rgb.z/mod(albedo.z),0.f),500.f)};
    const bool refresh=*reinterpret_cast<const unsigned*>(p.weights_refresh)!=0u;
    const auto old_albedo=prev.signal_physical_mean_rms;
    const bool same=surface && prev.surface_key==a.surface_key && prev.depth==a.hit_distance &&
                    prev.packed_normal==a.packed_geometric_normal_oct16 &&
                    old_albedo.x==albedo.x && old_albedo.y==albedo.y && old_albedo.z==albedo.z && old_albedo.w==albedo.w;
    unsigned n=1u;
    float prior=1.f,population_variance=0.f;
    Vec3 mean=sample;
    if(!refresh && same) {
        const Half4 stats=statistics[i];
        const Vec4 old=unpack(fixed[i]);
        n=min(unsigned(stats.x)+1u,65535u);
        prior=from_half(stats.y);
        const float alpha=1.f/float(n);
        const float previous_sigma=from_half(stats.w);
        const float innovation=luminance(sample)-luminance({old.x,old.y,old.z});
        population_variance=(1.f-alpha)*fmaf(alpha,innovation*innovation,previous_sigma*previous_sigma);
        mean={fmaf(alpha,sample.x-old.x,old.x),fmaf(alpha,sample.y-old.y,old.y),fmaf(alpha,sample.z-old.z,old.z)};
    } else if(same) {
        const float sum_squared_weights=from_half(prev.sigma_albedo.x);
        if(sum_squared_weights>0.f)prior=1.f/sum_squared_weights;
    }
    // n=1 has no estimate of within-ray temporal variation. It leaves the
    // measured motion branch in control across a one-frame reversal.
    const float independent=float(n-1u);
    const float fraction=independent/(independent+prior);
    const float mean_variance=n>1u?population_variance/independent:0.f;
    fixed[i]=pack({mean.x,mean.y,mean.z,mean_variance});
    statistics[i]={static_cast<unsigned short>(n),to_half(prior),to_half(fraction),to_half(sqrtf(fmaxf(population_variance,0.f)))};
}

__device__ __forceinline__ float present_fixed_transition(const EncodedAtrousNativeParams &p,
        unsigned i,float fixed_value,float moving_value) {
    // Both arguments are linear physical radiance for this same calibrated carrier.
    const auto *statistics=reinterpret_cast<const Half4*>(p.hold_state)+p.count;
    const float fraction=from_half(statistics[i].z);
    return denoise_native::canonical_physical_transfer(fmaf(fraction,fixed_value-moving_value,moving_value));
}
}
extern "C" __global__ void lfs_update_fixed_observation(EncodedAtrousNativeParams p) {update_fixed_observation(p);}
