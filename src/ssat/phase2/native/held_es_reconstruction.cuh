// Two disjoint temporal sets of current ES observations at a repeated physical
// ray. Each set selects spatial support for the other, so a bright observation
// never chooses its own bilateral weight.
namespace {
__device__ __forceinline__ bool held_measurement_surface(unsigned md) {
    const unsigned cls=(md>>2u)&127u;
#ifdef LFS_ENCODED_PLAIN_CURRENT
    return (md&3u)==1u && (cls==kSurfaceGlossy || cls==kSurfaceNearSpecular);
#else
    return (md&3u)==1u && (cls==kSurfaceDiffuse || cls==kSurfaceGlossy || cls==kSurfaceNearSpecular);
#endif
}

#ifdef LFS_ENCODED_PLAIN_CURRENT
__device__ __forceinline__ float held_physical_radiance(float display) {
    const float exposed=display<=.04045f?display/12.92f:__powf((display+.055f)/1.055f,2.4f);
    return -log1pf(-fminf(exposed,.99999994f));
}
__device__ __forceinline__ float accumulate_held_panel(const EncodedAtrousNativeParams &p,unsigned i,float current,float x=0.f,float y=0.f,float variance=-1.f) {
    auto *bank=reinterpret_cast<float*>(p.stationary_weights);
    const float prior_count=bank[3ull*p.count+i];
    const float count=fminf(prior_count+1.f,32.f);
    const float linear=held_physical_radiance(current);
    const float old=prior_count>0.f?bank[2ull*p.count+i]:linear;
    const auto md=reinterpret_cast<const AtlasObservation*>(p.packed_observation)[i].metadata;
    float gain=1.f/count;
    if((md&511u)==(1u|(kSurfaceGlossy<<2u)) && p.regular_measurement_geometry) {
        const auto &m=reinterpret_cast<const denoise_native::TemporalSample32*>(p.regular_measurement_geometry)[i];
        const float squared=fminf(fmaxf(from_half(m.sigma_albedo.x),0.f),1.f);
        float previous_weight=(count-1.f)*squared;
        if(variance>=0.f && previous_weight>0.f) {
            // Independent alternating exposures can both prefer the current
            // spatial estimate to the older displayed mean. Only their shared
            // reduction in squared error is evidence against the older vote.
            const float d=linear-old;
            const float evidence=fmaxf(0.f,fminf(d*(2.f*x-linear-old),d*(2.f*y-linear-old)));
            if(evidence>0.f)previous_weight*=variance/(variance+evidence);
        }
        gain=1.f/(1.f+previous_weight);
    }
    const float mean=fmaf(gain,linear-old,old);
    bank[2ull*p.count+i]=mean;bank[3ull*p.count+i]=count;
    return denoise_native::canonical_physical_transfer(fmaxf(mean,0.f));
}
#endif
__device__ void collect_held_measurement(const EncodedAtrousNativeParams &p,unsigned i) {
    auto *series=reinterpret_cast<Half4*>(p.hold_state);
    const auto a=reinterpret_cast<const AtlasObservation*>(p.packed_observation)[i];
    if(!held_measurement_surface(a.metadata)) {
        series[i].w=0u;
        return;
    }
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if(p.cell_coefficients) {
        auto *moments=reinterpret_cast<float4*>(p.cell_coefficients);
        const auto raw=reinterpret_cast<const float4*>(p.radiance)[i];
        const Vec3 current{raw.x,raw.y,raw.z};
        if(!finite3(current)){fail(p);return;}
        unsigned prior_count=0u;
        Vec3 prior_mean{};
        float prior_sigma=0.f;
        const bool same=p.previous_held_raw!=HeldRawStorage::Empty && field_same_physical_ray(p,i,a);
        if(same && p.previous_held_raw==HeldRawStorage::HeldSeries) {
            prior_count=series[i].w;
            if(prior_count) {
                const auto old=moments[i];
                prior_mean={old.x,old.y,old.z};prior_sigma=old.w;
            }
        } else if(same && p.previous_held_raw==HeldRawStorage::MovingExposure) {
            const auto old=unpack(reinterpret_cast<const Half4*>(p.atlas_moments)[i]);
            if(old.w>0.f) {
                prior_count=1u;
                prior_mean={old.x,old.y,old.z};
                // The Q panel is independently owned and still contains the
                // last successful frame until R2/R4. This collection is
                // enqueued after current M and before either panel writer.
                const unsigned slot=4u*(i/3u)+i%3u;
                const float previous=reinterpret_cast<const float*>(p.canonical)[slot];
                (reinterpret_cast<float2*>(p.hold_state)+3ull*p.count)[i]=make_float2(previous,previous);
            }
        }
        if((a.metadata&511u)!=(1u|(kSurfaceDiffuse<<2u)) && (!same || p.previous_held_raw!=HeldRawStorage::HeldSeries)) {
            auto *exposure=reinterpret_cast<float2*>(series+p.count);
            exposure[i]=make_float2(0.f,0.f);
            if(same && prior_count) {
                const float display=reinterpret_cast<const float*>(p.canonical)[4u*(i/3u)+i%3u];
                const float t=display<=.04045f?display/12.92f:__powf((display+.055f)/1.055f,2.4f);
                exposure[i]=make_float2(-log1pf(-fminf(t,.99999994f)),1.f);
            }
        }
        const unsigned count=min(prior_count+1u,65535u);
        const float inv=1.f/float(count);
        const Vec3 mean=prior_count?prior_mean+(current-prior_mean)*inv:current;
        const float delta=luminance(current-prior_mean);
        const float sigma=prior_count?sqrtf(fmaxf((1.f-inv)*fmaf(inv,delta*delta,prior_sigma*prior_sigma),0.f)):0.f;
        moments[i]=make_float4(mean.x,mean.y,mean.z,sigma);
        series[i].w=static_cast<unsigned short>(count);
        return;
    }
#endif
#ifdef LFS_ENCODED_PLAIN_CURRENT
    {
        auto *bank=reinterpret_cast<float*>(p.stationary_weights);
        const bool same=p.hold_previous && field_same_physical_ray(p,i,a);
        if(!same)bank[3ull*p.count+i]=0.f;
        else if(p.hold_same==1u) {
            // No writer of the current panel has been enqueued yet.
            const float previous=reinterpret_cast<const float*>(p.canonical)[4u*(i/3u)+i%3u];
            bank[2ull*p.count+i]=held_physical_radiance(previous);
            bank[3ull*p.count+i]=1.f;
        }
    }
#endif
    const Half4 input=series[i];
    const unsigned before=p.hold_same>1u && field_same_physical_ray(p,i,a)?input.w:0u;
    const unsigned count=before<65534u?before+1u:65534u+((before&1u)^1u);
    const unsigned group=before&1u,prior=before/2u;
    Half4 *slot=series+(1ull+group)*p.count+i;
    const Vec3 f{from_half(input.x),from_half(input.y),from_half(input.z)};
    Vec3 mean=f;
    float sigma=0.f;
    if(prior) {
        const Vec4 old=unpack(*slot);
        const Vec3 prev{old.x,old.y,old.z};
        const float inv=1.f/float(prior+1u),diff=luminance(f)-luminance(prev);
        mean=prev+(f-prev)*inv;
        sigma=sqrtf(fmaxf((1.f-inv)*fmaf(inv,diff*diff,old.w*old.w),0.f));
    }
    *slot=pack({mean.x,mean.y,mean.z,sigma});
    series[i].w=static_cast<unsigned short>(count);
}

struct HeldChannel {float mean,variance;};
__device__ __forceinline__ HeldChannel held_channel(Half4 h,unsigned ch,unsigned n) {
    const auto raw=unpack(h);
    const Vec3 rgb{raw.x,raw.y,raw.z};
    const float value=selected(rgb,ch);
    const float chroma=value/fmaxf(luminance(rgb),1.e-12f);
    const float sigma=raw.w*chroma;
    return {value,sigma*sigma/float(n-1u)};
}

__device__ __forceinline__ float compatible_held_color(HeldChannel a,HeldChannel b) {
    const float d=a.mean-b.mean;
    return __expf(-.5f*d*d/fmaxf(a.variance+b.variance,1.e-8f));
}

template<int Width,int Samples>
__device__ void present_held_measurement(const EncodedAtrousNativeParams &p,unsigned i) {
    const auto *obs=reinterpret_cast<const AtlasObservation*>(p.packed_observation);
    const auto a=obs[i];
    if(!held_measurement_surface(a.metadata))return;
    const auto *series=reinterpret_cast<const Half4*>(p.hold_state);
    const unsigned count=series[i].w,ch=i%3u;
    auto *history=reinterpret_cast<float2*>(p.hold_state)+3ull*p.count;
    auto *image=reinterpret_cast<float*>(p.canonical);
    const unsigned slot=4u*(i/3u)+ch;
    const float baseline=image[slot];
    if(count<4u) {
        history[i]=make_float2(baseline,baseline);
#ifdef LFS_ENCODED_PLAIN_CURRENT
        image[slot]=accumulate_held_panel(p,i,baseline);
#endif
        return;
    }
    const unsigned na=(count+1u)/2u,nb=count/2u;
    const HeldChannel ca=held_channel(series[p.count+i],ch,na);
    const HeldChannel cb=held_channel(series[2ull*p.count+i],ch,nb);
    const unsigned tid=chart_template_id(p,i);
    if(tid>=p.view_chart_templates){fail(p,kFrameStatusInvalidLayout);return;}
    // The raw 34-word dictionary contains the original 25-site set followed
    // by the nine nearest sites; the packed block table uses a different order.
    constexpr unsigned first=Samples==9?25u:0u;
    const auto *offsets=reinterpret_cast<const unsigned*>(p.view_chart_offsets)+34ull*tid+first;
    const Vec3 normal=unpack_normal(a.packed_geometric_normal_oct16);
    float wa=0.f,wb=0.f,ma=0.f,mb=0.f,sa=0.f,sb=0.f;
    for(unsigned k=0;k<Samples;++k) {
        const unsigned off=offsets[k];
        const int dx=static_cast<short>(off),dy=static_cast<short>(off>>16u);
        const unsigned j=unsigned(int(i)+dx+Width*dy);
        if(j>=p.count){fail(p,kFrameStatusInvalidLayout);return;}
        const unsigned n=series[j].w;
        if(n<4u)continue;
        const float g=atlas_geometry(a,obs[j],normal);
        if(!(g>0.f))continue;
        const float spatial=g*__expf(-(float(dx*dx)/9.f+float(dy*dy))*2.0943951f/p.static_view_count);
        const auto u=held_channel(series[p.count+j],ch,(n+1u)/2u);
        const auto v=held_channel(series[2ull*p.count+j],ch,n/2u);
        const float w_a=spatial*compatible_held_color(cb,v);
        const float w_b=spatial*compatible_held_color(ca,u);
        ma=fmaf(w_a,u.mean,ma);wa+=w_a;
        mb=fmaf(w_b,v.mean,mb);wb+=w_b;
        // ES1 footprints can overlap. Sum standard deviations before squaring
        // to avoid treating shared original paths as independent observations.
        sa=fmaf(w_a,sqrtf(u.variance),sa);
        sb=fmaf(w_b,sqrtf(v.variance),sb);
    }
    if(!(wa>0.f && wb>0.f)){fail(p,kFrameStatusInvalidLayout);return;}
    const auto albedo=a.linear_albedo_rgb_diffuse_factor;
    const float av=from_half(ch==0u?albedo.x:(ch==1u?albedo.y:albedo.z));
    const float modulation=sqrtf(fmaf(av,av,.01f));
    const float x=modulation*ma/wa,y=modulation*mb/wb;
    const float exposed=baseline<=.04045f?baseline/12.92f:__powf((baseline+.055f)/1.055f,2.4f);
    const float linear=-log1pf(-fminf(exposed,.99999994f));
    const float coherent=fmaxf((x-linear)*(y-linear),0.f);
    const float fa=float(na)/float(count),fb=1.f-fa;
    const float sa_mean=fa*modulation*sa/wa,sb_mean=fb*modulation*sb/wb;
    const float uncertainty=fmaf(sa_mean,sa_mean,sb_mean*sb_mean);
    const float gain=coherent/fmaxf(coherent+uncertainty,1.e-20f);
    const float target=fmaf(fb,y-x,x);
    const float desired=gain>0.f?denoise_native::canonical_physical_transfer(fmaxf(0.f,fmaf(gain,target-linear,linear))):baseline;
    const float2 previous=history[i];
    const float allowed=fabsf(baseline-previous.x);
    const float value=fminf(previous.y+allowed,fmaxf(previous.y-allowed,desired));
    if(!isfinite(value)){fail(p);return;}
#ifdef LFS_ENCODED_PLAIN_CURRENT
    image[slot]=accumulate_held_panel(p,i,value,x,y,uncertainty);
#else
    image[slot]=value;
#endif
    history[i]=make_float2(baseline,value);
}
}

extern "C" __global__ void lfs_collect_held_es(EncodedAtrousNativeParams p) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<p.count)collect_held_measurement(p,i);
}
#ifdef LFS_ENCODED_PLAIN_CURRENT
constexpr int kHeldSamples=9;
#else
constexpr int kHeldSamples=25;
#endif
extern "C" __global__ void lfs_present_held_es_2k(EncodedAtrousNativeParams p) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<p.count)present_held_measurement<4608,kHeldSamples>(p,i);
}
extern "C" __global__ void lfs_present_held_es_4k(EncodedAtrousNativeParams p) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<p.count)present_held_measurement<11520,kHeldSamples>(p,i);
}
