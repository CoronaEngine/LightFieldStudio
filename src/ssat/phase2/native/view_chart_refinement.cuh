// Intra-ViewId reconstruction of measured RGB. The table is calibration-only.
// This is not a secondary-path identity/virtual-address model.
#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
__device__ __forceinline__ bool field_same_physical_ray(const EncodedAtrousNativeParams &,unsigned,
        const denoise_native::PackedSurfaceObservationV1 &);
#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
__device__ __forceinline__ Vec4 repeated_es_measurement(const EncodedAtrousNativeParams &p,unsigned i) {
    const auto *state=reinterpret_cast<const unsigned*>(p.regular_repeat_state);
    const unsigned rg=state[3ull*p.count+i],bs=state[4ull*p.count+i];
    const float n=__uint_as_float(state[2ull*p.count+i]);
    const float sigma=from_half(static_cast<unsigned short>(bs>>16u));
    const float variance=sigma*sigma;
    return {from_half(static_cast<unsigned short>(rg)),from_half(static_cast<unsigned short>(rg>>16u)),
            from_half(static_cast<unsigned short>(bs)),n>1.f?variance/(n-1.f):-variance};
}
__device__ __forceinline__ void collect_repeated_es(const EncodedAtrousNativeParams &p,unsigned i) {
    auto *state=reinterpret_cast<float*>(p.regular_repeat_state);
    auto *packed=reinterpret_cast<unsigned*>(p.regular_repeat_state);
    const auto observation=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[i];
    const auto raw=reinterpret_cast<const float4*>(p.radiance)[i];
    const unsigned ch=i%3u;
    const float sample=ch==0u?raw.x:(ch==1u?raw.y:raw.z);
    const auto ab=observation.linear_albedo_rgb_diffuse_factor;
    const float ax=from_half(ab.x),ay=from_half(ab.y),az=from_half(ab.z);
    // A physical ray at each site supplies all three raw components. Carry
    // their linear demodulated mean from independent same-pose exposures.
    const Vec4 es{from_half(to_half(fminf(500.f,raw.x/sqrtf(fmaf(ax,ax,.01f))))),
        from_half(to_half(fminf(500.f,raw.y/sqrtf(fmaf(ay,ay,.01f))))),
        from_half(to_half(fminf(500.f,raw.z/sqrtf(fmaf(az,az,.01f))))),1.e-4f};
    float old_n=p.hold_same==1u?0.f:state[2ull*p.count+i];
    if(!(old_n>=1.f && old_n<=65535.f) || !field_same_physical_ray(p,i,observation))old_n=0.f;
    float mean=sample,raw_squares=0.f,n=1.f;
    Vec3 linear{es.x,es.y,es.z};float sigma=sqrtf(fabsf(es.w));
    if(old_n>0.f) {
        n=fminf(old_n+1.f,65535.f);
        const float inv=1.f/n,beta=1.f-inv;
        const float last=state[i],innovation=sample-last;
        mean=fmaf(innovation,inv,last);
        raw_squares=fmaxf(fmaf(innovation,sample-mean,state[p.count+i]*(old_n==65535.f?beta:1.f)),0.f);
        const unsigned rg=packed[3ull*p.count+i],bs=packed[4ull*p.count+i];
        const Vec3 previous{from_half(static_cast<unsigned short>(rg)),from_half(static_cast<unsigned short>(rg>>16u)),from_half(static_cast<unsigned short>(bs))};
        const float prior_sigma=old_n==1.f?0.f:from_half(static_cast<unsigned short>(bs>>16u));
        const float diff=luminance(linear)-luminance(previous);
        linear=previous+(linear-previous)*inv;
        sigma=sqrtf(fmaxf(beta*fmaf(inv,diff*diff,prior_sigma*prior_sigma),0.f));
    }
    state[i]=mean;state[p.count+i]=raw_squares;state[2ull*p.count+i]=n;
    packed[3ull*p.count+i]=unsigned(to_half(linear.x))|(unsigned(to_half(linear.y))<<16u);
    packed[4ull*p.count+i]=unsigned(to_half(linear.z))|(unsigned(to_half(sigma))<<16u);
}
#endif
struct ViewChartGuide36 {Vec3 p,n;unsigned instance;unsigned short a0,a1,a2,metadata;};
static_assert(sizeof(ViewChartGuide36)==36u);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
// The 36-byte chart guide has three direction words. Keep the geometric
// normal and reflected BSDF direction at signed 16-bit precision in those
// same words. A zero reflected vector marks an unavailable shading normal.
__device__ __forceinline__ float chart_directions(float geometric,float reflection) {
    const unsigned lo=unsigned(__float2int_rn(fminf(1.f,fmaxf(-1.f,geometric))*32767.f))&65535u;
    const unsigned hi=unsigned(__float2int_rn(fminf(1.f,fmaxf(-1.f,reflection))*32767.f));
    return __uint_as_float(lo|(hi<<16u));
}
template<bool Reflection>
__device__ __forceinline__ float chart_direction_component(float packed) {
    const unsigned bits=__float_as_uint(packed);
    return float(static_cast<short>(Reflection?bits>>16u:bits))*(1.f/32767.f);
}
template<bool Reflection>
__device__ __forceinline__ Vec3 chart_direction(Vec3 v) {
    return {chart_direction_component<Reflection>(v.x),chart_direction_component<Reflection>(v.y),
            chart_direction_component<Reflection>(v.z)};
}
#endif
__device__ __forceinline__ Vec3 view_chart_normal(const EncodedAtrousNativeParams &,const ViewChartGuide36 &g,const Vec3 *){return g.n;}

template<int Width>
__device__ __forceinline__ ViewChartGuide36 direct_chart_guide(const EncodedAtrousNativeParams& p,unsigned i,unsigned view) {
 const auto o=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[i];
 const auto contract=ssat_native::validate_packed_surface_observation_v1(o);
 ViewChartGuide36 g{};
 if(contract.status){fail(p);return g;}
 const unsigned valid=contract.domain==ssat_native::kPackedSurfaceDomainSurface?1u:0u;
 const unsigned cls=valid?contract.surface_flags:0u;
 g.instance=valid?unsigned(o.surface_key>>32u):~0u;
 g.metadata=static_cast<unsigned short>(valid|(cls<<2u));
 g.a0=o.linear_albedo_rgb_diffuse_factor.x;g.a1=o.linear_albedo_rgb_diffuse_factor.y;g.a2=o.linear_albedo_rgb_diffuse_factor.z;
 if(valid&&(cls==kSurfaceGlossy||cls==kSurfaceNearSpecular)) {
   const unsigned x=i%Width,y=i/Width;
   const float u=reinterpret_cast<const float*>(p.static_view_x)[view],iv=reinterpret_cast<const float*>(p.static_inverse_ray_length)[i];
   const Vec3 d{(reinterpret_cast<const float*>(p.static_focal_x)[x]-u)*iv,reinterpret_cast<const float*>(p.static_focal_y)[y]*iv,p.focal_distance*iv};
   g.p=Vec3{u,0.f,0.f}+d*o.hit_distance;
   g.n=local_normal(p,unpack_normal(o.packed_geometric_normal_oct16));
   if(!finite3(g.p)||!finite3(g.n))fail(p);
 }
 return g;
}
// Native fast division uses MUFU.RCP and a rounded multiply for normal
// positive denominators. These bounds come from the measured half signal and
// the explicitly positive reconstruction variance floor.
__device__ __forceinline__ float regular_chart_reciprocal(float denominator) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(denominator));
    return r;
}
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
struct ChartSlotStream {
    const unsigned *words;
    size_t stride;
    unsigned low, high, shift;

    __device__ __forceinline__ unsigned take(bool last=false) {
        const unsigned slot=__funnelshift_r(low,high,shift)&511u;
        shift+=9u;
        if(shift>=32u && !last) {
            shift-=32u;low=high;words+=stride;high=words[stride];
        }
        return slot;
    }
};

#endif
#include "view_chart_estimate.cuh"
#if defined(LFS_ENCODED_PLAIN_CURRENT) || defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
#define LFS_CHART_USE_SHARED_INVERSE
#include "view_chart_estimate.cuh"
#undef LFS_CHART_USE_SHARED_INVERSE
#endif
