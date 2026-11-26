#include "lf_history_query.cuh"
#include "ssat/phase3/native/early_measurement_params.h"
#include "ssat/phase3/native/rgb_measurement_query.cuh"

namespace {
using MrParams=lfs::denoise::mr_native::Params;
using Observation=denoise_native::PackedSurfaceObservationV1;
__device__ float mr_luma(float4 x) { return .2126f*x.x+.7152f*x.y+.0722f*x.z; }
__device__ bool mr_index(const MrParams &p,unsigned &i,unsigned &x,unsigned &y) {
    x=blockIdx.x*blockDim.x+threadIdx.x;y=blockIdx.y*blockDim.y+threadIdx.y;
    i=y*p.geometry.output_width*3u+x;
    return x<p.geometry.output_width*3u && y<p.geometry.output_height;
}
__device__ bool mr_finite(float4 v) {return isfinite(v.x)&&isfinite(v.y)&&isfinite(v.z)&&isfinite(v.w);}

}

template<bool Fast>
__device__ __forceinline__ float measured_expm1(float x) {
    if constexpr(!Fast)return expm1f(x);
    if(fabsf(x)<.0625f)
        return x*fmaf(x,fmaf(x,fmaf(x,fmaf(x,1.f/120.f,1.f/24.f),1.f/6.f),.5f),1.f);
    float v;const float a=x*1.4426950408889634f;
    asm("ex2.approx.ftz.f32 %0, %1;":"=f"(v):"f"(a));
    return v-1.f;
}
template<bool Fast>
__device__ __forceinline__ float measured_log1p(float x) {
    if constexpr(!Fast)return log1pf(x);
    if(x<.0625f) {
        const float t=x/(2.f+x),s=t*t;
        return 2.f*t*fmaf(s,fmaf(s,1.f/5.f,1.f/3.f),1.f);
    }
    const float a=1.f+x;float v;
    asm("lg2.approx.ftz.f32 %0, %1;":"=f"(v):"f"(a));
    return v*.6931471805599453f;
}
template<bool Fast>
__device__ __forceinline__ float measured_root(float x) {
    if constexpr(!Fast)return sqrtf(x);
    float v;asm("sqrt.approx.ftz.f32 %0, %1;":"=f"(v):"f"(x));return v;
}
template<bool Fast=false>
__device__ float4 mr_encode_glossy(float4 v,float4 modulation) {
    v.x*=modulation.x;v.y*=modulation.y;v.z*=modulation.z;
    const float l=mr_luma(v),scale=l>0.f?measured_log1p<Fast>(l)/l:1.f;
    return make_float4(v.x*scale,v.y*scale,v.z*scale,v.w);
}
template<bool Fast=false>
__device__ float4 mr_decode_glossy(float4 v,float4 modulation) {
    const float l=mr_luma(v),scale=l>0.f?measured_expm1<Fast>(l)/l:1.f;
    return make_float4(v.x*scale/modulation.x,v.y*scale/modulation.y,v.z*scale/modulation.z,v.w);
}


// Convert between the history's log-luminance variance and linear demodulated variance.
// Spatial reconstruction consumes the linear-domain uncertainty of the mean.
template<bool Fast=false>
__device__ __forceinline__ float mr_decode_jacobian(float4 encoded, float4 linear) {
    const float l = mr_luma(encoded);
    const float physical = measured_expm1<Fast>(l);
    return (Fast?1.f+physical:expf(l)) * (physical > 0.f ? mr_luma(linear) / physical : 1.f);
}

__device__ __forceinline__ bool update_stationary_ray(
        const MrParams &p, unsigned index, const native::CurrentTemporalInput &current,
        const Observation &observation, float4 sample, bool logarithmic, float4 modulation,
        float4 &mean, float &squared_weights, unsigned &age) {
    const auto &g = p.geometry;
    if (!g.history_valid || !g.static_layout) return false;
    if (surface_valid(current.metadata)) {
        const auto *rigid = rigid_motion_record(g, unsigned(current.surface_key >> 32u));
        if (!rigid || (rigid->metadata.x & (kRigidMotionValid | kRigidMotionIdentity)) !=
                      (kRigidMotionValid | kRigidMotionIdentity)) return false;
    }
    const auto previous = prior_temporal_sample(g,index);
    const auto a = observation.linear_albedo_rgb_diffuse_factor;
    const auto b = previous.signal_physical_mean_rms;
    if (previous.surface_key != current.surface_key ||
        previous.depth != current.depth || previous.packed_normal != current.packed_normal ||
        surface_valid(previous.metadata) != surface_valid(current.metadata) ||
        surface_class(previous.metadata) != surface_class(current.metadata) ||
        signal_domain(previous.metadata) != signal_domain(current.metadata) ||
        bool((previous.metadata >> 30u) & 1u) != logarithmic ||
        a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) return false;

    const auto stored = pointer<const native::Half4>(p.measurement_read)[index];
    const float sigma = half_to_float(stored.w);
    const float4 encoded = make_float4(half_to_float(stored.x),half_to_float(stored.y),
                                       half_to_float(stored.z),sigma*sigma);
    if (!mr_finite(encoded)) {
        mark_frame_status(g, native::kPhase3FrameStatusNonfinite);
        return false;
    }
    const float4 old = logarithmic ? mr_decode_glossy(encoded, modulation) : encoded;
    const float last_j = logarithmic ? mr_decode_jacobian(encoded, old) : 1.f;
    age = min(history_age(previous.metadata) + 1u, 32u);
    const float alpha = 1.f / float(age);
    const float beta = 1.f - alpha;
    const float innovation = mr_luma(sample) - mr_luma(old);
    float4 linear = make_float4(fmaf(alpha, sample.x - old.x, old.x),
                                fmaf(alpha, sample.y - old.y, old.y),
                                fmaf(alpha, sample.z - old.z, old.z),
                                beta * (encoded.w * last_j * last_j + alpha * innovation * innovation));
    mean = logarithmic ? mr_encode_glossy(linear, modulation) : linear;
    if (logarithmic) {
        const float j = mr_decode_jacobian(mean, linear);
        mean.w = linear.w / (j * j);
    }
    squared_weights = alpha * alpha + beta * beta * half_to_float(previous.sigma_albedo.x);
    return true;
}

__device__ __forceinline__ native::Half4 stationary_input(const MrParams &p, unsigned index, const Observation &observation) {
    const auto raw = pointer<const float4>(p.stationary_radiance)[index];
    const auto a = observation.linear_albedo_rgb_diffuse_factor;
    const bool surface = (observation.metadata & 3u) == 1u;
    if (!(isfinite(raw.x) && isfinite(raw.y) && isfinite(raw.z)))
        mark_frame_status(p.geometry, native::kPhase3FrameStatusNonfinite);
    const float x = raw.x / native_current::encoded_atrous_modulation(surface, half_to_float(a.x));
    const float y = raw.y / native_current::encoded_atrous_modulation(surface, half_to_float(a.y));
    const float z = raw.z / native_current::encoded_atrous_modulation(surface, half_to_float(a.z));
    return {float_to_half(clampf(x,0.f,500.f)),float_to_half(clampf(y,0.f,500.f)),
            float_to_half(clampf(z,0.f,500.f)),float_to_half(1.e-4f)};
}

#include "diffuse_reprojection_pair.cuh"

__device__ __forceinline__ float previous_panel_footprint(
        const MrParams &p,const QueryState<0,false> &state,const Observation &a) {
    const auto &g=p.geometry;
    const float3 point=state.predicted_previous_position_local;
    if(!state.query_valid || state.exact_identity_query || !(point.z>0.f))return 0.f;
    const float u=view_position(g,state.current_view);
    const float scale=point.z*g.inverse_focal_distance;
    const float fx=fmaf(point.x-u,1.f/scale,u);
    const float fy=point.y/scale;
    const float dx=(fmaf(fx+.5f*g.focal_width,g.inverse_focal_lattice_width,-.5f)-float(state.encoded_x))/3.f;
    const float dy=fmaf(.5f*g.focal_height-fy,g.inverse_focal_lattice_height,-.5f)-float(state.encoded_y);
    // Both lanes share the central nine real sites of this calibrated aperture.
    // Its disk has area 9*views/3 native pixels and per-axis second moment R^2/4.
    const float exponent=fmaf(dx,dx,dy*dy)*(3.14159265f/(3.f*float(g.view_count)));
    // All later confidence factors are <=1. exp(-18) lies below the half
    // rounding boundary 2^-25, so this sample cannot affect the actual panel.
    if(exponent>18.f)return 0.f;
    const float overlap=__expf(-exponent);
    const auto last=pointer<const native::TemporalSample>(g.history_read)[state.index];
    if(last.surface_key!=a.surface_key ||
       !surface_valid(last.metadata) || surface_class(last.metadata)!=state.current_class || !(last.depth>0.f))return 0.f;
    const auto alb=a.linear_albedo_rgb_diffuse_factor,old=last.signal_physical_mean_rms;
    // At different positions in the same glossy material, the directional
    // albedo can legitimately differ. Use the existing geometric donor test
    // and the continuous albedo kernel of the actual spatial reconstruction.
    const float3 a0=make_float3(half_to_float(alb.x),half_to_float(alb.y),half_to_float(alb.z));
    const float3 a1=make_float3(half_to_float(old.x),half_to_float(old.y),half_to_float(old.z));
    const float3 da=subtract(a0,a1);
    const float tolerance=g.maximum_albedo_relative_error;
    if(fabsf(da.x)>tolerance*fmaxf(fmaxf(a0.x,a1.x),.05f) ||
       fabsf(da.y)>tolerance*fmaxf(fmaxf(a0.y,a1.y),.05f) ||
       fabsf(da.z)>tolerance*fmaxf(fmaxf(a0.z,a1.z),.05f))return 0.f;
    const float reflectance=__expf(-4.f*sqrtf(dot3(da,da)));
    const float3 n=transform_vector(g.previous_world_to_local,unpack_normal(last.packed_normal));
    if(dot3(n,state.predicted_previous_normal_local)<g.minimum_normal_similarity)return 0.f;
    const auto ray=analytical_local_ray(g,state.encoded_x,state.encoded_y,state.current_view);
    const auto difference=subtract(add(ray.origin,multiply(ray.direction,last.depth)),point);
    const float plane=fabsf(dot3(difference,n));
    if(plane>g.base_position_tolerance*fmaxf(scale,1.f))return 0.f;
    const auto desired=reflect_direction(state.query_direction_previous_local,state.predicted_previous_normal_local);
    const auto actual=reflect_direction(ray.direction,n);
    const float c=clampf(dot3(desired,actual),-1.f,1.f);
    const float r=float((a.metadata>>9u)&255u)/255.f;
    const float alpha=fmaxf(r*r,.5f*g.array_angle_radians/fmaxf(float(g.view_count-1u),1.f));
    const float angle=(1.f-c)/fmaxf(1.f+c,1.e-8f);
    const float weight=reflectance*overlap*alpha*alpha/fmaxf(fmaf(alpha,alpha,angle),1.e-20f);
    // Two samples within one native pixel's angular support still observe the
    // same outgoing direction. The rigidly transported incident ray supplies
    // this test even when the projected surface point lies at a different site.
    const float3 turn=subtract(state.query_direction_previous_local,ray.direction);
    const float pixel_angle=fmaxf(3.f*g.focal_lattice_width,g.focal_lattice_height)*g.inverse_focal_distance;
    const bool same_direction=state.current_class==kSurfaceNearSpecular && dot3(turn,turn)<=pixel_angle*pixel_angle;
    return same_direction?-weight:weight;
}


#include "material_vertex_frame.cuh"
template<bool GlossyLog=false, bool Stationary=false, bool CertifiedRegular=false, bool VertexFrame=false, bool PrevalidatedRgb=false>
__device__ void early_measurement_body(MrParams p) {
    unsigned i,x,y;
    const unsigned arriving=__activemask();
    const bool in_bounds=mr_index(p,i,x,y);
    const unsigned live_sites=__ballot_sync(arriving,in_bounds);
    if(!in_bounds)return;
    const auto &g=p.geometry;
    if(p.history_storage!=lfs::denoise::mr_native::kHistoryFp16RgbRms) {
        mark_frame_status(g,native::kPhase3FrameStatusInvalidReconstruction);return;
    }
    const Observation observation=pointer<const Observation>(g.current_packed_observation)[i];
    native::Half4 input=Stationary ? stationary_input(p, i, observation) : pointer<const native::Half4>(g.current_phase2)[i];
    const bool linear_input=CertifiedRegular && GlossyLog && (input.w&0x8000u);
    if constexpr(CertifiedRegular && GlossyLog)input.w&=0x7fffu;
    float4 sample=make_float4(half_to_float(input.x),half_to_float(input.y),half_to_float(input.z),0.f);
    const float4 linear_sample=sample;sample.w=0.f;
    if(!mr_finite(sample)||sample.x<0.f||sample.y<0.f||sample.z<0.f)
        mark_frame_status(g,native::kPhase3FrameStatusNonfinite);
    const auto current=compact_current(g,i,x%3u,input);
    const unsigned cls=surface_class(current.metadata);
    if constexpr(GlossyLog && !CertifiedRegular){
        // One warp is one aligned32-site row. All lanes share a128x32 group.
        const bool needs_original=!(surface_valid(current.metadata)==1u&&(cls==kSurfaceGlossy||cls==kSurfaceNearSpecular));
        const unsigned active=__activemask(),mask=__ballot_sync(active,needs_original);
        if(mask&&(threadIdx.x&31u)==0u){
            const unsigned bits=((mask&0x11111111u)?1u:0u)|((mask&0x22222222u)?2u:0u)|((mask&0x44444444u)?4u:0u)|((mask&0x88888888u)?8u:0u);
            const unsigned columns=(g.output_width*3u+127u)/128u,group=(y/32u)*columns+x/128u;
            atomicOr(pointer<unsigned>(g.frame_status)+1u+group,bits<<((y&3u)*4u));
        }
    }
    const bool arithmetic_glossy=linear_input &&
        surface_valid(current.metadata)==1u && cls==kSurfaceGlossy;
    const bool logarithmic=GlossyLog && !arithmetic_glossy && surface_valid(current.metadata)==1u && cls==kSurfaceGlossy;
    float4 modulation=make_float4(1,1,1,1);
    if(logarithmic) {
        const auto a=observation.linear_albedo_rgb_diffuse_factor;
        modulation=make_float4(lfs::denoise::physical_lf::native_current::encoded_atrous_modulation(1u,half_to_float(a.x)),
            lfs::denoise::physical_lf::native_current::encoded_atrous_modulation(1u,half_to_float(a.y)),lfs::denoise::physical_lf::native_current::encoded_atrous_modulation(1u,half_to_float(a.z)),1.f);
        sample=mr_encode_glossy<CertifiedRegular>(sample,modulation);
    }
    if constexpr(VertexFrame) {
        unsigned material_word=0xffffffffu;
        // The current spatial reconstruction alone consumes this map. The
        // logarithmic finite lobe uses its calibrated broad angular integral.
        if(surface_valid(current.metadata) && (cls==kSurfaceNearSpecular || (cls==kSurfaceGlossy && !logarithmic)))
            material_word=bsdf_vertex_normal(p,i,observation);
        pointer<unsigned>(p.current_material_normals)[i]=material_word;
    }
    const float luminance=mr_luma(sample);
    float exposure_variance=fabsf(half_to_float(input.w));
    if constexpr(GlossyLog && CertifiedRegular) {
        if(logarithmic) {
            // ES1 reports variance of demodulated linear luminance. Compare it to
            // stored radial-log residuals after propagating through this receiver's
            // actual RGB modulation and exposure transform.
            const float lr=mr_luma(linear_sample);
            const float lp=linear_sample.x*(.2126f*modulation.x)+linear_sample.y*(.7152f*modulation.y)+
                           linear_sample.z*(.0722f*modulation.z);
            const float slope=(lr>0.f?lp/lr:mr_luma(modulation))/(1.f+lp);
            exposure_variance*=slope*slope;
        }
    }
    const bool delta_reflection=((cls&kSurfaceNearSpecular)!=0u && (cls&(kSurfaceDiffuse|kSurfaceGlossy|kSurfaceTransmission))==0u) || logarithmic || arithmetic_glossy;
    float4 mean=sample;float s2=1.f;unsigned age=1u;
    float current_innovation=0.f,panel_overlap=0.f;
    unsigned previous_display=0u;
    if constexpr(CertifiedRegular) {
        if(p.previous_panel_samples && (cls==kSurfaceGlossy || cls==kSurfaceNearSpecular))
            previous_display=pointer<const unsigned>(p.previous_panel_samples)[i]&65535u;
    }
    if constexpr (Stationary) {
        update_stationary_ray(p, i, current, observation, linear_sample, logarithmic, modulation, mean, s2, age);
    } else if(g.history_valid && !(CertifiedRegular && p.deterministic_motion_environment && (observation.metadata&3u)==2u)) {
        // The LF ray contains position AND direction. Use its original physical
        // query, without treating the current random continuation as a target.
        // This is not ordinary two-dimensional primary-surface reprojection.
        auto state=make_query_state<0,false,false,true,false,CertifiedRegular>(g,i,current,true);
        mr_surface_query<CertifiedRegular>(g,state);
        if constexpr(CertifiedRegular) {
            if(previous_display&0x8000u)
                panel_overlap=previous_panel_footprint(p,state,observation);
        }

        std::uint64_t prior_optics=0u;
        InterfaceFootprint receiver_interface{};
        float3 optical_normal=make_float3(0.f,0.f,0.f);
        if constexpr(CertifiedRegular) {
            if(state.query_valid && !state.exact_identity_query && p.current_material_normals && p.previous_material_normals &&
               (cls==kSurfaceGlossy || cls==kSurfaceNearSpecular)) {
                const unsigned normal=pointer<const unsigned>(p.current_material_normals)[i];
                const auto *motion=rigid_motion_record(g,unsigned(current.surface_key>>32u));
                if(normal!=0xffffffffu && motion && (motion->metadata.x&kRigidMotionValid)!=0u) {
                    const float3 previous_world=transform_motion_vector(*motion,unpack_normal(normal));
                    optical_normal=normalize3(transform_vector(g.previous_world_to_local,previous_world));
                    prior_optics=p.previous_material_normals;
                    if constexpr(GlossyLog) {
                        const auto *ns=pointer<const unsigned>(p.current_material_normals);
                        const float eta=__uint_as_float(ns[size_t(g.output_width)*3u*g.output_height+
                                                         unsigned(current.surface_key>>32u)]);
                        if(eta>0.f) {
                            const float rough=float((observation.metadata>>9u)&255u)/255.f;
                            const float width=fmaxf(rough*rough,.5f*g.array_angle_radians/fmaxf(float(g.view_count-1u),1.f));
                            receiver_interface=interface_footprint(state.query_direction_previous_local,optical_normal,
                                state.predicted_previous_normal_local,eta,width*width);
                        }
                    }
                }
            }
        }
        FixedPhaseTaps taps{};
        CompactRgbFootprint compact{};
        if(state.query_valid && state.exact_identity_query) {
            if constexpr(PrevalidatedRgb) {
                compact.set_corner(int(x),int(y));
                compact.views=state.current_view;compact.w0=1.f;
            } else {
                taps.tap0.index=i;taps.tap0.site=pack_phase_site(x/3u,y,state.current_view);taps.tap0.weight=1.f;
            }
        } else if constexpr (CertifiedRegular) {
            taps = project_rgb_manifold_taps<4,true,true,PrevalidatedRgb>(g, state,
                {p.rgb_b0x, p.rgb_b0y, p.rgb_b1x, p.rgb_b1y, p.rgb_basis_valid != 0u},&compact);
        } else {
            taps = project_rgb_manifold_taps<4>(g, state);
        }
        const PhaseTap candidates[4]={taps.tap0,taps.tap1,taps.tap2,taps.tap3};
        const bool bounded=PrevalidatedRgb?compact.common_scale():CertifiedRegular &&
            (taps.tap0.weight==0.f || taps.tap0.weight>=0x1p-32f) &&
            (taps.tap1.weight==0.f || taps.tap1.weight>=0x1p-32f) &&
            (taps.tap2.weight==0.f || taps.tap2.weight>=0x1p-32f) &&
            (taps.tap3.weight==0.f || taps.tap3.weight>=0x1p-32f);
        const bool process_history=CertifiedRegular && !state.exact_identity_query &&
            (cls==kSurfaceGlossy || cls==kSurfaceNearSpecular || (GlossyLog && cls==kSurfaceDiffuse));
        const float rough=float((observation.metadata>>9u)&255u)/255.f;
        const float half_pitch=.5f*g.array_angle_radians/fmaxf(float(g.view_count-1u),1.f);
        const float width=fmaxf(rough*rough,half_pitch),width2=width*width;
        const auto current_albedo=observation.linear_albedo_rgb_diffuse_factor;
        const float da[3]={half_to_float(current_albedo.x),half_to_float(current_albedo.y),half_to_float(current_albedo.z)};

        float4 history_mean=make_float4(0,0,0,0);float mass=0.f,sqrt_s2=0.f;unsigned retained_age=32u;
        float geometric_mass=0.f;float observed_mass=0.f;
        int weight_exponent=0;bool weight_scale_initialized=false;
        for(unsigned k=0;k<4u;++k) {
            const auto tap=PrevalidatedRgb?compact.take(p,k):candidates[k];if(!(tap.weight>0.f)||tap.index>=g.output_width*3u*g.output_height)continue;
            native::TemporalSample previous{};
            // A full RGB sample may come from any carrier. Validate the actual ray
            // and compare matching albedo components, not two different colors.
            auto evaluation_state=state;
            evaluation_state.channel=tap.index%3u;
            evaluation_state.current_albedo=evaluation_state.channel==0u?da[0]:(evaluation_state.channel==1u?da[1]:da[2]);
            evaluation_state.current_domain=state.current_valid!=0u &&
                (state.current_class&kSurfaceDiffuse)!=0u && (state.current_class&kResponsiveSurfaceMask)==0u &&
                half_to_float(current_albedo.w)>=.5f && evaluation_state.current_albedo>=.05f ? 1u:0u;
            // M stores aggregate reconstructed radiance, not the sampled path's
            // conditional contribution. Keep virtual query and primary validation,
            // but do not condition aggregate RGB on a single continuation's
            // secondary identity or hit distance, including near-specular RGB.
            if(delta_reflection)evaluation_state.responsive_path=false;
            const auto evaluated=evaluate_candidate<0,false,true,CertifiedRegular,CertifiedRegular,CertifiedRegular,CertifiedRegular && GlossyLog>(
                g,evaluation_state,tap.index,tap.site,previous,prior_optics,optical_normal,receiver_interface);
            const bool previous_log=((previous.metadata>>30u)&1u)!=0u;
            if(!evaluated.accepted || (!(CertifiedRegular && GlossyLog && cls==kSurfaceGlossy) && previous_log!=logarithmic))continue;
            // The reused 32B geometry record also carries all three albedos.
            const auto a=current_albedo,b=previous.signal_physical_mean_rms;
            const float db[3]={half_to_float(b.x),half_to_float(b.y),half_to_float(b.z)};
            bool material_ok=true;
            if(a.x!=b.x || a.y!=b.y || a.z!=b.z)
            for(unsigned c=0;c<3u;++c)material_ok=material_ok && fabsf(da[c]-db[c])<=g.maximum_albedo_relative_error*fmaxf(fmaxf(fabsf(da[c]),fabsf(db[c])),.05f);
            if(!material_ok)continue;
            const uint2 bits=pointer<const uint2>(p.measurement_read)[tap.index];
            const native::Half4 stored{static_cast<std::uint16_t>(bits.x),static_cast<std::uint16_t>(bits.x>>16u),
                static_cast<std::uint16_t>(bits.y),static_cast<std::uint16_t>(bits.y>>16u)};
            const float sigma=half_to_float(stored.w);
            float4 old=make_float4(half_to_float(stored.x),half_to_float(stored.y),half_to_float(stored.z),sigma*sigma);
            if(arithmetic_glossy && previous_log) {
                // The first moving frame can read a log-encoded static checkpoint.
                // Decode that donor with the albedo at its own physical surface.
                const float4 donor_modulation=make_float4(measured_root<CertifiedRegular>(fmaf(db[0],db[0],.01f)),
                    measured_root<CertifiedRegular>(fmaf(db[1],db[1],.01f)),measured_root<CertifiedRegular>(fmaf(db[2],db[2],.01f)),1.f);
                const float4 linear=mr_decode_glossy<CertifiedRegular>(old,donor_modulation);
                const float jacobian=mr_decode_jacobian<CertifiedRegular>(old,linear);
                old=make_float4(linear.x,linear.y,linear.z,old.w*jacobian*jacobian);
            }
            if(CertifiedRegular && logarithmic && !previous_log) {
                const float4 donor_modulation=make_float4(measured_root<CertifiedRegular>(fmaf(db[0],db[0],.01f)),
                    measured_root<CertifiedRegular>(fmaf(db[1],db[1],.01f)),measured_root<CertifiedRegular>(fmaf(db[2],db[2],.01f)),1.f);
                const float4 linear=old;
                old=mr_encode_glossy<CertifiedRegular>(linear,donor_modulation);
                const float j=mr_decode_jacobian<CertifiedRegular>(old,linear);old.w=linear.w/(j*j);
            }
            if constexpr(CertifiedRegular && GlossyLog) {
                if(cls==kSurfaceDiffuse && (a.x!=b.x || a.y!=b.y || a.z!=b.z)) {
                    // Lambertian paths measure albedo times irradiance. M stores
                    // radiance divided by sqrt(a^2+.01); the remaining albedo
                    // response is therefore a/sqrt(a^2+.01), including black.
                    float ratio[3];
                    bool informative=true;
                    #pragma unroll
                    for(unsigned c=0;c<3u;++c) {
                        if(da[c]==db[c])ratio[c]=1.f;
                        else if(db[c]>0.f)ratio[c]=(da[c]/db[c])*measured_root<CertifiedRegular>(fmaf(db[c],db[c],.01f))*rsqrtf(fmaf(da[c],da[c],.01f));
                        else {ratio[c]=0.f;informative=false;}
                    }
                    if(!informative)continue;
                    const float before=mr_luma(old);
                    old.x*=ratio[0];old.y*=ratio[1];old.z*=ratio[2];
                    const float scale=before>1.e-20f?mr_luma(old)/before:.2126f*ratio[0]+.7152f*ratio[1]+.0722f*ratio[2];
                    old.w*=scale*scale;
                }
            }
            if(!mr_finite(old)){mark_frame_status(g,native::kPhase3FrameStatusNonfinite);continue;}
            // A common binary scale preserves relative weights and keeps mass
            // O(1), even when all accepted Gaussian weights are subnormal.
            // Gaussian basis weights are in (0,1]. Clamp the exponent only to
            // the FP32 normal range; even a minimum subnormal becomes >=2^-23.
            // A 32-bit exponent offset keeps the minimum normal
            // production tap at one and preserves all four relative weights.
            // The original rebasing handles smaller tails.
            float w=tap.weight*0x1p32f;
            if(!bounded) {
                const int exponent=int(max((__float_as_uint(tap.weight)>>23u)&255u,1u))-127;
                if(!weight_scale_initialized){weight_exponent=exponent;weight_scale_initialized=true;}
                if(exponent>weight_exponent) {
                    const int shift=weight_exponent-exponent;
                    const float rescale=__uint_as_float(unsigned(127+shift)<<23u);
                    history_mean=make_float4(history_mean.x*rescale,history_mean.y*rescale,
                        history_mean.z*rescale,history_mean.w*rescale);
                    mass*=rescale;sqrt_s2*=rescale;geometric_mass*=rescale;observed_mass*=rescale;
                    weight_exponent=exponent;
                }
                w=tap.weight*__uint_as_float(unsigned(127-weight_exponent)<<23u);
            }
            if constexpr(CertifiedRegular) {
                float reliability=1.f;float count_reliability=1.f;
                if(process_history) {
                    const float c=evaluated.reflected_cosine;
                    const float scale=width2*fmaxf(1.f+c,1.e-8f);
                    const float overlap=cls==kSurfaceDiffuse?evaluated.diffuse_footprint:
                        (evaluated.dielectric_overlap>=0.f?evaluated.dielectric_overlap:scale/fmaxf(scale+(1.f-c),1.e-20f));
                    const float scatter=fmaxf(old.w+exposure_variance,1.e-6f);
                    const float innovation=luminance-mr_luma(old);
                    // With independent shot noise, the product of two innovations
                    // estimates persistent squared change without the self-noise of either sample.
                    // Their half squared difference estimates the local variance of one innovation.
                    const float z=innovation*rsqrtf(scatter);
                    const float prior_z=evaluated.previous_innovation;
                    const float difference=z-prior_z;
                    const float pair_noise=fmaxf(1.f,.5f*difference*difference);
                    const float change=scatter*(1.f-overlap*overlap)*
                        (fmaxf(z*prior_z,0.f)/pair_noise);
                    float process=change;
                    if constexpr(GlossyLog) {
                        if(cls==kSurfaceDiffuse) {
                            // The transported donor is a mean. Its standard error,
                            // rather than the scatter of a single path, sets the
                            // magnitude of a spatially confirmed drift. Its inverse
                            // variance is estimable only beyond two effective dof.
                            const float s2=fminf(1.f,fmaxf(0.f,half_to_float(previous.sigma_albedo.x)));
                            const float inverse=1.f/fmaxf(1.f-s2,1.e-20f);
                            const float information=fmaxf(1.f-3.f*s2,0.f)*inverse;
                            process*=measured_root<CertifiedRegular>(s2*inverse)*information;
                        }
                    }
                    reliability=scatter/(scatter+process);
                    count_reliability=reliability;
                    if constexpr(GlossyLog) {
                        if(evaluated.dielectric_overlap>=0.f) {
                            // First-interface directional affinity ranks donor colors,
                            // but does not establish additional independent exposures
                            // beyond the physical primary-ray registration.
                            // The same fitted history must support both the registered primary ray
                            // and the reflected/transmitted continuation. The angular displacement
                            // of an incoming ray is independent of either triangle's normal.
                            const float ci=clampf(evaluated.incident_cosine,-1.f,1.f);
                            const float incident_scale=width2*fmaxf(1.f+ci,1.e-8f);
                            const float incidence=incident_scale/fmaxf(incident_scale+(1.f-ci),1.e-20f);
                            const float raw_overlap=fminf(incidence,evaluated.dielectric_overlap);
                            const float stationary=scatter*(1.f-raw_overlap*raw_overlap)*
                                (fmaxf(z*prior_z,0.f)/pair_noise);
                            count_reliability=scatter/(scatter+stationary);
                        }
                    }
                }
                // A mapped footprint can cover distinct reflected fields on the same surface.
                // Carry each donor's measured compatibility into its RGB and its variance weight.
                geometric_mass+=w;observed_mass+=w*count_reliability;
                w*=reliability;
                if(!(w>0.f))continue;
            }
            history_mean.x+=w*old.x;history_mean.y+=w*old.y;history_mean.z+=w*old.z;history_mean.w+=w*old.w;
            sqrt_s2+=w*measured_root<CertifiedRegular>(fmaxf(half_to_float(previous.sigma_albedo.x),0.f));
            mass+=w;retained_age=min(retained_age,history_age(previous.metadata));
        }
        if(mass>0.f) {
            age=min(retained_age+1u,32u);
            float alpha=1.f/float(age);
            if constexpr(CertifiedRegular) {
                if(!state.exact_identity_query && (cls==kSurfaceGlossy || cls==kSurfaceNearSpecular || (GlossyLog && cls==kSurfaceDiffuse))) {
                    const float support=fminf(fmaxf(observed_mass/geometric_mass,0.f),1.f)*float(age-1u);
                    alpha=1.f/(1.f+support);
                    if(cls!=kSurfaceDiffuse)age=1u+__float2uint_rn(support);
                }
            }
            const float beta=1.f-alpha;
            const float inv=1.f/mass;

            const float innovation=luminance-mr_luma(history_mean)*inv;
            if constexpr(CertifiedRegular) {
                if(cls==kSurfaceGlossy || cls==kSurfaceNearSpecular || (GlossyLog && cls==kSurfaceDiffuse)) {
                    const float noise=fmaxf(history_mean.w*inv+exposure_variance,1.e-6f);
                    current_innovation=clampf(innovation*rsqrtf(noise),-65504.f,65504.f);
                }
            }
            mean=make_float4(alpha*sample.x+beta*history_mean.x*inv,alpha*sample.y+beta*history_mean.y*inv,alpha*sample.z+beta*history_mean.z*inv,beta*history_mean.w*inv+alpha*beta*innovation*innovation);
            s2=alpha*alpha+beta*beta*(sqrt_s2*inv)*(sqrt_s2*inv);
        } else if constexpr(CertifiedRegular && GlossyLog) {
            if(cls==kSurfaceDiffuse && surface_valid(current.metadata) && !g.static_layout)
                recover_diffuse_pair(p,state,observation,sample,mean,s2,age);
        }
    }

    if constexpr (GlossyLog && !CertifiedRegular) {
        if (p.held_angular_residuals) {
            auto *carry = pointer<unsigned short>(p.held_angular_residuals);
            unsigned short bits;
            if (p.committed_measurement) {
                const unsigned packed=g.history_valid && g.history_read ? pointer<const uint4>(g.history_read)[i].x : 0u;
                bits=static_cast<unsigned short>(((packed>>15u)&0x7fffu)|((packed>>16u)&0x8000u));
                carry[i]=bits;
            } else bits=carry[i];
            if (bits & 0x7fffu) {
                bool same = g.history_valid && g.static_layout && g.history_read &&
                            surface_valid(current.metadata) && age > 1u;
                if (same) {
                    const auto *motion = rigid_motion_record(g, unsigned(current.surface_key >> 32u));
                    const auto required = kRigidMotionValid | kRigidMotionIdentity;
                    same = motion && (motion->metadata.x & required) == required;
                }
                if (same) {
                    const auto prior = prior_temporal_sample(g,i);
                    const auto a = observation.linear_albedo_rgb_diffuse_factor;
                    const auto b = prior.signal_physical_mean_rms;
                    same = prior.surface_key == current.surface_key && prior.depth == current.depth &&
                           prior.packed_normal == current.packed_normal &&
                           surface_valid(prior.metadata) == surface_valid(current.metadata) &&
                           surface_class(prior.metadata) == surface_class(current.metadata) &&
                           signal_domain(prior.metadata) == signal_domain(current.metadata) &&
                           bool((prior.metadata >> 30u) & 1u) == logarithmic &&
                           a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
                }
                // The new M observation receives 1/age. Its complement is the
                // exact weight still carried by the previous ray measurement.
                carry[i] = same ? float_to_half(half_to_float(bits) * (1.f - 1.f / float(age))) : 0u;
            }
        }
    }

    static_assert(sizeof(native::TemporalSample)==32u);
    static_assert(offsetof(native::TemporalSample,surface_key)==0u && offsetof(native::TemporalSample,depth)==8u &&
                  offsetof(native::TemporalSample,packed_normal)==12u && offsetof(native::TemporalSample,metadata)==16u &&
                  offsetof(native::TemporalSample,sigma_albedo)==20u && offsetof(native::TemporalSample,signal_physical_mean_rms)==24u);
    if constexpr(GlossyLog && CertifiedRegular) {
        const bool spatial=!(surface_valid(current.metadata)==1u && (cls==kSurfaceGlossy || cls==kSurfaceNearSpecular));
        using lfs::denoise::RegularR2Coverage;
        const auto half_s2 = float_to_half(s2);
        const auto threshold = float_to_half(1.f / 3.f);
        const bool wider = spatial && !(surface_valid(current.metadata) == 1u && cls == kSurfaceDiffuse && age > 1u && half_s2 < threshold);
        const bool routed = p.regular_r2_coverage != RegularR2Coverage::Original;
        // Use the actual source predicate of the selected R2 kernel. Its witness
        // compares the stored half value, including equality at the threshold.
        const unsigned kind = observation.metadata & 511u;
        const bool near = kind == (1u | (kSurfaceNearSpecular << 2u));
        const bool glossy = kind == (1u | (kSurfaceGlossy << 2u));
        const bool direct_sky = p.regular_r2_coverage == RegularR2Coverage::DeterministicEnvironment && (kind & 3u) == 2u;
        const bool r2 = !direct_sky && !glossy && (!near || half_s2 >= threshold);
        // All valid physical sites contribute to a single published mask.
        // Subsets that followed different history paths must rendezvous here;
        // the first lane cannot publish a separate subset's ownership later.
        const unsigned active=live_sites;
        const unsigned lower = __ballot_sync(active, routed ? r2 : spatial);
        const unsigned upper = __ballot_sync(active, wider && !(direct_sky && !g.static_layout));
        if ((lower || upper) && (threadIdx.x & 31u) == 0u) {
            const unsigned r4_shift = (y & 3u) * 4u;
            const unsigned r4 = ((upper & 0x11111111u) ? 1u : 0u) | ((upper & 0x22222222u) ? 2u : 0u) |
                                ((upper & 0x44444444u) ? 4u : 0u) | ((upper & 0x88888888u) ? 8u : 0u);
            unsigned low;
            if (routed) {
                const unsigned pair = ((lower & 0x55555555u) ? 1u : 0u) | ((lower & 0xaaaaaaaau) ? 2u : 0u);
                low = pair << lfs::denoise::regular_r2_bit(x,y);
            } else {
                const unsigned original = ((lower & 0x11111111u) ? 1u : 0u) | ((lower & 0x22222222u) ? 2u : 0u) |
                                          ((lower & 0x44444444u) ? 4u : 0u) | ((lower & 0x88888888u) ? 8u : 0u);
                low = original << r4_shift;
            }
            const unsigned columns = (g.output_width * 3u + 127u) / 128u;
            const unsigned group = (y / 32u) * columns + x / 128u;
            atomicOr(pointer<unsigned>(g.frame_status) + 1u + group, low | (r4 << (16u + r4_shift)));
        }
    }
    native::TemporalSample output{};
    output.surface_key=current.surface_key;output.depth=current.depth;output.packed_normal=current.packed_normal;
    output.metadata=pack_metadata(surface_valid(current.metadata),signal_domain(current.metadata),surface_class(current.metadata),age,0u)|(logarithmic?1u<<30u:0u);
    if constexpr(CertifiedRegular) {
        const unsigned bits=float_to_half(current_innovation);
        output.metadata|=((bits&0x7fffu)<<15u)|((bits&0x8000u)<<16u);
    }
    output.sigma_albedo={float_to_half(s2),current.signal_albedo.y};
    if constexpr(CertifiedRegular) {
        if(p.current_panel_samples) {
            const bool optical=surface_valid(current.metadata) && (cls==kSurfaceGlossy || cls==kSurfaceNearSpecular);
            const unsigned packed=optical?(unsigned(float_to_half(panel_overlap))<<16u)|previous_display:0x7e000000u;
            pointer<unsigned>(p.current_panel_samples)[i]=packed;
        }
    }
    output.signal_physical_mean_rms=observation.linear_albedo_rgb_diffuse_factor;
    // MR fields are not ordinary TemporalSample signal/variance semantics.
    if constexpr (CertifiedRegular) {
        pointer<native::TemporalSample>(g.history_write)[i] = output;
    } else if(p.history_metadata_staging) {
        uint2 metadata_and_sigma;
        memcpy(&metadata_and_sigma,reinterpret_cast<const unsigned char*>(&output)+sizeof(uint4),sizeof(metadata_and_sigma));
        pointer<uint2>(p.history_metadata_staging)[i]=metadata_and_sigma;
    } else if(g.history_geometry_write) {
        uint4 tail;
        memcpy(&tail,reinterpret_cast<const unsigned char*>(&output)+sizeof(uint4),sizeof(tail));
        pointer<uint4>(g.history_write)[i]=tail;
    } else {
        pointer<native::TemporalSample>(g.history_write)[i]=output;
    }
    const native::Half4 stored_mean{float_to_half(mean.x),float_to_half(mean.y),
        float_to_half(mean.z),float_to_half(measured_root<CertifiedRegular>(fmaxf(mean.w,0.f)))};
    if(!half_is_finite(stored_mean.x)||!half_is_finite(stored_mean.y)||
       !half_is_finite(stored_mean.z)||!half_is_finite(stored_mean.w))
        mark_frame_status(g,native::kPhase3FrameStatusNonfinite);
    const uint2 mean_bits=make_uint2(
        unsigned(stored_mean.x)|(unsigned(stored_mean.y)<<16u),
        unsigned(stored_mean.z)|(unsigned(stored_mean.w)<<16u));
    if(!CertifiedRegular && p.committed_measurement){
        // Each thread owns one complete 16-byte hit slot. Commit reads its
        // own x/y words before replacing this slot with the current geometry.
        auto &slot=pointer<uint4>(p.measurement_write)[i];slot.x=mean_bits.x;slot.y=mean_bits.y;
    } else pointer<uint2>(p.measurement_write)[i]=mean_bits;
    // One observation is an exact identity, including the ES variance wire.
    // Spatial reconstruction never feeds either of the two histories.
    if(age==1u) {
        native::Half4 unknown=input;
        unknown.w |= 0x8000u; // Internal suffix wire: magnitude is ES variance, sign is unknown temporal variance.
        pointer<native::Half4>(g.current_phase2)[i]=unknown;
    } else {
        const auto linear_mean=logarithmic?mr_decode_glossy<CertifiedRegular>(mean,modulation):mean;
        float jacobian=1.f;
        if(logarithmic) {
            const float log_luma=mr_luma(mean),physical_luma=measured_expm1<CertifiedRegular>(log_luma);
            jacobian=(CertifiedRegular?1.f+physical_luma:expf(log_luma))*(physical_luma>0.f?mr_luma(linear_mean)/physical_luma:1.f);
        }
        const float variance=fmaxf(mean.w,0.f)*s2/fmaxf(1.f-s2,1.e-6f)*jacobian*jacobian;
        native::Half4 result{float_to_half(linear_mean.x),float_to_half(linear_mean.y),
            float_to_half(linear_mean.z),float_to_half(variance)};
        if(!half_is_finite(result.x)||!half_is_finite(result.y)||
           !half_is_finite(result.z)||!half_is_finite(result.w))
            mark_frame_status(g,native::kPhase3FrameStatusNonfinite);
        pointer<native::Half4>(g.current_phase2)[i]=result;
    }
}

extern "C" __global__ void lfs_early_measurement_f16_rms(MrParams p){early_measurement_body<false>(p);}
extern "C" __global__ void lfs_early_measurement_rgb_primary_ray(MrParams p){early_measurement_body<true>(p);}

extern "C" __global__ void lfs_regular_measurement_f16_rms(MrParams p) {
    early_measurement_body<false, false, true>(p);
}
extern "C" __global__ void lfs_vertex_measurement_f16_rms(MrParams p) {
    early_measurement_body<false,false,true,true>(p);
}
extern "C" __global__ void lfs_vertex_measurement_rgb_primary_ray(MrParams p) {
    early_measurement_body<true,false,true,true>(p);
}
extern "C" __global__ void lfs_regular_measurement_rgb_primary_ray(MrParams p) {
    early_measurement_body<true, false, true>(p);
}

extern "C" __global__ __launch_bounds__(128,4)
void lfs_certified_measurement_f16_rms(MrParams p) {
    early_measurement_body<false,false,true,false,true>(p);
}
extern "C" __global__ __launch_bounds__(128,4)
void lfs_certified_measurement_rgb_primary_ray(MrParams p) {
    early_measurement_body<true,false,true,false,true>(p);
}
extern "C" __global__ __launch_bounds__(128,4)
void lfs_certified_vertex_measurement_rgb_primary_ray(MrParams p) {
    early_measurement_body<true,false,true,true,true>(p);
}

extern "C" __global__ void lfs_stationary_measurement(MrParams p){early_measurement_body<false,true>(p);}
extern "C" __global__ void lfs_stationary_measurement_glossy(MrParams p){early_measurement_body<true,true>(p);}
// A source TriangleHit has finite barycentrics; a miss has invalid instance/primitive IDs.
// This tag preserves the actual bank choice across both offset-aware and base-address native submissions.
__device__ __forceinline__ uint4 history_slot_marker(){return make_uint4(0x5117c34du,0x193ab8c2u,0x7f800000u,0xff800000u);}
__device__ __forceinline__ bool retained_history_slot(uint4 x){
    const auto m=history_slot_marker();return x.x==m.x&&x.y==m.y&&x.z==m.z&&x.w==m.w;
}
extern "C" __global__ void lfs_restore_native_history(MrParams p){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=3u*p.geometry.output_width*p.geometry.output_height)return;
    const auto *first=pointer<const uint4>(p.geometry.history_geometry_write);
    auto *second=pointer<uint4>(p.native_second_hit);
    // No thread writes either marker. Site zero has its separately retained real head.
    if(retained_history_slot(second[0]))return;
    if(!retained_history_slot(first[0])){
        if(i==0u)mark_frame_status(p.geometry,native::kPhase3FrameStatusInvalidReconstruction);
        return;
    }
    if(i!=0u)second[i]=first[i];
}

// Enqueued after all reconstruction and presentation consumers of the preceding
// frame. The exclusive framebuffer exposes its packed guide through commit.
extern "C" __global__ void lfs_commit_visible_geometry(MrParams p){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p.geometry.output_width*3u*p.geometry.output_height)return;
    const auto o=pointer<const Observation>(p.geometry.current_packed_observation)[i];
    const uint4 head=make_uint4(unsigned(o.surface_key),unsigned(o.surface_key>>32),
                               __float_as_uint(o.hit_distance),o.packed_geometric_normal_oct16);
    if(p.history_metadata_staging) {
        const uint2 state=pointer<const uint2>(p.history_metadata_staging)[i];
        uint2 albedo;memcpy(&albedo,&o.linear_albedo_rgb_diffuse_factor,sizeof(albedo));
        if(p.committed_measurement){
            auto *first=pointer<uint4>(p.geometry.history_geometry_write);
            auto *second=pointer<uint4>(p.native_second_hit);
            const uint4 measured=first[i];
            pointer<uint2>(p.committed_measurement)[i]=make_uint2(measured.x,measured.y);
            const unsigned residual=p.current_angular_residuals?pointer<const unsigned short>(p.current_angular_residuals)[i]:0u;
            if(state.x & ~kCurrentMrMetadataMask)mark_frame_status(p.geometry,native::kPhase3FrameStatusInvalidReconstruction);
            const unsigned metadata=state.x|((residual&0x7fffu)<<15u)|((residual&0x8000u)<<16u);
            pointer<uint4>(p.geometry.history_write)[i]=make_uint4(metadata,state.y,albedo.x,albedo.y);
            if(i==0u){
                *pointer<uint4>(p.geometry.native_first_history_head)=head;
                first[i]=history_slot_marker();second[i]=history_slot_marker();
            } else {first[i]=head;second[i]=head;}
        } else {
            auto *record=pointer<uint4>(p.geometry.history_write)+2ull*i;
            record[0]=head;record[1]=make_uint4(state.x,state.y,albedo.x,albedo.y);
        }
    } else {
        pointer<uint4>(p.geometry.history_geometry_write)[i]=head;
    }
}
