
struct FiniteChartCenter {
    Vec3 position,normal,axis,albedo;
    unsigned instance,interface_code,channel;
    float color,variance,luminance,inverse_phi,width;
    bool known,optical;
};

template<bool Inner,bool EstimateOuter,bool RadiometricOnly=false>
__device__ __forceinline__ void accumulate_finite_chart(const FiniteChartCenter &center,
        const ViewChartGuide36 &guide,const Half4 &value,float inverse_luma,float outer_width,
        Vec3 &sum,float &mass,float &spread,float &optical_mass,unsigned &terminal) {
    if(guide.instance!=center.instance || (guide.metadata&511u)!=(1u|(kSurfaceGlossy<<2u)))return;
    const auto sample=unpack(value);
    const Vec3 color{sample.x,sample.y,sample.z};
    const float component=selected(color,center.channel);
    const float diff=center.color-component;
    const float delta2=diff*diff;
    const float chroma=__fmul_rn(component,inverse_luma);
    const float donor_variance=fabsf(sample.w)*chroma*chroma;
    const float variance=center.variance+donor_variance;
    const bool donor_measured=sample.w>=0.f;
    const bool measured_variance=center.known && donor_measured;
    const float contrast=measured_variance?
        __fmul_rn(.5f*delta2,regular_chart_reciprocal(fmaxf(variance,1.e-20f))):
        __fmul_rn(fabsf(center.luminance-luminance(color)),center.inverse_phi);
    const auto offset=guide.p-center.position;
    const Vec3 albedo{from_half(guide.a0),from_half(guide.a1),from_half(guide.a2)};
    const float distance=length(albedo-center.albedo);
    const float plane=fabsf(dot(offset,center.normal))*rsqrtf(fmaxf(dot(offset,offset),1.e-6f));
    const float geometry=pow32(fmaxf(dot(center.normal,chart_direction<false>(guide.n)),0.f));
    const float exponent=-plane-4.f*distance;
    float w=geometry*__expf(exponent-contrast);
    float terminal_overlap=1.f;
    // A below-noise outer difference has exactly unit angular weight.
    // Inner taps also measure the angular footprint even below that noise.
    const bool resolved_structure=delta2>variance && (Inner || outer_width>0.f);
    if constexpr(RadiometricOnly) {
        // The measured contrast and variance already determine whether a
        // real edge can cross this same-surface footprint on the full chart.
        // Each ray carries its own uncertainty. Expand the interval
        // for its measured component and retain the conservative ES bound
        // for the component whose history reset.
        const float measured_part=(center.known?center.variance:0.f)+(donor_measured?donor_variance:0.f);
        const float resolved_noise=Inner?variance:fmaf(2.841458820694124f,measured_part,variance);
        if(delta2>resolved_noise) {
            const float agreement=fminf(1.f,2.f*resolved_noise*regular_chart_reciprocal(fmaxf(delta2+resolved_noise,1.e-20f)));
            if constexpr(Inner)terminal_overlap=agreement;
            else w*=agreement;
        }
    } else if(center.optical && ((Inner && EstimateOuter) || resolved_structure)) {
        const auto axis=chart_direction<true>(guide.n);
        if(dot(axis,axis)>.5f) {
            const unsigned op=guide.metadata>>9u;
            const float branch=((op^center.interface_code)&1u)?0.f:
                (1.f-float(max(op>>1u,center.interface_code>>1u))*(1.f/126.f));
            const auto turn=center.axis-axis;
            const float chord=.25f*dot(turn,turn);
            if constexpr(Inner && EstimateOuter) {
                spread=fmaf(w*branch,chord,spread);
                optical_mass+=w*branch;
            }
            if(resolved_structure) {
                const float width=Inner?center.width:outer_width;
                const float denominator=fmaf(1.f-center.width,chord,width);
                const float overlap=denominator>0.f?width*regular_chart_reciprocal(denominator):1.f;
                const float structure=(delta2-variance)*regular_chart_reciprocal(fmaxf(delta2+variance,1.e-20f));
                const float optics=fmaf(structure*branch,overlap*overlap-1.f,1.f);
                if constexpr(Inner)terminal_overlap=optics;
                else w*=optics;
            }
        }
    }
    if constexpr(Inner)
        terminal=__float2uint_rn(255.f*fminf(1.f,from_half(to_half(geometry*__expf(exponent)*terminal_overlap))));
    sum+=color*w;mass+=w;
}

template<int Width>
__device__ __forceinline__ void stream_finite_chart(const EncodedAtrousNativeParams &p,
        unsigned i,unsigned rank,unsigned center_slot,const RefinementGuide &guide,Vec4 value,bool known,
        const Half4 *samples,const ViewChartGuide36 *guides,const float *inverse,Vec3 axis) {
    const unsigned channel=i%3u;
    const Vec3 rgb{value.x,value.y,value.z};
    const float component=selected(rgb,channel),chroma=__fmul_rn(component,inverse[center_slot]);
    const auto &g=reinterpret_cast<const denoise_native::TemporalSample32*>(p.regular_measurement_geometry)[i];
    const bool optical=!(g.metadata&(1u<<30u)) && dot(axis,axis)>.5f;
    const auto md=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[i].metadata;
    const float rough=float((md>>9u)&255u)/255.f;
    const FiniteChartCenter center{guide.p,guide.n,axis,
        {from_half(guide.a.x),from_half(guide.a.y),from_half(guide.a.z)},guide.instance,guide.a.w,channel,
        component,value.w*chroma*chroma,luminance(rgb),
        regular_chart_reciprocal(32.f*sqrtf(fmaxf(value.w,1.e-4f))+1.e-3f),(rough*rough)*(rough*rough),known,optical};
    const auto *words=reinterpret_cast<const unsigned*>(p.view_chart_slots)+rank;
    ChartSlotStream stream{words,p.count,words[0],words[p.count],0u};
    Vec3 sum{};float mass=0.f,spread=0.f,optical_mass=0.f;
    unsigned left=0u,right=0u,center_tap=8u;
    #pragma unroll 1
    for(unsigned tap=0u;tap<9u;++tap) {
        const unsigned slot=stream.take();
        unsigned byte=0u;
        accumulate_finite_chart<true,Width!=4608,Width==4608>(center,guides[slot],samples[slot],inverse[slot],0.f,
                                      sum,mass,spread,optical_mass,byte);
        if(slot==center_slot)center_tap=tap;
        const unsigned dest=tap==8u?center_tap:tap;
        if(slot!=center_slot && dest<8u) {
            const unsigned delta=byte<<((dest&3u)*8u);
            if(dest<4u)left|=delta;else right|=delta;
        }
    }
    {
        const float outer=optical_mass>0.f?center.width+spread*regular_chart_reciprocal(optical_mass):0.f;
        #pragma unroll 1
        for(unsigned tap=9u;tap<25u;++tap) {
            const unsigned slot=stream.take(tap==24u);unsigned unused=0u;
            accumulate_finite_chart<false,false,Width==4608>(center,guides[slot],samples[slot],inverse[slot],outer,
                                           sum,mass,spread,optical_mass,unused);
        }
    }
    if(!(mass>0.f)){fail(p,kFrameStatusInvalidLayout);return;}
    const Vec3 result=sum*regular_chart_reciprocal(mass);
    if(!finite3(result)){fail(p);return;}
    auto *cache=reinterpret_cast<unsigned*>(p.regular_repeat_state);
    cache[i]=left;cache[p.count+i]=right;
    // The moving terminal filters the original RGB footprint. Its fourth
    // word is a finite-input guard and has no contribution to the output.
    store_signal(reinterpret_cast<Half4*>(p.scratch_b),i,pack({result.x,result.y,result.z,1.e-4f}));
}
