#ifdef LFS_CHART_USE_SHARED_INVERSE
template<int Width,int Height,bool Cache=false,bool Directional=false,bool MovingCache=false,bool CompactColors=false,bool StreamNear=false>
__device__ __forceinline__ void compact_view_chart_refinement(EncodedAtrousNativeParams p,int x,int y,const RefinementGuide &cg,Vec4 cv,bool ck,const Half4 *signals=nullptr,const ViewChartGuide36 *guides=nullptr,unsigned rank=0u,const Vec3 *normal_values=nullptr,const Vec4 *source_colors=nullptr,const float *source_inverse=nullptr,float center_inverse=0.f,Vec3 receiver_axis={},unsigned center_slot=0u){
#else
template<int Width,int Height,bool Cache=false,bool Directional=false,bool MovingCache=false>
__device__ __forceinline__ void view_chart_refinement(EncodedAtrousNativeParams p,int x,int y,const RefinementGuide &cg,Vec4 cv,bool ck,const Half4 *signals=nullptr,const ViewChartGuide36 *guides=nullptr,unsigned rank=0u,const Vec3 *normal_values=nullptr,const Vec4 *source_colors=nullptr){
#endif
 const unsigned index=unsigned(y*Width+x);
 if(!p.view_chart_ids||!p.view_chart_offsets||(Cache&&!p.view_chart_slots)||p.view_chart_wire!=0x56434832u||p.view_chart_count!=unsigned(Width*Height)||p.view_chart_views!=p.static_view_count||!p.view_chart_templates){fail(p,kFrameStatusInvalidLayout);return;}
 const Vec3 c{cv.x,cv.y,cv.z},ca{from_half(cg.a.x),from_half(cg.a.y),from_half(cg.a.z)};const unsigned channel=index%3u;
#ifdef LFS_CHART_USE_SHARED_INVERSE
 const float lc=luminance(c),cc=selected(c,channel),cs=CompactColors?__fmul_rn(cc,center_inverse):cc/fmaxf(lc,1.e-20f),phi=32.f*sqrtf(fmaxf(cv.w,1.e-4f))+1.e-3f;
#else
 constexpr bool StreamNear=false;
 const float lc=luminance(c),cc=selected(c,channel),cs=cc/fmaxf(lc,1.e-20f),phi=32.f*sqrtf(fmaxf(cv.w,1.e-4f))+1.e-3f;
#endif
#ifdef LFS_ENCODED_PLAIN_CURRENT
 float carrier_sum=0.f, spatial_sum=0.f, spatial_mass=0.f, spatial_variance=0.f;
 const bool measure_local=!p.resident_storage && p.register_motion && cg.flags==kSurfaceNearSpecular;
 const float center_variance=cv.w*cs*cs;
 const float density_information=measure_local && ck && center_variance>1.e-20f?measured_density_information(p,index):0.f;
 const float inverse_twice_center=.5f/fmaxf(center_variance,1.e-20f);
 const bool repeated=!MovingCache && measure_local && p.hold_same && p.hold_previous &&
     field_same_physical_ray(p,index,reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[index]);

#else
 Vec3 sum{},original_sum{};
 unsigned short terminal_geometry[9]{};unsigned terminal_center=8u;
 const bool cache_geometry=!StreamNear && Directional && p.register_motion && !p.hold_same && cg.flags==kSurfaceGlossy;
 float original_mass=0.f,original_variance=0.f;
 const bool narrow_lobe=StreamNear || (Directional && p.register_motion && cg.flags==kSurfaceNearSpecular);
 const bool use_secondary_support=narrow_lobe && !p.hold_same &&
     reinterpret_cast<const denoise_native::TemporalSample32*>(p.regular_measurement_geometry)[index].sigma_albedo.x>=to_half(1.f/3.f);
 const Vec4 independent=use_secondary_support?unpack(load_signal(reinterpret_cast<const Half4*>(p.scratch_b),index)):
                                                Vec4{0.f,0.f,0.f,-1.f};
 Vec3 inner_sum{};float inner_mass=0.f,inner_variance=0.f;
 const float q_center_variance=cv.w*cs*cs;
 const float q_density_information=narrow_lobe && ck && q_center_variance>1.e-20f?measured_density_information(p,index):0.f;
 const float q_inverse_twice_center=.5f/fmaxf(q_center_variance,1.e-20f);
 const bool repeated=!MovingCache && narrow_lobe && p.hold_same && p.hold_previous &&
     field_same_physical_ray(p,index,reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[index]);
#endif
#if defined(LFS_CHART_USE_SHARED_INVERSE) && defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
 const bool finite_lobe=!StreamNear && Directional && CompactColors && cg.flags==kSurfaceGlossy &&
        p.regular_measurement_geometry &&
        !(reinterpret_cast<const denoise_native::TemporalSample32*>(p.regular_measurement_geometry)[index].metadata&(1u<<30u));
 const bool optical=CompactColors && (Width==4608 || Width==11520) && (narrow_lobe || finite_lobe) &&
                    dot(receiver_axis,receiver_axis)>.5f;
 float rough_width=0.f,inner_spread=0.f,inner_optical_mass=0.f,angular_variance=0.f;
 const bool mixed_interface=optical && narrow_lobe && cg.a.w!=0u;
 float signal_angle=0.f,angle_energy=0.f,angular_gradient=0.f;
 if(optical) {
     const auto md=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[index].metadata;
     const float rough=float((md>>9u)&255u)/255.f;
     rough_width=(rough*rough)*(rough*rough);
 }
#endif
 float mass=0.f,var=0.f;bool known=ck;
#ifdef LFS_ENCODED_PLAIN_CURRENT
 constexpr unsigned footprint=9u;
#else
 constexpr unsigned footprint=25u;
#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
 ChartSlotStream slots{};
 if constexpr(StreamNear) {
     static_assert(Cache);
     const auto *words=reinterpret_cast<const unsigned*>(p.view_chart_slots)+rank;
     slots={words,p.count,words[0],words[p.count],0u};
 }
#endif
 #pragma unroll (StreamNear?1:footprint)
 for(unsigned k=0u;k<footprint;++k){
#if defined(LFS_CHART_USE_SHARED_INVERSE) && defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
  if(k==9u && optical && inner_optical_mass>0.f) {
      angular_variance=rough_width+inner_spread/inner_optical_mass;
      if(mixed_interface && angle_energy>0.f)angular_gradient=fmaxf(signal_angle,0.f)/angle_energy;
  }
#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
  if(k==9u && use_secondary_support){inner_sum=sum;inner_mass=mass;inner_variance=var;}
#endif
  ViewChartGuide36 sg{}; Half4 value_signal{};
#ifdef LFS_CHART_USE_SHARED_INVERSE
  Vec4 color_terms{};float donor_inverse=0.f;unsigned optical_slot=0u;
#else
  Vec4 color_terms{};
#endif
  if constexpr (Cache) {
    unsigned slot;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if constexpr(StreamNear)slot=slots.take(k+1u==footprint);
    else
#endif
    {
        const unsigned bit=k*9u,col=bit>>5u,shift=bit&31u;
        const auto* words=reinterpret_cast<const unsigned*>(p.view_chart_slots);
        unsigned value=words[size_t(col)*p.count+rank]>>shift;
        if(shift>23u)value|=words[size_t(col+1u)*p.count+rank]<<(32u-shift);
        slot=value&511u;
    }
    sg=guides[slot];value_signal=signals[slot];
#ifdef LFS_CHART_USE_SHARED_INVERSE
    if constexpr (CompactColors) {donor_inverse=source_inverse[slot];optical_slot=slot;}
    else if constexpr (MovingCache) color_terms=source_colors[slot];
#else
    if constexpr (MovingCache) color_terms=source_colors[slot];
#endif
  } else {
    const unsigned tid=chart_template_id(p,index);
    if(tid>=p.view_chart_templates){fail(p,kFrameStatusInvalidLayout);return;}
    const unsigned bits=reinterpret_cast<const unsigned*>(p.view_chart_offsets)[size_t(tid)*34u+k];
    const int xx=x+int(static_cast<short>(bits)),yy=y+int(static_cast<short>(bits>>16));
    if(xx<0||xx>=Width||yy<0||yy>=Height){fail(p,kFrameStatusInvalidLayout);return;}
    const unsigned j=unsigned(yy*Width+xx);
    const auto* views=reinterpret_cast<const unsigned char*>(p.static_view_id);
    if(views[j]!=views[index]){fail(p,kFrameStatusInvalidLayout);return;}
    sg=direct_chart_guide<Width>(p,j,views[j]);
    value_signal=load_signal(reinterpret_cast<const Half4*>(p.scratch_a),j);
  }
  const unsigned validity=sg.metadata&3u,flags=(sg.metadata>>2u)&127u;
  if(validity!=1u||sg.instance!=cg.instance||flags!=cg.flags)continue;
#if defined(LFS_CHART_USE_SHARED_INVERSE) && defined(LFS_ENCODED_GLOSSY_MEASUREMENT)
  Vec3 packed_axis{};
  if constexpr(CompactColors && (Width==4608 || Width==11520)) {
      packed_axis=sg.n;
      sg.n=chart_direction<false>(sg.n);
  }
#endif
  const Vec4 raw=unpack(value_signal);const bool sk=raw.w>=0.f;const float v=fabsf(raw.w);const Vec3 z{raw.x,raw.y,raw.z},dp=sg.p-cg.p,sa{from_half(sg.a0),from_half(sg.a1),from_half(sg.a2)};
#ifdef LFS_CHART_USE_SHARED_INVERSE
  const float component=selected(z,channel),delta=cc-component;
  const float l=MovingCache && !CompactColors?color_terms.w:luminance(z);
  const float scale=CompactColors?__fmul_rn(component,donor_inverse):
      (MovingCache?selected({color_terms.x,color_terms.y,color_terms.z},channel):component/fmaxf(l,1.e-20f));
  const float vv=cv.w*cs*cs+v*scale*scale;
#else
  const float l=MovingCache?color_terms.w:luminance(z),component=selected(z,channel),scale=MovingCache?selected({color_terms.x,color_terms.y,color_terms.z},channel):component/fmaxf(l,1.e-20f),delta=cc-component,vv=cv.w*cs*cs+v*scale*scale;
#endif
  float albedo_distance=0.f;
  if constexpr(MovingCache) {
     const bool equal_albedo=cg.a.x==sg.a0 && cg.a.y==sg.a1 && cg.a.z==sg.a2;
     albedo_distance=equal_albedo?0.f:length(ca-sa);
  }
#ifdef LFS_CHART_USE_SHARED_INVERSE
  const float residual=CompactColors?
      (ck&&sk?__fmul_rn(.5f*delta*delta,regular_chart_reciprocal(fmaxf(vv,1.e-20f))):
               __fmul_rn(fabsf(lc-l),regular_chart_reciprocal(phi))):
      (ck&&sk?.5f*delta*delta/fmaxf(vv,1.e-20f):fabsf(lc-l)/phi);
#else
  const float residual=ck&&sk?.5f*delta*delta/fmaxf(vv,1.e-20f):fabsf(lc-l)/phi;
#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
  const float geo=pow32(fmaxf(dot(cg.n,view_chart_normal(p,sg,normal_values)),0.f));
  const float exponent=-fabsf(dot(dp,cg.n))*rsqrtf(fmaxf(dot(dp,dp),1.e-6f))-4.f*(MovingCache?albedo_distance:length(ca-sa));
  float w=geo*expf(exponent-residual);
  float terminal_overlap=1.f;
#ifdef LFS_CHART_USE_SHARED_INVERSE
  if(optical) {
      const auto axis=chart_direction<true>(packed_axis);
      if(dot(axis,axis)>.5f) {
          const unsigned center_mode=cg.a.w,source_mode=sg.metadata>>9u;
          const float direct_share=((center_mode^source_mode)&1u)?0.f:
              (1.f-float(max(center_mode>>1u,source_mode>>1u))*(1.f/126.f));
          const Vec3 delta_axis=receiver_axis-axis;
          const float sin2=.25f*dot(delta_axis,delta_axis);
          const float delta2=delta*delta;
          const float structure=fmaxf(delta2-vv,0.f)/fmaxf(delta2+vv,1.e-20f);
          if(k<9u) {
              if(finite_lobe && cache_geometry) {
                  // Resolve the terminal nine-ray filter only where the observed
                  // radiance change and the measured BSDF directions both identify structure.
                  const float den=fmaf(1.f-rough_width,sin2,rough_width);
                  const float overlap=den>0.f?rough_width/den:1.f;
                  terminal_overlap=fmaf(structure*direct_share,overlap*overlap-1.f,1.f);
              }
              inner_spread=fmaf(w*direct_share,sin2,inner_spread);
              inner_optical_mass+=w*direct_share;
              if(mixed_interface) {
                  // Fit E[(Ls-Lc)^2] = Var(Ls-Lc) + gradient^2 * angular chord^2.
                  // The independent outer ring is evaluated after the inner fit.
                  const float wa=w*direct_share*sin2;
                  signal_angle=fmaf(wa,delta2-vv,signal_angle);
                  angle_energy=fmaf(wa,sin2,angle_energy);
              }
          } else if(inner_optical_mass>0.f) {
              // Angular variance of the measured central footprint broadens
              // the BRDF before we assess the outer region. The core remains usable on a delta reflector.
              const float scale=angular_variance;
              const float denom=fmaf(1.f-rough_width,sin2,scale);
              const float ggx=denom>0.f?scale/denom:1.f;
              // A changing reflected direction displaces a signal only if the
              // measured radiance also differs above this tap pair's shot noise.
              // Low-contrast mirrors still benefit from neighboring observations.
              const float prediction=angular_gradient*sin2;
              const float signal=mixed_interface?prediction/fmaxf(prediction+vv,1.e-20f):
                                 (finite_lobe?structure:1.f);
              w*=fmaf(signal*direct_share,ggx*ggx-1.f,1.f);
          }
      }
  }
#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
  if(k<9u && cache_geometry) {
      terminal_geometry[k]=to_half(geo*expf(exponent)*terminal_overlap);
#ifdef LFS_CHART_USE_SHARED_INVERSE
      if(CompactColors && optical_slot==center_slot)terminal_center=k;
#endif
  }
#endif
#else
  const float geo=pow32(fmaxf(dot(cg.n,view_chart_normal(p,sg,normal_values)),0.f));float w=geo*expf(-fabsf(dot(dp,cg.n))*rsqrtf(fmaxf(dot(dp,dp),1.e-6f))-4.f*(MovingCache?albedo_distance:length(ca-sa))-residual);
#endif
#ifdef LFS_ENCODED_PLAIN_CURRENT
  // Singular reflectors have highly unequal measured donor variances. Give
  // their Gaussian difference density the same normalization as the center.
  if(measure_local) {
      spatial_sum=fmaf(w,component,spatial_sum);
      spatial_mass+=w;
      if(repeated)spatial_variance=fmaf(w*w,v*scale*scale,spatial_variance);
  }
  if(density_information>0.f && sk)
      w*=calibrated_density_ratio(fmaxf(vv,1.e-20f),inverse_twice_center,density_information);
#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
  if(repeated) {
      original_sum+=z*w;original_mass+=w;
      original_variance=fmaf(w*w,v*scale*scale,original_variance);
  }
  if(q_density_information>0.f && sk)
      w*=calibrated_density_ratio(fmaxf(vv,1.e-20f),q_inverse_twice_center,q_density_information);
#endif
  known=known&&(sk||w==0.f);mass+=w;
#ifdef LFS_ENCODED_PLAIN_CURRENT
  carrier_sum+=component*w;

#else
  sum+=z*w;
#endif
  var+=v*w*w;
 }
 if(!(mass>0.f)){fail(p,kFrameStatusInvalidLayout);return;}const float inv=1.f/mass;
#ifdef LFS_ENCODED_PLAIN_CURRENT
 float carrier_result=carrier_sum*inv;
 const float inv_mass=measure_local && spatial_mass>0.f?1.f/spatial_mass:0.f;
 const float local=spatial_sum*inv_mass;
 if(repeated && spatial_mass>0.f) {
     // On a repeated physical ray a rare reflected event can leave a near-zero
     // temporal variance for several exposures. Use the measured uncertainty
     // of the compatible nine-site estimate when trusting its density ratio.
     const float center_variance=cv.w*cs*cs;
     const float confidence=center_variance/fmaxf(center_variance+spatial_variance*inv_mass*inv_mass,1.e-20f);
     carrier_result=fmaf(confidence,carrier_result-local,local);
 }
 const Vec3 result{carrier_result,carrier_result,carrier_result};
#else
 Vec3 result=sum*inv;
 const float original_carrier=selected(result,channel);
 if(use_secondary_support && independent.w>=0.f && inner_mass>0.f && mass>inner_mass) {
     const Vec3 inner=inner_sum*(1.f/inner_mass);
     const Vec3 outer=(sum-inner_sum)*(1.f/(mass-inner_mass));
     const float ci=selected(inner,channel),co=selected(outer,channel),d=co-ci;
     const float ratio=ci/fmaxf(luminance(inner),1.e-20f);
     const float inner_noise=inner_variance*ratio*ratio/(inner_mass*inner_mass);
     const float outer_ratio=co/fmaxf(luminance(outer),1.e-20f);
     const float outer_noise=fmaxf(var-inner_variance,0.f)*outer_ratio*outer_ratio/
                             ((mass-inner_mass)*(mass-inner_mass));
     const float noise=fmaxf(independent.w,1.e-10f);
     const float nominal=(mass-inner_mass)*inv;
     // Minimize the observed error at neighboring real rays, regularized by
     // their uncertainty toward the original 25-site angular measure.
     const float accepted=fminf(nominal,fmaxf(0.f,
         fmaf(independent.x-ci,d,fmaf(noise,nominal,inner_noise))/fmaf(d,d,noise+inner_noise+outer_noise)));
     result=inner+(outer-inner)*accepted;
 }
 if(repeated && original_mass>0.f) {
     const float inv_mass=1.f/original_mass;
     const Vec3 local=original_sum*inv_mass;
     const float center_variance=cv.w*cs*cs;
     const float confidence=center_variance/fmaxf(center_variance+original_variance*inv_mass*inv_mass,1.e-20f);
     result=local+(result-local)*confidence;
 }
#endif
#ifdef LFS_ENCODED_PLAIN_CURRENT
 // This terminal owner exports a single radiance component. Angular reconstruction
 // uses the actual temporal M variance. No later consumer reads this suffix variance.
 const float output_variance=p.resident_storage?fmaxf(var*inv*inv,1.e-4f):1.e-4f;
#else
 const float output_variance=fmaxf(var*inv*inv,1.e-4f);
#endif
 if(!finite3(result)||!isfinite(output_variance)){fail(p);return;}
 const auto encoded=pack({result.x,result.y,result.z,known?output_variance:-output_variance});
#ifdef LFS_ENCODED_PLAIN_CURRENT
 // The center guide already contains the original packed albedo and a validated surface domain.
 present_encoded_carrier<true,Width>(p,index,1u|(cg.flags<<2u),encoded,selected(ca,channel),local,repeated);
#else
 if(cache_geometry) {
     auto *cache=reinterpret_cast<unsigned*>(p.regular_repeat_state);
#ifdef LFS_CHART_USE_SHARED_INVERSE
     if constexpr(CompactColors && (Width==4608 || Width==11520)) {
         // The center always uses its exact fixed coefficient. Use that byte
         // to carry the ninth tap when the center occurs earlier in the template.
         unsigned low=0u,high=0u;
         #pragma unroll
         for(unsigned k=0;k<8u;++k) {
             const unsigned short value=k==terminal_center?terminal_geometry[8]:terminal_geometry[k];
             const unsigned byte=__float2uint_rn(255.f*fminf(1.f,from_half(value)));
             if(k<4u)low|=byte<<(k*8u);else high|=byte<<((k-4u)*8u);
         }
         cache[index]=low;cache[p.count+index]=high;
     } else
#endif
     {
     #pragma unroll
     for(unsigned k=0;k<4u;++k)
         cache[size_t(k)*p.count+index]=unsigned(terminal_geometry[2u*k])|(unsigned(terminal_geometry[2u*k+1u])<<16u);
     cache[4ull*p.count+index]=terminal_geometry[8];
     }
 }
 if(!MovingCache && Directional && p.register_motion && p.hold_same && p.hold_previous && cg.flags==kSurfaceGlossy)
     collect_repeated_es(p,index);
 if(p.register_motion || cg.flags!=kSurfaceNearSpecular) {
     auto stored=encoded;
     if(narrow_lobe)stored.w=to_half(use_secondary_support?original_carrier:selected(result,channel));
     store_signal(reinterpret_cast<Half4*>(p.scratch_b),index,stored);
 }
 if(cg.flags==kSurfaceNearSpecular)
     present_encoded_carrier<false,Width>(p,index,reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation)[index].metadata,encoded);
#endif
}
