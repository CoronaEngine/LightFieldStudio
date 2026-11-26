// Ordinary Quality, after two actual observations of the same ray.
// Each cell belongs to one calibrated view and owns two Half4 values.
namespace {

__device__ __forceinline__ Vec3 held_square(Vec3 v) {
    return {v.x*v.x,v.y*v.y,v.z*v.z};
}

__device__ __forceinline__ bool held_directional_surface(const AtlasObservation &a) {
    const unsigned cls=(a.metadata>>2u)&127u;
    return cls==kSurfaceGlossy || cls==kSurfaceNearSpecular;
}

template<int Width>
__device__ __forceinline__ Vec3 held_reflected_direction(const EncodedAtrousNativeParams &p,unsigned i,Vec3 world_normal) {
    const unsigned view=reinterpret_cast<const unsigned char*>(p.static_view_id)[i];
    const float u=reinterpret_cast<const float*>(p.static_view_x)[view];
    const float inverse=reinterpret_cast<const float*>(p.static_inverse_ray_length)[i];
    const float x=reinterpret_cast<const float*>(p.static_focal_x)[i%Width];
    const float y=reinterpret_cast<const float*>(p.static_focal_y)[i/Width];
    const Vec3 incident{(x-u)*inverse,y*inverse,p.focal_distance*inverse};
    const Vec3 normal=local_normal(p,world_normal);
    return incident-normal*(2.f*dot(incident,normal));
}

__device__ __forceinline__ float held_lobe_overlap(const AtlasObservation &a,const AtlasObservation &b,Vec3 ra,Vec3 rb) {
    // Normalized GGX NDF at half the separation of the actual reflected
    // directions. The narrower of the two material lobes sets their common
    // directional support. The squared difference also resolves mirror rays.
    const float r=float(min((a.metadata>>9u)&255u,(b.metadata>>9u)&255u))*(1.f/255.f);
    const float alpha=r*r,alpha2=alpha*alpha;
    const Vec3 difference=ra-rb;
    const float sin2=.25f*dot(difference,difference);
    if(alpha2==0.f)return sin2==0.f?1.f:0.f;
    const float ratio=alpha2/fmaf(1.f-alpha2,sin2,alpha2);
    return ratio*ratio;
}

// Include spatial structure in the prediction error of the filtered cell.
// Independent raw paths allow squared weights for the temporal mean noise.
// Subtract its contribution from the measured spatial scatter before adding
// the uncertainty of the predicted mean.
__device__ __forceinline__ float held_cell_sigma(float mean,float second,float average_noise,float mean_noise) {
    const float structure=fmaxf(fmaf(-mean,mean,second)-average_noise+mean_noise,0.f);
    return sqrtf(structure+mean_noise);
}

template<int Width>
__device__ void build_held_view_cell(const EncodedAtrousNativeParams &p,unsigned cell) {
    auto *out=reinterpret_cast<Half4*>(p.atlas_moments)+2ull*cell;
    const auto *sites=reinterpret_cast<const unsigned*>(p.atlas);
    const auto *series=reinterpret_cast<const Half4*>(p.hold_state);
    const unsigned i=sites[cell];
    // A zero support marks an empty, newly disclosed or unsupported ray.
    out[0]={};
    if(i>=p.count || series[i].w<2u)return;
    const auto *surface=reinterpret_cast<const AtlasObservation*>(p.packed_observation);
    const auto a=surface[i];
    const auto normal=unpack_normal(a.packed_geometric_normal_oct16);
    const bool directional=held_directional_surface(a);
    const Vec3 reflection=directional?held_reflected_direction<Width>(p,i,normal):Vec3{};
    const auto *views=reinterpret_cast<const unsigned char*>(p.static_view_id);
    const unsigned view=views[i],pattern=chart_template_id(p,i);
    if(pattern>=p.view_chart_templates || view>=p.static_view_count) {fail(p,kFrameStatusInvalidLayout);return;}
    const auto *offsets=reinterpret_cast<const unsigned*>(p.view_chart_offsets)+34ull*pattern;
    const auto *raw=reinterpret_cast<const float4*>(p.cell_coefficients);
    const float inverse_radius=18.8495559f/(25.f*p.static_view_count);
    Vec3 sum{},second{},average_noise{},mean_noise{};
    float mass=0.f,nominal=0.f,directional_mass=0.f;
    for(unsigned tap=0;tap<25u;++tap) {
        const unsigned offset=offsets[tap];
        const int dx=static_cast<short>(offset),dy=static_cast<short>(offset>>16u);
        const unsigned j=unsigned(int(i)+dy*Width+dx);
        if(j>=p.count || views[j]!=view) {fail(p,kFrameStatusInvalidLayout);return;}
        const float spatial=__expf(-(dx*dx/9.f+float(dy*dy))*inverse_radius);
        nominal+=spatial;
        const unsigned count=series[j].w;
        if(count<2u)continue;
        const auto neighbor=surface[j];
        const float weight=spatial*atlas_geometry(a,neighbor,normal);
        if(!(weight>0.f))continue;
        if(directional) {
            const Vec3 other=held_reflected_direction<Width>(p,j,unpack_normal(neighbor.packed_geometric_normal_oct16));
            directional_mass=fmaf(weight,held_lobe_overlap(a,neighbor,reflection,other),directional_mass);
        }
        const auto moment=raw[j];
        const Vec3 rgb{moment.x,moment.y,moment.z};
        const float chroma_scale=moment.w/fmaxf(luminance(rgb),1.e-12f);
        const Vec3 squared=held_square(rgb);
        // sigma is the population deviation, hence var(mean)=sigma^2/(n-1).
        const float noise_weight=weight*chroma_scale*chroma_scale/float(count-1u);
        sum+=rgb*weight;
        second+=squared*weight;
        average_noise+=squared*noise_weight;
        mean_noise+=squared*(weight*noise_weight);
        mass+=weight;
    }
    if(!(mass>0.f))return;
    const float inv=1.f/mass;
    const Vec3 mean=sum*inv,sq=second*inv,noise=average_noise*inv,mn=mean_noise*(inv*inv);
    const Vec3 sigma{held_cell_sigma(mean.x,sq.x,noise.x,mn.x),
                     held_cell_sigma(mean.y,sq.y,noise.y,mn.y),
                     held_cell_sigma(mean.z,sq.z,noise.z,mn.z)};
    // A source outside the node format's finite range supplies no evidence.
    // The path and its full-precision temporal observation are still intact.
    if(!finite3(mean) || !finite3(sigma) ||
       fmaxf(fmaxf(mean.x,mean.y),mean.z)>65504.f || fmaxf(fmaxf(sigma.x,sigma.y),sigma.z)>65504.f)return;
    out[0]=pack({mean.x,mean.y,mean.z,mass/nominal});
    out[1]=pack({sigma.x,sigma.y,sigma.z,directional?directional_mass*inv:1.f});
}

struct HeldSideEvidence {
    float value{};
    float variance{};
    float support{};
    float directional_fraction{};
};

template<int Width,int Height>
__device__ HeldSideEvidence held_view_side(const EncodedAtrousNativeParams &p,const AtlasObservation &center,
    Vec3 normal,Vec3 reflection,bool directional,unsigned view,unsigned ch,float px,int y) {
    if(!(px>=-.5f && px<float(Width)-.5f))return {};
    const int side=int(p.atlas_side),cols=(Width+3*side-1)/(3*side),rows=(Height+side-1)/side;
    const float inverse=1.f/float(side);
    const float cx=(px+.5f)*(inverse/3.f)-.5f,cy=(float(y)+.5f)*inverse-.5f;
    const int left=int(floorf(cx)),top=int(floorf(cy));
    const float fx=cx-left,fy=cy-top;
    const auto *sites=reinterpret_cast<const unsigned*>(p.atlas);
    const auto *surface=reinterpret_cast<const AtlasObservation*>(p.packed_observation);
    const auto *nodes=reinterpret_cast<const Half4*>(p.atlas_moments);
    float mass=0.f,total=0.f,second=0.f,noise=0.f,directional_mass=0.f;
    #pragma unroll
    for(unsigned tap=0;tap<4u;++tap) {
        const int x=left+int(tap&1u),row=top+int(tap>>1u);
        if(unsigned(x)>=unsigned(cols) || unsigned(row)>=unsigned(rows))continue;
        const unsigned cell=(view*rows+unsigned(row))*cols+unsigned(x);
        const unsigned j=sites[cell];
        if(j>=p.count)continue;
        const Half4 color=nodes[2ull*cell];
        const float coverage=from_half(color.w);
        if(!(coverage>0.f))continue;
        const float area=((tap&1u)?fx:1.f-fx)*((tap&2u)?fy:1.f-fy);
        const auto neighbor=surface[j];
        const float weight=area*coverage*atlas_geometry(center,neighbor,normal);
        if(!(weight>0.f))continue;
        const Half4 uncertainty=nodes[2ull*cell+1u];
        if(directional) {
            const auto other=held_reflected_direction<Width>(p,j,unpack_normal(neighbor.packed_geometric_normal_oct16));
            const float matched=from_half(uncertainty.w)*held_lobe_overlap(center,neighbor,reflection,other);
            directional_mass=fmaf(weight,matched,directional_mass);
        }
        const float value=from_half(ch==0u?color.x:ch==1u?color.y:color.z);
        const float sigma=from_half(ch==0u?uncertainty.x:ch==1u?uncertainty.y:uncertainty.z);
        total=fmaf(weight,value,total);
        second=fmaf(weight,value*value,second);
        // Neighboring cells of ONE view can use overlapping 25-site stencils.
        // Their standard deviations add before squaring.
        noise=fmaf(weight,sigma,noise);
        mass+=weight;
    }
    if(!(mass>0.f))return {};
    const float inv=1.f/mass,mean=total*inv,sigma=noise*inv;
    const float scatter=fmaxf(fmaf(-mean,mean,second*inv),0.f);
    return {mean,fmaf(sigma,sigma,scatter),fminf(mass,1.f),directional?fminf(1.f,directional_mass*inv):1.f};
}

struct HeldRingEstimate {
    float sum{},second{},noise{},mean_noise{},mass{},nominal{};
};
template<int Width>
__device__ float first_held_footprint(const EncodedAtrousNativeParams &p,unsigned i,const AtlasObservation &a,
                                    unsigned ch,float current) {
    const unsigned pattern=chart_template_id(p,i);
    if(pattern>=p.view_chart_templates)return current;
    const auto *dictionary=reinterpret_cast<const unsigned*>(p.view_chart_offsets)+34ull*pattern;
    const auto *obs=reinterpret_cast<const AtlasObservation*>(p.packed_observation);
    const auto *series=reinterpret_cast<const Half4*>(p.hold_state);
    const auto *moments=reinterpret_cast<const float4*>(p.cell_coefficients);
    const auto *views=reinterpret_cast<const unsigned char*>(p.static_view_id);
    const unsigned view=views[i];
    const auto normal=unpack_normal(a.packed_geometric_normal_oct16);
    const float inv_radius=18.8495559f/(25.f*p.static_view_count);
    HeldRingEstimate ring[2]{};
    for(unsigned k=0;k<25u;++k) {
        const unsigned offset=dictionary[k];
        const int dx=static_cast<short>(offset),dy=static_cast<short>(offset>>16u);
        if(!dx && !dy)continue;
        const unsigned j=unsigned(int(i)+dy*Width+dx);
        if(j>=p.count || views[j]!=view)continue;
        // The 25-site dictionary is in physical-distance order. Its central
        // nine and outer sixteen rays have disjoint original path samples.
        auto &r=ring[k<9u?0u:1u];
        const float spatial=__expf(-(dx*dx/9.f+float(dy*dy))*inv_radius);
        r.nominal+=spatial;
        const auto &b=obs[j];
        const unsigned n=series[j].w;
        if(n<2u || b.surface_key!=a.surface_key || (b.metadata&511u)!=(a.metadata&511u))continue;
        const float w=spatial*atlas_geometry(a,b,normal);
        if(!(w>0.f))continue;
        const auto m=moments[j];const Vec3 rgb{m.x,m.y,m.z};
        const float signal=selected(rgb,ch),sigma=m.w*signal/fmaxf(luminance(rgb),1.e-12f);
        const float variance=sigma*sigma/float(n-1u);
        r.sum=fmaf(w,signal,r.sum);r.second=fmaf(w,signal*signal,r.second);
        r.noise=fmaf(w,variance,r.noise);r.mean_noise=fmaf(w*w,variance,r.mean_noise);
        r.mass+=w;
    }
    if(!(ring[0].mass>0.f && ring[1].mass>0.f))return current;
    float center[2],uncertainty[2],coverage=1.f;
    for(unsigned k=0;k<2u;++k) {
        const auto &r=ring[k];const float inv=1.f/r.mass;
        center[k]=r.sum*inv;
        const float noise=r.mean_noise*inv*inv;
        const float scatter=fmaxf(r.second*inv-center[k]*center[k]-r.noise*inv+noise,0.f);
        uncertainty[k]=scatter+noise;
        coverage=fminf(coverage,r.mass/fmaxf(r.nominal,1.e-20f));
    }
    const float shared=(center[0]-current)*(center[1]-current);
    if(!(shared>0.f))return current;
    const float gap=center[1]-center[0];
    const float t=ring[1].mass/(ring[0].mass+ring[1].mass);
    const float variance=fmaf(1.f-t,uncertainty[0],t*uncertainty[1])+t*(1.f-t)*gap*gap;
    const float target=fmaf(t,gap,center[0]);
    const auto m=unpack(reinterpret_cast<const Half4*>(p.field_measurements)[i]);
    const auto av=a.linear_albedo_rgb_diffuse_factor;
    const float albedo=from_half(ch==0u?av.x:(ch==1u?av.y:av.z));
    const float z=ch==0u?m.x:(ch==1u?m.y:m.z);
    const float scale=p.final_energy*sqrtf(fmaf(albedo,albedo,.01f))*z/fmaxf(luminance({m.x,m.y,m.z}),1.e-20f);
    const float local_variance=fabsf(m.w)*scale*scale;
    const float delta=target-current;
    const float structural=fmaxf(delta*delta-local_variance-variance,0.f);
    const float precision=local_variance/fmaxf(local_variance+variance+structural,1.e-20f);
    const float fraction=precision*coverage*shared/fmaxf(shared+variance,1.e-20f);
    return fmaxf(0.f,fmaf(fraction,target-current,current));
}

template<int Width,int Height>
__device__ void present_held_view_cell(const EncodedAtrousNativeParams &p,unsigned i) {
    const auto *series=reinterpret_cast<const Half4*>(p.hold_state);
    const unsigned count=series[i].w;
    if(count==0u)return;
    const unsigned ch=i%3u,slot=4u*(i/3u)+ch;
    auto *history=reinterpret_cast<float2*>(p.hold_state)+3ull*p.count;
    auto *image=reinterpret_cast<float*>(p.canonical);
    const float baseline=image[slot];
    if(count<2u) {
        history[i]=make_float2(baseline,baseline);
        return;
    }
    const auto a=reinterpret_cast<const AtlasObservation*>(p.packed_observation)[i];
    const unsigned view=reinterpret_cast<const unsigned char*>(p.static_view_id)[i];
    const float depth=a.hit_distance*reinterpret_cast<const float*>(p.static_inverse_ray_length)[i];
    float desired=baseline;
    if(view>0u && view+1u<p.static_view_count && depth>0.f) {
        const auto *positions=reinterpret_cast<const float*>(p.static_view_x);
        const float ul=positions[view-1u],u=positions[view],ur=positions[view+1u];
        const float parallax=(1.f-1.f/depth)*p.atlas_inverse_dx;
        const int x=int(i%Width),y=int(i/Width);
        const Vec3 normal=unpack_normal(a.packed_geometric_normal_oct16);
        const bool directional=held_directional_surface(a);
        const Vec3 reflection=directional?held_reflected_direction<Width>(p,i,normal):Vec3{};
        const auto left=held_view_side<Width,Height>(p,a,normal,reflection,directional,view-1u,ch,fmaf(ul-u,parallax,float(x)),y);
        const auto right=held_view_side<Width,Height>(p,a,normal,reflection,directional,view+1u,ch,fmaf(ur-u,parallax,float(x)),y);
        if(left.support>0.f && right.support>0.f && ur!=ul) {
            const float exposed=baseline<=.04045f?baseline/12.92f:__powf((baseline+.055f)/1.055f,2.4f);
            const float linear=-log1pf(-fminf(exposed,.99999994f));
            const float coherent=(left.value-linear)*(right.value-linear);
            if(coherent>0.f) {
                const float t=fminf(1.f,fmaxf(0.f,(u-ul)/(ur-ul)));
                const float delta=right.value-left.value;
                // Distinct calibrated views use distinct path observations.
                // Disagreement across them is further evidence of an edge.
                const float uncertainty=(1.f-t)*(1.f-t)*left.variance+t*t*right.variance+t*(1.f-t)*delta*delta;
                float fraction=fminf(left.support,right.support)*coherent/fmaxf(coherent+uncertainty,1.e-20f);
                const float target=fmaf(t,delta,left.value);
                if(directional) {
                    // Outside the common lobe, require confirmation from the
                    // receiver's own independent repeated path observations.
                    const float prior=fminf(left.directional_fraction,right.directional_fraction);
                    const auto raw=reinterpret_cast<const float4*>(p.cell_coefficients)[i];
                    const Vec3 mean{raw.x,raw.y,raw.z};
                    const float component=selected(mean,ch);
                    const float sigma=raw.w*component/fmaxf(luminance(mean),1.e-12f);
                    const float variance=sigma*sigma/float(count-1u);
                    const float shift=target-linear;
                    const float agree=fminf(1.f,fmaxf(0.f,shift*(component-linear)/fmaxf(fmaf(shift,shift,variance),1.e-20f)));
                    fraction*=fmaf(1.f-prior,agree,prior);
                } else {
                    // Adjacent projected cells can straddle the same surface's
                    // shadow. Give them less influence only when both the local
                    // R2 and the repeated original physical ray prefer the base.
                    const auto rgb=unpack(load_signal(reinterpret_cast<const Half4*>(p.scratch_b),i));
                    const Vec3 local{rgb.x,rgb.y,rgb.z};
                    const float z=selected(local,ch),ratio=z/fmaxf(luminance(local),1.e-12f);
                    const auto albedo=a.linear_albedo_rgb_diffuse_factor;
                    const float av=from_half(ch==0u?albedo.x:(ch==1u?albedo.y:albedo.z));
                    const float mod=sqrtf(fmaf(av,av,.01f));
                    const float measured=from_half(to_half(z*p.final_energy))*mod,scale=p.final_energy*mod;
                    const auto raw=reinterpret_cast<const float4*>(p.cell_coefficients)[i];
                    const Vec3 mean{raw.x,raw.y,raw.z};
                    const float fresh=selected(mean,ch),sigma=raw.w*fresh/fmaxf(luminance(mean),1.e-12f);
                    const float d=target-linear;
                    const float cost_local=d*(target+linear-2.f*measured);
                    const float cost_raw=d*(target+linear-2.f*fresh);
                    const float vlocal=fabsf(rgb.w)*ratio*ratio*scale*scale+uncertainty;
                    const float vraw=sigma*sigma/float(count-1u)+uncertainty;
                    const float evidence=fmaxf(0.f,fminf(cost_local/fmaxf(vlocal,1.e-12f),cost_raw/fmaxf(vraw,1.e-12f)));
                    fraction/=1.f+evidence;
                }
                desired=denoise_native::canonical_physical_transfer(fmaxf(0.f,fmaf(fraction,target-linear,linear)));
            }
        }
    }
    const float2 previous=history[i];
    const float limit=fabsf(baseline-previous.x);
    const float value=fminf(previous.y+limit,fmaxf(previous.y-limit,desired));
    if(!isfinite(value)){fail(p);return;}
    history[i]=make_float2(baseline,value);
    const auto *committed=reinterpret_cast<const denoise_native::TemporalSample32*>(p.regular_measurement_geometry);
    const bool angular_integral=(committed[i].metadata&(1u<<30u))!=0u;
    if(angular_integral) {
        // The moving finite-lobe panel already combines consecutive footprints.
        // Continue that two-exposure measure at the same registered ray. M's
        // squared-weight sum limits the additional vote of the previous panel.
        const float s2=fminf(1.f,fmaxf(from_half(committed[i].sigma_albedo.x),0.f));
        image[slot]=fmaf(s2/(1.f+s2),previous.y-value,value);
    } else if((a.metadata&511u)==(1u|(kSurfaceDiffuse<<2u))) {
        image[slot]=value;
    } else {
        auto *exposure=reinterpret_cast<float2*>(p.hold_state)+p.count;
        const auto old=exposure[i];
        const float next=fminf(old.y+1.f,32.f);
        const float t=value<=.04045f?value/12.92f:__powf((value+.055f)/1.055f,2.4f);
        const float current=-log1pf(-fminf(t,.99999994f));
        // M is already a weighted temporal mean. Its squared-weight sum is
        // its variance relative to a single exposure. Do not count each previous
        // reconstruction as another independent copy of all its source paths.
        const auto *geometry=reinterpret_cast<const denoise_native::TemporalSample32*>(p.regular_measurement_geometry);
        const float s2=fminf(1.f,fmaxf(0.f,from_half(geometry[i].sigma_albedo.x)));
        // With at most three effective samples the usual inverse variance
        // has no finite expectation. The two disjoint measured spatial rings
        // can then estimate the uncertainty of the shared optical prediction.
        const bool immature=geometry[i].sigma_albedo.x>=to_half(1.f/3.f);
        float previous_variance=1.f;
        if(!immature && p.previous_held_raw==HeldRawStorage::MovingExposure && count==2u && p.hold_previous &&
           (a.metadata&511u)==(1u|(kSurfaceNearSpecular<<2u))) {
            // The preceding display was itself reconstructed from a weighted
            // temporal measurement. Both squared-weight sums refer to the
            // same physical ray's single-exposure variance.
            const auto &prior=reinterpret_cast<const denoise_native::TemporalSample32*>(p.hold_previous)[i];
            previous_variance=fminf(1.f,fmaxf(from_half(prior.sigma_albedo.x),1.e-10f));
        }
        const float gain=previous_variance/fmaf(next-1.f,s2,previous_variance);
        float mean=fmaf(gain,current-old.x,old.x);
        if(immature && p.previous_held_raw==HeldRawStorage::MovingExposure && count==2u && p.field_measurements &&
           (a.metadata&511u)==(1u|(kSurfaceNearSpecular<<2u)))
            mean=first_held_footprint<Width>(p,i,a,ch,mean);
        exposure[i]=make_float2(mean,next);
        image[slot]=denoise_native::canonical_physical_transfer(fmaxf(mean,0.f));
    }
}
} // namespace

#define LFS_HELD_VIEW_CELLS(Name,Width,Height) \
extern "C" __global__ void lfs_build_held_view_cells_##Name(EncodedAtrousNativeParams p) { \
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x; \
    if(cell<p.atlas_cells)build_held_view_cell<Width>(p,cell); \
} \
extern "C" __global__ void lfs_present_held_view_cells_##Name(EncodedAtrousNativeParams p) { \
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x; \
    if(i<p.count)present_held_view_cell<Width,Height>(p,i); \
}
LFS_HELD_VIEW_CELLS(2k,4608,2048)
LFS_HELD_VIEW_CELLS(4k,11520,2160)
#undef LFS_HELD_VIEW_CELLS
