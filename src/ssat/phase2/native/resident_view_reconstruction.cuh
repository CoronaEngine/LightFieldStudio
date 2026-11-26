// Moving-camera reconstruction. The current aperture estimate already includes its
// spatial filter. Add measurements from adjacent apertures at the projected first hit.
namespace {
template<int Width, int Height>
__device__ void finish_registered_site(const EncodedAtrousNativeParams &p) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p.count)return;
    auto *residuals=reinterpret_cast<unsigned short*>(p.resident_angular_residuals);
    // Every site gets a defined value after a moving frame, including clipped
    // apertures, occlusions and surfaces which do not accept registration.
    if(residuals && p.register_motion==1u)residuals[i]=0u;
    const auto *observations=reinterpret_cast<const AtlasObservation*>(p.packed_observation);
    const auto center=observations[i];
    if(!register_motion_surface(center.metadata))return;
    if(*reinterpret_cast<const unsigned*>(p.frame_status))return;
    const unsigned ch=i%3u, slot=4u*(i/3u)+ch;
    auto *image=reinterpret_cast<float*>(p.canonical);
    const float physical=image[slot];
    if(p.register_motion==2u && residuals){
        const float transported=from_half(residuals[i]);
        // Previously removed light scales with the available center radiance.
        // Additional angular radiance is an absolute, nonnegative observation.
        const float corrected=transported<0.f?fmaf(physical,transported,physical):physical+transported;
        image[slot]=denoise_native::canonical_physical_transfer(fmaxf(0.f,corrected));
        return;
    }
    const auto *measurement=reinterpret_cast<const Half4*>(p.scratch_a);
    const Vec4 current=unpack(measurement[i]);
    const float depth=center.hit_distance*reinterpret_cast<const float*>(p.static_inverse_ray_length)[i];
    if(current.w<0.f || !(depth>0.f)) {
        image[slot]=denoise_native::canonical_physical_transfer(physical);
        return;
    }
    const float parallax=(1.f-1.f/depth)*p.atlas_inverse_dx;
    const auto *origins=reinterpret_cast<const float*>(p.static_view_x);
    const auto *lookup=reinterpret_cast<const unsigned*>(p.atlas);
    const auto *filtered=reinterpret_cast<const Half4*>(p.scratch_b);
    const int own=reinterpret_cast<const unsigned char*>(p.static_view_id)[i];
    const int x=i%Width,y=i/Width;
    const float inverse_side=1.f/float(p.atlas_side);
    const float inverse_width=inverse_side/3.f;
    const int cols=(Width+3*p.atlas_side-1)/(3*p.atlas_side);
    const int rows=(Height+p.atlas_side-1)/p.atlas_side;
    const int cy=int(floorf((float(y)+.5f)*inverse_side-.5f));
    const auto normal=unpack_normal(center.packed_geometric_normal_oct16);
    const unsigned cls=(center.metadata>>2u)&127u;
    const int span=cls==kSurfaceDiffuse?1:2;
    const float rough=float((center.metadata>>9u)&255u)/255.f;
    const float sigma=fmaxf(1.f,rough*rough*(p.static_view_count-1)*p.atlas_inverse_span);
    const float inv_sigma=cls==kSurfaceDiffuse?0.f:.5f/(sigma*sigma);
    const Vec3 rgb{current.x,current.y,current.z};
    const float lc=luminance(rgb), z=selected(rgb,ch);
    const float scale=z/fmaxf(lc,1.e-20f), center_variance=current.w*scale*scale;
    const float inv_phi=1.f/(32.f*sqrtf(fmaxf(current.w,1.e-4f))+1.e-3f);
    float support=0.f,external=0.f;
    const auto* inverse_length=reinterpret_cast<const float*>(p.static_inverse_ray_length);
    unsigned measured_sides=0u, unblocked_sides=0u, incomplete_sides=0u;
    for(int dv=-span;dv<=span;++dv) {
        const int view=own+dv;
        if(!dv)continue;
        const unsigned side_bit=dv<0?1u:2u;
        if(view<0 || view>=int(p.static_view_count)) {
            incomplete_sides|=side_bit;
            continue;
        }
        const float projected=fmaf(origins[view]-origins[own],parallax,float(x));
        if(!(projected>=-3.f*p.atlas_side && projected<float(Width)+3.f*p.atlas_side)) {
            incomplete_sides|=side_bit;
            continue;
        }
        const int cx=int(floorf((projected+.5f)*inverse_width-.5f));
        float nominal=0.f,mass=0.f,sum=0.f;
        bool all_front_surfaces=true;
        const bool inside=cx>=0 && cx+1<cols && cy>=0 && cy+1<rows;
        #pragma unroll
        for(int oy=0;oy<2;++oy) {
            #pragma unroll
            for(int ox=0;ox<2;++ox) {
                const int ax=cx+ox,ay=cy+oy;
                if(ax<0 || ay<0 || ax>=cols || ay>=rows)continue;
                const unsigned cell=(view*rows+ay)*cols+ax;
                unsigned j;
                if(p.aperture_cell_bits){
                    const unsigned bits=p.aperture_cell_bits, entries=32u/bits,mask=(1u<<bits)-1u;
                    const unsigned local=(lookup[cell/entries]>>((cell%entries)*bits))&mask;
                    if(local==mask)continue;
                    const unsigned cell_width=3u*p.atlas_side;
                    j=(ay*p.atlas_side+local/cell_width)*Width+ax*cell_width+local%cell_width;
                } else j=lookup[cell];
                if(j>=p.count)continue;
                const float dx=(float(j%Width)-projected)*inverse_width;
                const float dy=(float(j/Width)-float(y))*inverse_side;
                const float spatial=__expf(-2.f*(dx*dx+dy*dy));
                nominal+=spatial;
                const auto neighbor=observations[j];
                if(all_front_surfaces) {
                    // For parallel aperture rays this is distance along the common optical axis,
                    // expressed relative to the calibrated focal plane.
                    const bool different_surface=(neighbor.metadata&3u)==1u && neighbor.surface_key!=center.surface_key;
                    const float neighbor_depth=different_surface?neighbor.hit_distance*inverse_length[j]:0.f;
                    all_front_surfaces=neighbor_depth>0.f && neighbor_depth<depth;
                }
                const float geometry=atlas_geometry(center,neighbor,normal);
                if(!(geometry>0.f))continue;
                const Vec4 value=unpack(filtered[j]);
                const Vec3 source{value.x,value.y,value.z};
                const float luma=luminance(source),component=selected(source,ch);
                const float ratio=component/fmaxf(luma,1.e-20f);
                const float variance=center_variance+value.w*ratio*ratio;
                const float delta=component-z;
                const float residual=value.w>=0.f?.5f*delta*delta/fmaxf(variance,1.e-20f)
                                                 :fabsf(luma-lc)*inv_phi;
                // Relative normalization of the difference of two measured Gaussian estimates.
                // A large donor variance must not make a rare observation more credible.
                const float amplitude=p.resident_angular_residuals && value.w>=0.f && variance>0.f
                    ? sqrtf(2.f*center_variance/variance) : 1.f;
                const float weight=spatial*geometry*__expf(-residual)*amplitude;
                mass+=weight;
                sum+=component*weight;
            }
        }
        if(!inside || !(nominal>0.f))incomplete_sides|=side_bit;
        else {
            measured_sides|=side_bit;
            if(!all_front_surfaces)unblocked_sides|=side_bit;
        }
        if(nominal>0.f) {
            const float weight=__expf(-float(dv*dv)*inv_sigma)/nominal;
            support+=mass*weight;
            external+=sum*weight;
        }
    }
    // A measured foreground occluder on every available representative on one
    // side invalidates one-sided angular extrapolation. Missing views, missing
    // samples and clipped footprints carry no evidence of foreground occlusion.
    if(measured_sides==3u && incomplete_sides==0u && unblocked_sides!=3u) {
        image[slot]=denoise_native::canonical_physical_transfer(physical);
        return;
    }
    const auto a=center.linear_albedo_rgb_diffuse_factor;
    const float albedo=from_half(ch==0u?a.x:(ch==1u?a.y:a.z));
    const float modulation=p.final_energy*sqrtf(fmaf(albedo,albedo,.01f));
    const float result=(physical+external*modulation)/(1.f+support);
    if(!isfinite(result)){fail(p);return;}
    if(residuals){
        const float change=result-physical;
        // Physical estimates are nonnegative, so the removed fraction is in [0,1].
        const auto delta=to_half(change<0.f?fmaxf(-1.f,change/physical):change);
        residuals[i]=(delta&0x7c00u)!=0x7c00u?delta:0u;
    }
    image[slot]=denoise_native::canonical_physical_transfer(result);
}
}
#define LFS_REGISTERED(Name,Width,Height) \
extern "C" __global__ void lfs_finish_registered_##Name(EncodedAtrousNativeParams p) {finish_registered_site<Width,Height>(p);}
LFS_REGISTERED(2k,4608,2048)
LFS_REGISTERED(4k,11520,2160)
LFS_REGISTERED(8k,23040,4320)
#undef LFS_REGISTERED
