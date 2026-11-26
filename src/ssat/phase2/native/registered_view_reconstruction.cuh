// Estimate a physical angular radiance and its variance on a coarser grid in
// each actual aperture. The full-resolution reconstruction keeps its own
// measured spatial estimate and admits the field only on matching geometry.
namespace {
#include "registration_increment.cuh"

struct alignas(16) AngularFieldSample {
    unsigned long long surface;
    unsigned normal, metadata;
    Half4 radiance_and_sigma;
    unsigned short support, structural_sigma;
    unsigned site;
};
static_assert(sizeof(AngularFieldSample) == 32);
constexpr unsigned kFieldAddressMask=(1u<<27u)-1u;
__device__ __forceinline__ void finish_field_record(AngularFieldSample &record,unsigned site,unsigned metadata) {
    const Vec4 rounded=unpack(record.radiance_and_sigma);
    // The receiver only consumes one carrier. Retain the exact FP32 inverse
    // luminance of the rounded RGB once, before the record is read many times.
    const float inverse=1.f/fmaxf(luminance({rounded.x,rounded.y,rounded.z}),1.e-12f);
    record.metadata=__float_as_uint(inverse);
    record.site=site|(((metadata>>2u)&31u)<<27u);
}

template<int Width,int Height>
__device__ void build_diffuse_surface_field(const EncodedAtrousNativeParams &p,unsigned cell) {
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    const bool bands=(Width==4608 || Width==11520) && !p.resident_storage && !p.hold_same && p.register_motion && p.static_view_count>1u;
    const unsigned band_lane=bands?cell&7u:0u;
    if(bands)cell>>=3u;
#else
    constexpr bool bands=false;
    constexpr unsigned band_lane=0u;
#endif
    const int side=p.atlas_side;
    const int columns=(Width+3*side-1)/(3*side), rows=(Height+side-1)/side;
    const unsigned plane=unsigned(columns*rows);
    if(cell>=2u*plane)return;
    const unsigned layer=cell/plane,local=cell%plane;
    const unsigned aperture=layer==0?p.static_view_count/4u:3u*p.static_view_count/4u;
    const auto *sites=reinterpret_cast<const unsigned*>(p.atlas);
    const unsigned i=sites[aperture*plane+local];
    auto &destination=reinterpret_cast<AngularFieldSample*>(p.motion_cell_values)[p.motion_cell_count+cell];
    AngularFieldSample record{};
    record.site=i;
    if(i>=p.count){if(!band_lane){destination.site=i;destination.support=0;}return;}
    const auto *obs=reinterpret_cast<const AtlasObservation*>(p.packed_observation);
    const unsigned md=obs[i].metadata;
    if((md&511u)!=(1u|(kSurfaceDiffuse<<2u))) {
        // Empty sites still contribute their spatial mass to the receiver.
        if(!band_lane){destination.site=i;destination.support=0;}return;
    }
    const auto a=obs[i];record.surface=a.surface_key;record.normal=a.packed_geometric_normal_oct16;
    record.metadata=a.metadata;
    const float depth=a.hit_distance*reinterpret_cast<const float*>(p.static_inverse_ray_length)[i];
    if(!(depth>0.f)) {
        if(!band_lane)destination=record;return;
    }
    const float *origins=reinterpret_cast<const float*>(p.static_view_x);
    const auto *measurements=reinterpret_cast<const Half4*>(p.field_measurements);
    const Vec3 n=unpack_normal(a.packed_geometric_normal_oct16);
    const float shift=(1.f-1.f/depth)*p.atlas_inverse_dx;
    const int x=i%Width,y=i/Width;
    Vec3 sum{}; float mass=0.f,square=0.f,observation_noise=0.f,mean_noise=0.f;
#ifdef LFS_ENCODED_PLAIN_CURRENT
    constexpr unsigned samples=4;
#else
    constexpr unsigned samples=9;
#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if(!p.resident_storage && !p.hold_same && p.register_motion && p.static_view_count>1u) {
        // Integrate every native direction. Neighboring M estimates can reuse
        // the same projected history; accumulate a covariance upper bound
        // within each interval of the original quadrature.
        const unsigned last=p.static_view_count-1u;
        const float step=2.f/float(last);
        unsigned previous_band=0u;
        float band_sigma=0.f;
        const unsigned begin=bands?(band_lane*p.static_view_count+7u)/8u:0u;
        const unsigned end=bands?((band_lane+1u)*p.static_view_count+7u)/8u:p.static_view_count;
        for(unsigned v=begin;v<end;++v) {
            const unsigned band=min(7u,(v*8u)/p.static_view_count);
            if(band!=previous_band) {
                mean_noise=fmaf(band_sigma,band_sigma,mean_noise);
                band_sigma=0.f;previous_band=band;
            }
            const float xx=fmaf(origins[v]-origins[aperture],shift,float(x));
            if(!(xx>=-.5f && xx<float(Width)-.5f))continue;
            const unsigned col=__float2uint_rd((xx+.5f)/float(3*side));
            const unsigned j=sites[(v*rows+unsigned(y/side))*columns+col];
            if(j>=p.count)continue;
            const float g=atlas_geometry(a,obs[j],n);
            if(!(g>0.f))continue;
            const float w=(v==0u || v==last?.5f:1.f)*step*g;
            const auto m=unpack(load_signal(measurements,j));
            const Vec3 color{m.x,m.y,m.z};const float z=luminance(color),variance=fabsf(m.w);
            sum+=color*w;mass+=w;square=fmaf(w,z*z,square);
            observation_noise=fmaf(w,variance,observation_noise);
            band_sigma=fmaf(w,sqrtf(variance),band_sigma);
        }
        mean_noise=fmaf(band_sigma,band_sigma,mean_noise);
        if(bands) {
            const unsigned active=0xffu<<((unsigned(threadIdx.x)&31u)&~7u);
            #pragma unroll
            for(unsigned skip=4u;skip;skip>>=1u) {
                sum.x+=__shfl_down_sync(active,sum.x,skip,8);
                sum.y+=__shfl_down_sync(active,sum.y,skip,8);
                sum.z+=__shfl_down_sync(active,sum.z,skip,8);
                mass+=__shfl_down_sync(active,mass,skip,8);
                square+=__shfl_down_sync(active,square,skip,8);
                observation_noise+=__shfl_down_sync(active,observation_noise,skip,8);
                mean_noise+=__shfl_down_sync(active,mean_noise,skip,8);
            }
            if(band_lane)return;
        }
    } else
#endif
    {
    // Equal intervals of physical aperture angle. Every input is the measured
    // radiance of a calibrated ray on the same primary diffuse surface.
    #pragma unroll
    for(unsigned k=0;k<samples;++k) {
#ifdef LFS_ENCODED_PLAIN_CURRENT
        const unsigned v=(2u*k+1u)*p.static_view_count/(2u*samples);
#else
        // The four odd nodes are the actual midpoint rays used by Performance.
        // Add the interval boundaries to refine the same physical angular integral.
        const unsigned v=min(k*p.static_view_count/8u,p.static_view_count-1u);
        const float quadrature=(k&1u)?4.f:((k==0u||k==8u)?1.f:2.f);
#endif
        const float xx=fmaf(origins[v]-origins[aperture],shift,float(x));
        if(!(xx>=-.5f && xx<float(columns*3*side)-.5f))continue;
        const unsigned column=__float2uint_rd((xx+.5f)/float(3*side));
        const unsigned j=sites[(v*rows+unsigned(y/side))*columns+column];
        if(j>=p.count)continue;
#ifdef LFS_ENCODED_PLAIN_CURRENT
        const float w=(2.f/samples)*atlas_geometry(a,obs[j],n);
#else
        const float w=(quadrature/12.f)*atlas_geometry(a,obs[j],n);
#endif
        if(!(w>0.f))continue;
        const Vec4 m=unpack(load_signal(measurements,j));
        const Vec3 color{m.x,m.y,m.z};
        const float value=luminance(color),variance=fabsf(m.w);
        sum+=color*w;mass+=w;
        square=fmaf(w,value*value,square);
        observation_noise=fmaf(w,variance,observation_noise);
        mean_noise=fmaf(w*w,variance,mean_noise);
    }
    }
    if(mass>1.e-10f) {
        const float inverse=1.f/mass;
        const Vec3 color=sum*inverse;
        record.radiance_and_sigma=pack({color.x,color.y,color.z,sqrtf(mean_noise)*inverse});
        const float l=luminance(color);
        record.structural_sigma=to_half(sqrtf(fmaxf((square-observation_noise)*inverse-l*l,0.f)));
        record.support=to_half(mass);
        finish_field_record(record,i,a.metadata);
    }
    destination=record;
}

#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
#include "aperture_rows.cuh"
#endif
template<int Width, int Height>
__device__ void build_motion_field(const EncodedAtrousNativeParams &p) {
    static_assert(unsigned(Width)*unsigned(Height)<=kFieldAddressMask);
    const unsigned cell = blockIdx.x*blockDim.x+threadIdx.x;
    if (cell >= p.motion_cell_count) {
        unsigned first=p.motion_cell_count;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        if((Width==4608 || Width==11520) && !p.resident_storage && !p.hold_same && p.register_motion && p.static_view_count>1u) {
            first=(first+255u)&~255u;
            if(cell>=first)integrate_aperture_rows<Width,Height>(p,cell-first);
            return;
        }
#endif
        if(cell<first)return;
        build_diffuse_surface_field<Width,Height>(p,cell-first);
        return;
    }
    const auto *sites = reinterpret_cast<const unsigned *>(p.motion_cell_sites);
    auto *field = reinterpret_cast<AngularFieldSample *>(p.motion_cell_values);
    const unsigned i = sites[cell];
    AngularFieldSample record{};
    record.site = i;
#ifdef LFS_ENCODED_PLAIN_CURRENT
    if (i >= p.count) { field[cell]=record; return; }
#else
    if (i >= p.count) {field[cell].site=i;field[cell].support=0;return;}
#endif
    const auto *observations = reinterpret_cast<const AtlasObservation *>(p.packed_observation);
    const auto a = observations[i];
    if((a.metadata&511u)!=(1u|(kSurfaceNearSpecular<<2u))) {
        field[cell].site=i;field[cell].support=0;return;
    }
    record.surface=a.surface_key;
    record.normal=a.packed_geometric_normal_oct16;
    record.metadata=a.metadata;
    const auto *signal=reinterpret_cast<const Half4 *>(p.scratch_a);
    const auto center=unpack(signal[i]);
    const float depth=a.hit_distance*reinterpret_cast<const float *>(p.static_inverse_ray_length)[i];
    if (!register_motion_surface(a.metadata) || center.w<0.f || !(depth>0.f)) {
        field[cell]=record;
        return;
    }
    const int own=reinterpret_cast<const unsigned char *>(p.static_view_id)[i];
    const auto *origins=reinterpret_cast<const float *>(p.static_view_x);
    const auto *fine=reinterpret_cast<const unsigned *>(p.atlas);
    const int x=i%Width, y=i/Width, side=p.atlas_side;
    const int columns=(Width+3*side-1)/(3*side), rows=(Height+side-1)/side;
    const float inv_side=1.f/float(side), inv_width=inv_side/3.f;
    const int cy=__float2int_rd((y+.5f)*inv_side-.5f);
    const float parallax=(1.f-1.f/depth)*p.atlas_inverse_dx;
    const unsigned cls=(a.metadata>>2)&127u;
    const int extent=cls==kSurfaceDiffuse?1:2;
    const float rough=float((a.metadata>>9)&255u)*(1.f/255.f);
    const float sigma=fmaxf(1.f,rough*rough*(p.static_view_count-1)*p.atlas_inverse_span);
    const float angular_precision=cls==kSurfaceDiffuse?0.f:.5f/(sigma*sigma);
    const Vec3 n=unpack_normal(record.normal), rgb{center.x,center.y,center.z};
    const float center_luma=luminance(rgb);
    Vec3 total{};
    float strength=0.f, total_variance=0.f, second_moment=0.f, observation_noise=0.f;
    for (int dv=-extent; dv<=extent; ++dv) {
        const int view=own+dv;
        if (!dv || unsigned(view)>=p.static_view_count) continue;
        const float px=fmaf(origins[view]-origins[own],parallax,float(x));
        if (!(px>=-3.f*side && px<Width+3.f*side)) continue;
        const int cx=__float2int_rd((px+.5f)*inv_width-.5f);
        float norm=0.f, mass=0.f, variance=0.f, square=0.f, noise=0.f;
        Vec3 sum{};
        #pragma unroll
        for (int oy=0; oy<2; ++oy) {
            #pragma unroll
            for (int ox=0; ox<2; ++ox) {
                const int xx=cx+ox, yy=cy+oy;
                if (unsigned(xx)>=unsigned(columns) || unsigned(yy)>=unsigned(rows)) continue;
                const unsigned j=fine[(view*rows+yy)*columns+xx];
                if (j>=p.count) continue;
                const float dx=(float(j%Width)-px)*inv_width, dy=(float(j/Width)-float(y))*inv_side;
                const float spatial=__expf(-2.f*fmaf(dx,dx,dy*dy));
                norm+=spatial;
                const float geometric=atlas_geometry(a,observations[j],n);
                if (!(geometric>0.f)) continue;
                const auto m=unpack(signal[j]);
                const Vec3 value{m.x,m.y,m.z};
                // Form an independent angular estimate on projected geometry.
                // Its photometric confidence is evaluated once at the actual receiver.
                const float luma=luminance(value), var=fabsf(m.w);
                const float w=spatial*geometric;
                square=fmaf(w,luma*luma,square);noise=fmaf(w,var,noise);
                sum+=value*w; mass+=w; variance=fmaf(w*w,var,variance);
            }
        }
        if (norm>0.f) {
            const float weight=__expf(-dv*dv*angular_precision)/norm;
            total+=sum*weight; strength+=mass*weight;
            second_moment=fmaf(weight,square,second_moment);
            observation_noise=fmaf(weight,noise,observation_noise);
            total_variance=fmaf(weight*weight,variance,total_variance);
        }
    }
    // At this support even a maximal (500) input contributes less than one
    // FP32 code near white. A reciprocal of a subnormal sum can overflow.
    if (strength>1.e-10f) {
        const float inverse=1.f/strength;
        const Vec3 value=total*inverse;
        record.radiance_and_sigma=pack({value.x,value.y,value.z,sqrtf(total_variance)*inverse});
        const float l=luminance(value);
        record.structural_sigma=to_half(sqrtf(fmaxf((second_moment-observation_noise)*inverse-l*l,0.f)));
        record.support=to_half(strength);
        finish_field_record(record,i,a.metadata);
    }
    field[cell]=record;
}

#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
__device__ __forceinline__ float panel_material_response(const AtlasObservation &a,unsigned ch) {
    const auto albedo=a.linear_albedo_rgb_diffuse_factor;
    const float value=from_half(ch==0u?albedo.x:(ch==1u?albedo.y:albedo.z));
    return sqrtf(fmaf(value,value,.01f));
}

__device__ __forceinline__ float matching_spatial_base(const EncodedAtrousNativeParams &p,unsigned i,
        const AtlasObservation &a,float base,float response) {
    if((a.metadata&511u)!=(1u|(kSurfaceNearSpecular<<2u)))return base;
    const float original=from_half(reinterpret_cast<const Half4*>(p.scratch_b)[i].w);
    return from_half(to_half(original*p.final_energy))*response;
}

__device__ __forceinline__ float matching_spatial_base(const EncodedAtrousNativeParams &p,unsigned i,
        const AtlasObservation &a,float base) {
    if((a.metadata&511u)!=(1u|(kSurfaceNearSpecular<<2u)))return base;
    const float original=from_half(reinterpret_cast<const Half4*>(p.scratch_b)[i].w);
    return from_half(to_half(original*p.final_energy))*panel_material_response(a,i%3u);
}
#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
template<int Width,int Height>
__device__ float bilateral_diffuse_support(const EncodedAtrousNativeParams &p,unsigned i,float narrow,float wide) {
    const unsigned ch=i%3u;
    const auto *views=reinterpret_cast<const unsigned char*>(p.static_view_id);
    const int own=int(views[i]),last=int(p.static_view_count)-1;
    if(own<=0 || own>=last)return 0.f;
    const auto *obs=reinterpret_cast<const AtlasObservation*>(p.packed_observation);
    const auto a=obs[i];
    const float depth=a.hit_distance*reinterpret_cast<const float*>(p.static_inverse_ray_length)[i];
    if(!(depth>0.f))return 0.f;
    const auto normal=unpack_normal(a.packed_geometric_normal_oct16);
    const auto *m=reinterpret_cast<const Half4*>(p.field_measurements);
    const auto *sites=reinterpret_cast<const unsigned*>(p.atlas);
    const auto *origins=reinterpret_cast<const float*>(p.static_view_x);
    const float shift=(1.f-1.f/depth)*p.atlas_inverse_dx,d=narrow-wide;
    const int side=int(p.atlas_side),rows=(Height+side-1)/side,cols=(Width+3*side-1)/(3*side);
    const float iy=1.f/float(side),ix=iy/3.f;
    const int sy=int(i/Width),sx=int(i%Width);
    float admissible=1.f;
    for(int dv=-1;dv<=1;dv+=2) {
        // The two witnesses are different native apertures. None is the
        // receiver, and neither uses the wide reconstruction being tested.
        const int v=own+dv;
        const float px=fmaf(origins[v]-origins[own],shift,float(sx));
        if(!(px>=-.5f && px<Width-.5f))return 0.f;
        const float cx=(px+.5f)*ix-.5f,cy=(float(sy)+.5f)*iy-.5f;
        const int left=int(floorf(cx)),top=int(floorf(cy));
        const float fx=cx-left,fy=cy-top;
        float mass=0.f,sum=0.f,second=0.f,sigma=0.f;
        for(unsigned t=0;t<4u;++t) {
            const int x=left+int(t&1u),y=top+int(t>>1u);
            if(unsigned(x)>=unsigned(cols) || unsigned(y)>=unsigned(rows))continue;
            const unsigned j=sites[(v*rows+y)*cols+x];
            if(j>=p.count)continue;
            const float spatial=(t&1u?fx:1.f-fx)*(t&2u?fy:1.f-fy);
            const float w=spatial*atlas_geometry(a,obs[j],normal);
            if(!(w>0.f))continue;
            const auto sample=unpack(load_signal(m,j));
            const Vec3 rgb{sample.x,sample.y,sample.z};
            const float z=selected(rgb,ch),ratio=z/fmaxf(luminance(rgb),1.e-12f);
            sum=fmaf(w,z,sum);second=fmaf(w,z*z,second);mass+=w;
            sigma=fmaf(w,ratio*sqrtf(fabsf(sample.w)),sigma);
        }
        if(!(mass>0.f))return 0.f;
        const float inv=1.f/mass,g=sum*inv;
        const float noise=fmaf(sigma*inv,sigma*inv,fmaxf(fmaf(-g,g,second*inv),0.f));
        const float support=fminf(1.f,fmaxf(0.f,(g-wide)*d/fmaxf(fmaf(d,d,noise),1.e-20f)));
        admissible=fminf(admissible,mass*support);
        if(!(admissible>0.f))return 0.f;
    }
    return admissible;
}
#endif
#ifdef LFS_ENCODED_PLAIN_CURRENT
template<int Width,bool diffuse>
__device__ __forceinline__ float resolve_motion_field(const EncodedAtrousNativeParams &p,unsigned i,float base) {
    const auto a=reinterpret_cast<const AtlasObservation *>(p.packed_observation)[i];
    const float unchanged_support=base;
#else
template<int Width,bool SharedObservation=false>
__device__ __forceinline__ float resolve_motion_field(const EncodedAtrousNativeParams &p,unsigned i,float base,
        const AtlasObservation *center=nullptr,float center_support=0.f,float response=0.f) {
    const auto a=SharedObservation?*center:reinterpret_cast<const AtlasObservation*>(p.packed_observation)[i];
    const float unchanged_support=SharedObservation?center_support:matching_spatial_base(p,i,a,base);
#endif
    constexpr int Height=Width==4608?2048:(Width==11520?2160:4320);
    const unsigned channel=i%3u;
    const auto m=unpack(reinterpret_cast<const Half4 *>(p.field_measurements)[i]);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if(m.w<0.f && (a.metadata&511u)==(1u|(kSurfaceNearSpecular<<2u)))
        return update_registration_increment(p,i,a,unchanged_support,base);
#endif
    if(m.w<0.f) {
#ifdef LFS_ENCODED_PLAIN_CURRENT
        if constexpr(!diffuse)return base;
#endif
        auto *history=reinterpret_cast<float *>(p.stationary_weights);
        history[i]=0.f;
        if(p.hold_same) {
            history[p.count+i]=history[2ull*p.count+i]=0.f;
            set_registration_age(p,history+3ull*p.count+i,1.f);
        }
        return base;
    }
    float result=base;
    if (m.w>=0.f) {
        const int v=reinterpret_cast<const unsigned char *>(p.static_view_id)[i];
#ifndef LFS_ENCODED_PLAIN_CURRENT
        const bool diffuse=(a.metadata&511u)==(1u|(kSurfaceDiffuse<<2u));
#endif
        const int layer=int(v)>=int(p.static_view_count/2u)?1:0;
        const int reference=layer?3u*p.static_view_count/4u:p.static_view_count/4u;
        const int side=diffuse?int(p.atlas_side):2*int(p.atlas_side);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        const bool reduced_rows=Width==11520 && diffuse && !p.resident_storage &&
            !p.hold_same && p.register_motion && p.static_view_count>1u;
#else
        constexpr bool reduced_rows=false;
#endif
        const int vertical_side=reduced_rows?2*side:side;
        const int cols=(Width+3*side-1)/(3*side),rows=(Height+vertical_side-1)/vertical_side;
        const int x=i%Width,y=i/Width;
        const float inv_side=1.f/float(side),inv_width=inv_side/3.f;
        const float inv_height=reduced_rows?.5f*inv_side:inv_side;
        float query_x=float(x);
        if(diffuse) {
            const float depth=a.hit_distance*reinterpret_cast<const float*>(p.static_inverse_ray_length)[i];
            if(!(depth>0.f))return update_registration_increment(p,i,a,base,base);
            const auto *u=reinterpret_cast<const float*>(p.static_view_x);
            query_x=fmaf(u[reference]-u[v],(1.f-1.f/depth)*p.atlas_inverse_dx,float(x));
            if(!(query_x>=-3.f*side && query_x<Width+3.f*side))return update_registration_increment(p,i,a,base,base);
        }
        const int cx=__float2int_rd((query_x+.5f)*inv_width-.5f),cy=__float2int_rd((y+.5f)*inv_height-.5f);
        const Vec3 rgb{m.x,m.y,m.z};
        const float z=selected(rgb,channel),scale=z/fmaxf(luminance(rgb),1.e-12f);
        const float center_var=m.w*scale*scale;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        const float near_information=!diffuse && center_var>1.e-12f?measured_density_information(p,i):0.f;
        const float near_inverse_twice_center=.5f/fmaxf(center_var,1.e-12f);
#endif
#ifdef LFS_ENCODED_PLAIN_CURRENT
        const Vec3 normal=unpack_normal(a.packed_geometric_normal_oct16);
#endif
        const auto *field=reinterpret_cast<const AngularFieldSample *>(p.motion_cell_values);
        float nominal=0.f, mass=0.f, total=0.f;
        #pragma unroll
        for (int oy=0; oy<2; ++oy) {
            #pragma unroll
            for (int ox=0; ox<2; ++ox) {
                const int xx=cx+ox,yy=cy+oy;
                if (unsigned(xx)>=unsigned(cols)||unsigned(yy)>=unsigned(rows)) continue;
                const size_t address=diffuse?size_t(p.motion_cell_count)+(size_t(layer)*rows+yy)*cols+xx:
                    (size_t(yy)*cols+xx)*p.static_view_count+v;
                const auto q=field[address];
                const unsigned site=q.site&kFieldAddressMask;
                if(site>=p.count)continue;
                const float dx=(float(site%Width)-query_x)*inv_width;
                const float dy=(float(site/Width)-float(y))*inv_height;
                const float spatial=__expf(-2.f*fmaf(dx,dx,dy*dy));
                nominal+=spatial;
                if (!q.support || a.surface_key!=q.surface || (((a.metadata>>2u)&31u)!=(q.site>>27u))) continue;
                float geometry=1.f;
                if (a.packed_geometric_normal_oct16!=q.normal) {
                    #ifndef LFS_ENCODED_PLAIN_CURRENT
                    const auto normal=unpack_normal(a.packed_geometric_normal_oct16);
#endif
                    geometry=pow32(fmaxf(dot(normal,unpack_normal(q.normal)),0.f));
                }
                const auto rgb=q.radiance_and_sigma;
                const float component=from_half(channel==0u?rgb.x:(channel==1u?rgb.y:rgb.z));
                const float chroma=component*__uint_as_float(q.metadata);
                const float s=chroma*from_half(rgb.w),delta=component-z;
                const float model=from_half(q.structural_sigma)*chroma;
                const float measurement=fmaxf(center_var+s*s,1.e-8f);
                const float confidence=measurement/(measurement+model*model);
#ifdef LFS_ENCODED_PLAIN_CURRENT
                // Compare Gaussian densities at the measured variances. A very
                // noisy angular field must not pass solely because its error bar is wide.
                const float determinant=center_var>1.e-12f?sqrtf(2.f*center_var/measurement):1.f;
                const float w=spatial*geometry*from_half(q.support)*confidence*determinant*__expf(-.5f*delta*delta/measurement);
#else
                const bool moving_plane=Width==11520 && diffuse && !p.hold_same && !p.resident_storage;
                float normalization,exponent;
                if(moving_plane) {
                    const float inverse=regular_chart_reciprocal(measurement);
                    normalization=center_var>1.e-12f?sqrtf((2.f*center_var)*inverse):1.f;
                    exponent=(-.5f*delta*delta)*inverse;
                } else {
                    normalization=diffuse?(center_var>1.e-12f?sqrtf(2.f*center_var/measurement):1.f):
                        (near_information>0.f?calibrated_density_ratio(measurement,near_inverse_twice_center,near_information):1.f);
                    exponent=-.5f*delta*delta/measurement;
                }
                const float w=spatial*geometry*from_half(q.support)*confidence*normalization*__expf(exponent);
#endif
                mass+=w;total=fmaf(w,component,total);
            }
        }
        if (mass>0.f) {
#ifdef LFS_ENCODED_PLAIN_CURRENT
            const auto ab=a.linear_albedo_rgb_diffuse_factor;
            const float albedo=from_half(channel==0u?ab.x:(channel==1u?ab.y:ab.z));
            const float modulation=p.final_energy*sqrtf(fmaf(albedo,albedo,.01f));
#else
            const float modulation=p.final_energy*(SharedObservation?response:panel_material_response(a,channel));
#endif
            result=(base*nominal+total*modulation)/(nominal+mass);
        }
    }
    if (!isfinite(result)) {fail(p);return 0.f;}
#ifdef LFS_ENCODED_PLAIN_CURRENT
    if constexpr(!diffuse)return result;
#endif
    return update_registration_increment(p,i,a,unchanged_support,result);
}
#ifdef LFS_ENCODED_PLAIN_CURRENT

#endif
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
#include "repeat_field_canonical.cuh"
#endif
}
#define LFS_MOTION_FIELD(Name,Width,Height) \
extern "C" __global__ void lfs_build_epipolar_##Name(EncodedAtrousNativeParams p) {build_motion_field<Width,Height>(p);}
LFS_MOTION_FIELD(2k,4608,2048)
LFS_MOTION_FIELD(4k,11520,2160)
LFS_MOTION_FIELD(8k,23040,4320)
#undef LFS_MOTION_FIELD
