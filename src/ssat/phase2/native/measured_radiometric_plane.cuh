template<int Width>
__device__ __forceinline__ Vec3 physical_plane_point(const EncodedAtrousNativeParams &p,unsigned i,float hit) {
    const auto view=reinterpret_cast<const unsigned char*>(p.static_view_id)[i];
    const float u=reinterpret_cast<const float*>(p.static_view_x)[view];
    const float depth=hit*reinterpret_cast<const float*>(p.static_inverse_ray_length)[i];
    return {fmaf(reinterpret_cast<const float*>(p.static_focal_x)[i%Width]-u,depth,u),
            reinterpret_cast<const float*>(p.static_focal_y)[i/Width]*depth,p.focal_distance*depth};
}
template<int SW,int SH,int Threads>
__device__ __forceinline__ bool one_ray_plane(const SharedGuide *guides,const Vec3 *normals,int reference,int thread) {
    const auto ref=load_tile_guide(guides,normals,reference);
    const auto n=guide_normal(ref);
    bool covered=true;
    for(int j=thread;j<SW*SH;j+=Threads) {
        const auto g=load_tile_guide(guides,normals,j);
        const auto m=guide_normal(g),delta=g.point-ref.point;
        const float perpendicular=fabsf(dot(n,delta));
        // Include the error of oct16 normals and of subtracting two floating
        // primary hit positions. Separated parallel faces fail this bound.
        const float extent=fabsf(delta.x)+fabsf(delta.y)+fabsf(delta.z);
        const float numeric=8.f*1.1920929e-7f*(1.f+fabsf(g.point.x)+fabsf(g.point.y)+fabsf(g.point.z));
        covered=covered && m.x==n.x && m.y==n.y && m.z==n.z && perpendicular<=extent*(4.f/65535.f)+numeric;
    }
    return __syncthreads_and(covered);
}

template<int SW,bool Statistical,bool ChangingAlbedo>
__device__ __forceinline__ Half4 integrate_radiometric_plane(const Half4 *signal,const float *luma,
        const SharedGuide *guides,const Vec3 *normals,int sx,int sy) {
    const int j=sy*SW+sx;const auto center=unpack(signal[j]);const float z=luma[j];
    Half4 albedo{};
    Vec3 ac{};
    if constexpr(ChangingAlbedo) {
        albedo=load_tile_guide(guides,normals,j).albedo;
        ac={from_half(albedo.x),from_half(albedo.y),from_half(albedo.z)};
    }
    const float inv_phi=1.f/(32.f*sqrtf(fmaxf(fabsf(center.w),1.e-4f))+1.e-3f);
    Vec3 sum{};float mass=0.f,noise=0.f;bool known=!Statistical || center.w>=0.f;
    for(int dy=-2;dy<=2;++dy) {
        #pragma unroll
        for(int dx=-2;dx<=2;++dx) {
            const int q=(sy+dy)*SW+sx+dx;const auto v=unpack(signal[q]);
            const float kx=dx==0?.375f:((dx==-1 || dx==1)?.25f:.0625f);
            const float ky=dy==0?.375f:((dy==-1 || dy==1)?.25f:.0625f);
            float w=kx*ky;
            if(dx || dy) {
                const float d=z-luma[q];
                float contrast=(Statistical && center.w>=0.f && v.w>=0.f)?
                    .5f*d*d/fmaxf(fabsf(center.w)+fabsf(v.w),1.e-20f):fabsf(d)*inv_phi;
                if constexpr(ChangingAlbedo) {
                    const auto as=load_tile_guide(guides,normals,q).albedo;
                    if(as.x!=albedo.x || as.y!=albedo.y || as.z!=albedo.z) {
                        const auto da=ac-Vec3{from_half(as.x),from_half(as.y),from_half(as.z)};
                        contrast+=4.f*sqrtf(dot(da,da));
                    }
                }
                w*=__expf(-contrast);
            }
            sum+=Vec3{v.x,v.y,v.z}*w;mass+=w;noise=fmaf(w*w,fabsf(v.w),noise);
            if(Statistical && v.w<0.f && w>0.f)known=false;
        }
    }
    const float inv=1.f/fmaxf(mass,1.e-4f);const Vec3 mean=sum*inv;
    const float var=fmaxf(mass*mass>=1.e-4f?noise*inv*inv:noise/1.e-4f,1.e-4f);
    return pack({mean.x,mean.y,mean.z,known?var:-var});
}


__device__ __forceinline__ float plane_inverse_response(float variance) {
    return 1.f/(32.f*sqrtf(fmaxf(variance,1.e-4f))+1.e-3f);
}

template<bool Horizontal,bool Gray,bool CachedResponse>
__device__ __forceinline__ Half4 integrate_plane_axis(const Half4 *filtered,Half4 *strip,unsigned short *masses,
        const float *luma,const float *reflectance,const float *response,int sx,int sy) {
    constexpr int width=38;
    const int i=sy*width+sx;
    const float z=luma[i];
    float inverse_phi;
    if constexpr(CachedResponse)inverse_phi=response[i];
    else inverse_phi=plane_inverse_response(from_half(filtered[i].w));
    Vec3 sum{};float mass=0.f,variance=0.f;
    #pragma unroll
    for(int dt=-2;dt<=2;++dt) {
        const int j=Horizontal?i+dt:i+dt*width;
        const int stripe=(sy-1+(Horizontal?0:dt))*32+sx-3;
        const float b=dt==0?.375f:((dt==-1 || dt==1)?.25f:.0625f);
        float contrast=fabsf(z-luma[j])*inverse_phi;
        if constexpr(Gray)contrast+=6.92820323028f*fabsf(reflectance[i]-reflectance[j]);
        float w=dt==0?b:b*__expf(-contrast);
        if constexpr(!Horizontal)w*=from_half(masses[stripe]);
        const auto sample=unpack(Horizontal?filtered[j]:strip[stripe]);
        sum+=Vec3{sample.x,sample.y,sample.z}*w;
        mass+=w;variance=fmaf(w*w,sample.w,variance);
    }
    if constexpr(Horizontal)masses[(sy-1)*32+sx-3]=to_half(mass);
    const float inv=1.f/fmaxf(mass,1.e-4f);
    const auto rgb=sum*inv;
    return pack({rgb.x,rgb.y,rgb.z,fmaxf(variance*inv*inv,Horizontal?0.f:1.e-4f)});
}

template<int Width,bool Gray,bool CachedResponse=false>
__device__ __forceinline__ void plane_exposure_axes(const EncodedAtrousNativeParams &p,const Half4 *filtered,Half4 *bank,
                                                   float *luma,const float *reflectance,int thread,const float *response=nullptr) {
    // The entire rounded prefilter remains in the other shared bank.
    // Store its luminance in the unused point coordinates, across both passes.
    auto *masses=reinterpret_cast<unsigned short*>(bank+640);
    static_assert(640u*(sizeof(Half4)+sizeof(unsigned short))<=38u*22u*sizeof(Half4));
    if constexpr(!CachedResponse) {
        for(unsigned k=thread;k<36u*20u;k+=512u) {
            const unsigned i=(k/36u+1u)*38u+k%36u+1u;
            const auto v=unpack(filtered[i]);luma[i]=luminance({v.x,v.y,v.z});
        }
        __syncthreads();
    }
    for(unsigned k=thread;k<640u;k+=512u) {
        const int sx=int(k%32u)+3,sy=int(k/32u)+1;
        bank[k]=integrate_plane_axis<true,Gray,CachedResponse>(filtered,bank,masses,luma,reflectance,response,sx,sy);
    }
    __syncthreads();
    const int sx=3+int(threadIdx.x),sy=3+int(threadIdx.y);
    auto signal=integrate_plane_axis<false,Gray,CachedResponse>(filtered,bank,masses,luma,reflectance,response,sx,sy);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    signal.w|=0x8000u;
#endif
    const unsigned i=(blockIdx.y*16u+threadIdx.y)*Width+blockIdx.x*32u+threadIdx.x;
    store_signal(reinterpret_cast<Half4*>(p.scratch_b),i,signal);
}

__device__ __forceinline__ float plane_pair_weight(float a,float b,float kernel) {
    return kernel*__expf(-3.5f*fabsf(a-b)/(a+b+.002f));
}

__device__ __forceinline__ void build_plane_pair_weights(float *edges,const float *luma,unsigned thread) {
    constexpr unsigned sw=38,sh=22,count=sw*sh;
    for(unsigned j=thread;j<count;j+=512u) {
        const unsigned x=j%sw,y=j/sw;
        const bool east=x+1u<sw,down=y+1u<sh,west=x!=0u;
        const float z=luma[j];
        if(east)edges[j]=plane_pair_weight(z,luma[j+1u],.125f);
        if(down)edges[count+j]=plane_pair_weight(z,luma[j+sw],.125f);
        if(east && down)edges[2u*count+j]=plane_pair_weight(z,luma[j+sw+1u],.0625f);
        if(west && down)edges[3u*count+j]=plane_pair_weight(z,luma[j+sw-1u],.0625f);
    }
    __syncthreads();
}


template<bool SharedEdges=false>
__device__ __forceinline__ void prefilter_plane_exposure(Half4 *raw,Half4 *filtered,
        float *raw_luma,float *filtered_luma,unsigned thread,const float *edges=nullptr,float *response=nullptr) {
    constexpr int sw=38,count=38*22;
    Half4 pending[2]{};
    #pragma unroll
    for(unsigned pass=0;pass<2;++pass) {
        const unsigned k=thread+512u*pass;if(k>=36*20)continue;
        const int sx=1+int(k%36),sy=1+int(k/36),j=sy*sw+sx;
        const auto v=unpack(raw[j]);const float z=raw_luma[j];
        Vec3 sum{},neighbors{};float mass=0.f,first=0.f,second=0.f;
        #pragma unroll
        for(int dy=-1;dy<=1;++dy) {
            #pragma unroll
            for(int dx=-1;dx<=1;++dx) {
                const int q=(sy+dy)*sw+sx+dx;const auto h=unpack(raw[q]);const Vec3 rgb{h.x,h.y,h.z};
                const float lum=raw_luma[q],kernel=dx && dy?.0625f:(dx || dy?.125f:.25f);
                float w;
                if constexpr(SharedEdges) {
                    if(dy==0)w=dx==0?.25f:edges[dx>0?j:q];
                    else if(dy>0)w=edges[(dx==0?1:(dx>0?2:3))*count+j];
                    else w=edges[(dx==0?1:(dx<0?2:3))*count+q];
                } else w=(dx || dy)?plane_pair_weight(z,lum,kernel):kernel;
                sum+=rgb*w;mass+=w;first=fmaf(kernel,lum,first);second=fmaf(kernel,lum*lum,second);
                if(dx || dy)neighbors+=rgb;
            }
        }
        Vec3 base{v.x,v.y,v.z};
        const float avg=fmaxf(safe_luminance(neighbors*.125f),.01f),isolation=z/avg;
        if(isolation>3.f) {
            const float tv=fmaxf(z,.01f)+(20.f+(10.f-20.f)/48.f)*.5f;
            const float limit=fmaxf(tv+.7f*(avg*3.f-tv),.1f);
            if(z>limit) {
                const float compressed=1.f+asinhf((z/limit-1.f)*2.f)*.425f;
                base=base*fminf(limit*compressed/fmaxf(z,1.e-3f),1.f);
            }
        }
        const auto mean=clamp3(base+(sum/fmaxf(mass,1.e-4f)-base)*.24f,0.f,500.f);
        const float variation=fmaxf(second-first*first,z*.08f);
        const float variance=fmaxf(v.w+.3f*(fmaxf(variation,.001f)-v.w),.001f);
        pending[pass]=pack({mean.x,mean.y,mean.z,variance});
    }
    __syncthreads();
    #pragma unroll
    for(unsigned pass=0;pass<2;++pass) {
        const unsigned k=thread+512u*pass;if(k>=36*20)continue;
        const unsigned j=(1u+k/36u)*sw+1u+k%36u;
        const auto v=unpack(pending[pass]);
        filtered[j]=pending[pass];
        filtered_luma[j]=luminance({v.x,v.y,v.z});
        if constexpr(SharedEdges)response[j]=plane_inverse_response(v.w);
    }
    __syncthreads();
}

template<int Width,int Height>
__device__ __forceinline__ void plane_exposure(const EncodedAtrousNativeParams &p,Half4 *raw,Half4 *filtered,
                                               SharedGuide *guides,Vec3 *normals) {
    constexpr int sw=38,count=38*22;const unsigned thread=threadIdx.y*32u+threadIdx.x;
    float *raw_luma=reinterpret_cast<float*>(filtered),*filtered_luma=reinterpret_cast<float*>(raw);
    const auto a=load_tile_guide(guides,normals,3*sw+3).albedo;
    bool same=true,diffuse=true;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    bool gray_glossy=true;
#endif
    const int bx=int(blockIdx.x)*32-3,by=int(blockIdx.y)*16-3;
    const auto *observations=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation);
    for(unsigned j=thread;j<count;j+=512) {
        if constexpr(Width==11520) {
            const unsigned i=(by+int(j/sw))*Width+bx+int(j%sw);
            diffuse &= (observations[i].metadata&511u)==(1u|(kSurfaceDiffuse<<2u));
        }
        const auto b=load_tile_guide(guides,normals,j).albedo;
        same=same && a.x==b.x && a.y==b.y && a.z==b.z;
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        if constexpr(Width==4608) {
            const unsigned i=(by+int(j/sw))*Width+bx+int(j%sw);
            gray_glossy=gray_glossy && (observations[i].metadata&511u)==(1u|(kSurfaceGlossy<<2u)) &&
                        b.x==b.y && b.y==b.z;
        }
#endif
        const auto v=unpack(raw[j]);raw_luma[j]=safe_luminance({v.x,v.y,v.z});
    }
    const bool constant=__syncthreads_and(same);
    bool use_axes=false;
    if constexpr(Width==11520)use_axes=constant && __syncthreads_and(diffuse);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    if constexpr(Width==4608)use_axes=__syncthreads_and(gray_glossy);
#endif

    prefilter_plane_exposure(raw,filtered,raw_luma,filtered_luma,thread);
    if(use_axes) {
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
        if constexpr(Width==4608) {
            // All albedos have the same three half channels on this physical
            // plane. The original 4*RGB distance is exactly 4*sqrt(3)*|a-b|.
            auto *reflectance=reinterpret_cast<float*>(normals);
            for(unsigned j=thread;j<count;j+=512)reflectance[j]=from_half(guides[j].albedo.x);
            __syncthreads();
            plane_exposure_axes<Width,true>(p,filtered,raw,reinterpret_cast<float*>(guides),reflectance,thread);
        } else
#endif
        plane_exposure_axes<Width,false>(p,filtered,raw,reinterpret_cast<float*>(guides),nullptr,thread);
        return;
    }
    const int x=3+int(threadIdx.x),y=3+int(threadIdx.y);
    Half4 measured=constant?integrate_radiometric_plane<sw,false,false>(filtered,filtered_luma,guides,normals,x,y):
        integrate_radiometric_plane<sw,false,true>(filtered,filtered_luma,guides,normals,x,y);
#ifdef LFS_ENCODED_GLOSSY_MEASUREMENT
    measured.w|=0x8000u;
#endif
    const unsigned i=(blockIdx.y*16u+threadIdx.y)*Width+blockIdx.x*32u+threadIdx.x;
    store_signal(reinterpret_cast<Half4*>(p.scratch_b),i,measured);
}

template<int Width,int Height,bool CaptureRaw>
__device__ __forceinline__ bool measure_uniform_diffuse_plane(
        const EncodedAtrousNativeParams &p,Half4 *raw,Half4 *filtered,SharedGuide *guides) {
    constexpr int sw=38,sh=22;
    const int bx=int(blockIdx.x)*32-3,by=int(blockIdx.y)*16-3;
    if(bx<0 || by<0 || bx+sw>Width || by+sh>Height)return false;
    const unsigned lane=threadIdx.y*32u+threadIdx.x;
    const auto *observations=reinterpret_cast<const denoise_native::PackedSurfaceObservationV1*>(p.packed_observation);
    const auto *views=reinterpret_cast<const unsigned char*>(p.static_view_id);
    const auto *inverse=reinterpret_cast<const float*>(p.static_inverse_ray_length);
    const unsigned reference=(by+3)*Width+bx+3;
    const auto ref=observations[reference];
    const auto a=ref.linear_albedo_rgb_diffuse_factor;
    constexpr unsigned diffuse=1u|(kSurfaceDiffuse<<2u);
    if((ref.metadata&511u)!=diffuse || ref.primitive_id==~0u || ref.surface_key==~std::uint64_t{0u} ||
       !ssat_native::packed_half_is_finite(a.x) || !ssat_native::packed_half_is_finite(a.y) ||
       !ssat_native::packed_half_is_finite(a.z))return false;

    const auto normal=local_normal(p,unpack_normal(ref.packed_geometric_normal_oct16));
    if(!finite3(normal))return false;
    bool coplanar=true;
    for(unsigned j=lane;j<sw*sh;j+=512u) {
        const unsigned i=(by+j/sw)*Width+bx+j%sw;
        const auto o=observations[i];
        const auto b=o.linear_albedo_rgb_diffuse_factor;
        const bool compatible=(o.metadata&(ssat_native::kPackedSurfaceReservedMask|511u))==diffuse &&
            (o.surface_key>>32u)==(ref.surface_key>>32u) && o.primitive_id!=~0u &&
            o.packed_geometric_normal_oct16==ref.packed_geometric_normal_oct16 && a.x==b.x && a.y==b.y && a.z==b.z &&
            ssat_native::packed_half_is_finite(b.w) && views[i]<p.static_view_count &&
            fabsf(inverse[i])<=1.f && o.hit_distance>=0.f && o.hit_distance<=FLT_MAX*.25f;
        if(!compatible) {
            coplanar=false;
            continue;
        }
        if(o.primitive_id!=ref.primitive_id || o.surface_key!=ref.surface_key) {
            if(views[reference]>=p.static_view_count || !(fabsf(inverse[reference])<=1.f) ||
               !(ref.hit_distance>=0.f && ref.hit_distance<=FLT_MAX*.25f)) {
                coplanar=false;
                continue;
            }
            const auto point=physical_plane_point<Width>(p,i,o.hit_distance);
            const auto origin=physical_plane_point<Width>(p,reference,ref.hit_distance);
            const auto delta=point-origin;
            const float extent=fabsf(delta.x)+fabsf(delta.y)+fabsf(delta.z);
            const float numeric=8.f*FLT_EPSILON*(1.f+fabsf(point.x)+fabsf(point.y)+fabsf(point.z));
            coplanar &= fabsf(dot(normal,delta))<=extent*(4.f/65535.f)+numeric;
        }
    }
    if(!__syncthreads_and(coplanar))return false;
    // The complete tile shares one measured physical Lambertian plane.
    const auto modulation=a.x==a.y && a.y==a.z?
        make_vec3(safe_albedo(make_vec3(from_half(a.x))).x):
        safe_albedo({from_half(a.x),from_half(a.y),from_half(a.z)});
    // All rays on this certified tile share the same finite albedo.
    const Vec3 inverse_modulation{1.f/modulation.x,1.f/modulation.y,1.f/modulation.z};
    const auto *radiance=reinterpret_cast<const float4*>(p.radiance);
    auto *raw_luma=reinterpret_cast<float*>(filtered);
    for(unsigned j=lane;j<sw*sh;j+=512u) {
        const unsigned i=(by+j/sw)*Width+bx+j%sw;
        const auto r=radiance[i];
        if constexpr(CaptureRaw) {
            if(p.atlas_moments && unsigned(j%sw)-3u<32u && unsigned(j/sw)-3u<16u &&
               register_motion_surface(ref.metadata)) {
                const bool valid=finite3({r.x,r.y,r.z}) && fmaxf(fmaxf(r.x,r.y),r.z)<=65504.f;
                store_signal(reinterpret_cast<Half4*>(p.atlas_moments),i,valid?pack({r.x,r.y,r.z,1.f}):Half4{});
            }
        }
        Vec3 z{fmaxf(r.x,0.f),fmaxf(r.y,0.f),fmaxf(r.z,0.f)};
        if(!finite3(z)) {
            fail(p);
            z={};
        }
        z=clamp3({z.x*inverse_modulation.x,z.y*inverse_modulation.y,z.z*inverse_modulation.z},0.f,500.f);
        const auto signal=pack({z.x,z.y,z.z,1.e-4f});
        raw[j]=signal;
        const auto rounded=unpack(signal);
        raw_luma[j]=safe_luminance({rounded.x,rounded.y,rounded.z});
    }
    __syncthreads();
    // The prefilter finishes reading the symmetric edge weights before
    // replacing that storage with the rounded luminance and its response.
    static_assert(4u*sizeof(float)<=sizeof(SharedGuide));
    auto *edges=reinterpret_cast<float*>(guides);
    auto *response=edges+sw*sh;
    build_plane_pair_weights(edges,raw_luma,lane);
    prefilter_plane_exposure<true>(raw,filtered,raw_luma,edges,lane,edges,response);
    plane_exposure_axes<Width,false,true>(p,filtered,raw,edges,nullptr,int(lane),response);
    return true;
}
