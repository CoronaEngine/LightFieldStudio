
__device__ __forceinline__ float plane_spline(unsigned k) {
    return k==2u?.375f:((k&1u)?.25f:.0625f);
}

template<int Width,int Height>
__device__ bool reconstruct_lambertian_plane(const EncodedAtrousNativeParams &p,
        const Half4 *src,Half4 *destination,Half4 *tile,SharedGuide *geometry,Vec3 *normal_scratch,float *luma) {
    constexpr int sw=36,sh=12;
    const int thread=int(threadIdx.y)*32+int(threadIdx.x);
    const int bx=int(blockIdx.x)*64+int(blockIdx.z%2u),by=int(blockIdx.y)*16+int(blockIdx.z/2u);
    if(bx<4 || by<4 || bx+66>=Width || by+18>=Height)return false;
    using Obs=denoise_native::PackedSurfaceObservationV1;
    const auto *obs=reinterpret_cast<const Obs*>(p.packed_observation);
    const unsigned first=by*Width+bx;
    const auto ref=obs[first];
    if((ref.metadata&511u)!=(1u|(kSurfaceDiffuse<<2u)))return false;
    const auto alb=ref.linear_albedo_rgb_diffuse_factor;
    const unsigned i=(by+int(threadIdx.y)*2)*Width+bx+int(threadIdx.x)*2;
    const auto n=local_normal(p,unpack_normal(ref.packed_geometric_normal_oct16));
    const auto point=physical_plane_point<Width>(p,first,ref.hit_distance);
    const float epsilon=8.f*FLT_EPSILON*(fabsf(point.x)+fabsf(point.y)+fabsf(point.z)+1.f);
    constexpr unsigned horizontal_count=32u*sh;
    auto *means=reinterpret_cast<Vec4*>(geometry);
    static_assert(horizontal_count*sizeof(Vec4)+sw*sh*sizeof(Vec3)<=sw*sh*sizeof(SharedGuide));
    auto *rounded_rgb=reinterpret_cast<Vec3*>(means+horizontal_count);
    // The guide scratch carries squared source luminance in z and the
    // two horizontal uncertainty moments in x and y.
    auto *uncertainty=normal_scratch;
    bool flat=true,has_measurement=false;
    for(int j=thread;j<sw*sh;j+=256) {
        const unsigned k=(by+2*(j/sw-2))*Width+bx+2*(j%sw-2);
        const auto hit=obs[k];
        const auto a=hit.linear_albedo_rgb_diffuse_factor;
        // Adjacent triangles on the same measured plane need only agree in
        // instance, actual normal, and material response. A disjoint parallel
        // surface still fails the residual test below.
        const bool compatible=(hit.surface_key>>32u)==(ref.surface_key>>32u) &&
            hit.packed_geometric_normal_oct16==ref.packed_geometric_normal_oct16 &&
            (hit.metadata&511u)==(1u|(kSurfaceDiffuse<<2u)) &&
            a.x==alb.x && a.y==alb.y && a.z==alb.z;
        if(!compatible) {
            flat=false;
            continue;
        }
        const bool same_triangle=hit.surface_key==ref.surface_key && hit.primitive_id==ref.primitive_id &&
            hit.primitive_id!=~0u && isfinite(hit.hit_distance) && hit.hit_distance>=0.f;
        if(!same_triangle) {
            // A different triangle still requires the measured coplanarity
            // residual. All sites of the same actual triangle share its plane.
            const auto d=physical_plane_point<Width>(p,k,hit.hit_distance)-point;
            const float spread=fabsf(d.x)+fabsf(d.y)+fabsf(d.z);
            flat &= fabsf(dot(n,d))<=epsilon+spread*(4.f/65535.f);
        }
        const auto sample=load_signal(src,k);
        tile[j]=sample;
        const auto v=unpack(sample);
        const Vec3 rgb{v.x,v.y,v.z};
        rounded_rgb[j]=rgb;
        const float z=luminance(rgb);
        luma[j]=z;
        uncertainty[j].z=z*z;
        has_measurement|=v.w>=0.f;
    }
    if(!__syncthreads_and(flat) || !__syncthreads_or(has_measurement))return false;
    // Both separable moments retain all 25 native sites. Carry completeness
    // through the two passes so each receiver is bound to its own evidence.
    for(int j=thread;j<32*sh;j+=256) {
        const int at=(j/32)*sw+j%32;
        Vec3 sum{};float second=0.f,total_noise=0.f,mean_noise=0.f;
        bool measured=true;
        #pragma unroll
        for(unsigned k=0;k<5;++k) {
            const float w=plane_spline(k),z=uncertainty[at+k].z;
            const float stored=from_half(tile[at+k].w),variance=fabsf(stored);
            measured&=stored>=0.f;
            sum+=rounded_rgb[at+k]*w;second=fmaf(w,z,second);
            total_noise=fmaf(w,variance,total_noise);mean_noise=fmaf(w*w,variance,mean_noise);
        }
        means[j]={sum.x,sum.y,sum.z,second};
        uncertainty[j].x=total_noise;
        uncertainty[j].y=measured?mean_noise:-mean_noise;
    }
    __syncthreads();
    Vec3 base{};float square=0.f,observed=0.f,mean_noise=0.f;
    bool measured=true;
    #pragma unroll
    for(unsigned row=0;row<5;++row) {
        const unsigned j=thread+32u*row;const float w=plane_spline(row);
        const auto v=means[j];const auto noise=uncertainty[j];
        base+=Vec3{v.x,v.y,v.z}*w;square=fmaf(w,v.w,square);
        measured&=(__float_as_uint(noise.y)&0x80000000u)==0u;
        observed=fmaf(w,noise.x,observed);mean_noise=fmaf(w*w,fabsf(noise.y),mean_noise);
    }
    const unsigned sx=unsigned(threadIdx.x)+2u,sy=unsigned(threadIdx.y)+2u;
    const auto v=unpack(tile[sy*sw+sx]);
    Half4 result;
    if(measured) {
        const float z=luminance(base);
        const float structure=fmaxf(fmaf(-z,z,square)-(observed-mean_noise),0.f);
        const float own=.375f*.375f;
        const float innovation=fmaxf(fmaf(1.f-2.f*own,v.w,mean_noise),0.f);
        const float gain=structure/fmaxf(structure+innovation,1.e-20f);
        const auto rgb=base+(Vec3{v.x,v.y,v.z}-base)*gain;
        const float beta=1.f-gain;
        const float var=fmaf(beta*beta,mean_noise,(gain*gain+2.f*gain*beta*own)*v.w);
        result=pack({rgb.x,rgb.y,rgb.z,fmaxf(var,1.e-4f)});
    } else {
        result=integrate_radiometric_plane<sw,true,false>(tile,luma,nullptr,nullptr,int(sx),int(sy));
    }
    store_signal(destination,i,result);
    if(v.w<0.f)return true;
    const unsigned ch=i%3u;
    const float component=from_half(ch==0u?result.x:(ch==1u?result.y:result.z));
    const float a=from_half(ch==0u?alb.x:(ch==1u?alb.y:alb.z));
    const float physical=from_half(to_half(component*p.final_energy))*sqrtf(fmaf(a,a,.01f));
    auto *panel=reinterpret_cast<float*>(p.canonical);
    panel[4u*(i/3u)+ch]=physical;
    if(ch==2u)panel[4u*(i/3u)+3u]=1.f;
    return true;
}
