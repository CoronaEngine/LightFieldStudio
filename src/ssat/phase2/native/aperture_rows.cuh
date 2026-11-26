// Each warp reads a single angular interval across 32 neighboring receiver cells.
__device__ __forceinline__ float aperture_band_sum(const float4 *parts,unsigned column,unsigned component) {
    float values[8];
    #pragma unroll
    for(unsigned band=0;band<8;++band) {
        const auto a=parts[band*32u+column];
        values[band]=component==0u?a.x:(component==1u?a.y:(component==2u?a.z:a.w));
    }
    return ((values[0]+values[4])+(values[2]+values[6]))+((values[1]+values[5])+(values[3]+values[7]));
}

struct ApertureWitness {
    Vec3 rgb{};
    float luminance{},variance{};
    bool valid{};
};

template<int Width>
__device__ __forceinline__ ApertureWitness aperture_witness(
    const EncodedAtrousNativeParams &p,const AtlasObservation &receiver,unsigned view,
    float aperture,float shift,int x,unsigned row,int columns,int rows) {
    const auto *origins=reinterpret_cast<const float*>(p.static_view_x);
    const float xx=fmaf(origins[view]-aperture,shift,float(x));
    if(!(xx>=-.5f && xx<float(Width)-.5f))return {};
    const unsigned col=__float2uint_rd((xx+.5f)/float(3*p.atlas_side));
    const unsigned site=reinterpret_cast<const unsigned*>(p.atlas)[(view*rows+row)*columns+col];
    if(site>=p.count)return {};
    const auto donor=reinterpret_cast<const AtlasObservation*>(p.packed_observation)[site];
    if(donor.surface_key!=receiver.surface_key ||
       donor.packed_geometric_normal_oct16!=receiver.packed_geometric_normal_oct16 ||
       ((donor.metadata^receiver.metadata)&511u))return {};
    const auto value=unpack(load_signal(reinterpret_cast<const Half4*>(p.field_measurements),site));
    const Vec3 color{value.x,value.y,value.z};
    return {color,luminance(color),fabsf(value.w),value.w>=0.f};
}

template<int Width>
__device__ __forceinline__ bool integrate_coherent_aperture_band(
    const EncodedAtrousNativeParams &p,const AtlasObservation &receiver,
    unsigned first,unsigned end,float aperture,float shift,int x,unsigned row,int columns,int rows,
    float step,float3 &sums,float &mass,float &square,float &observed,float &sigma) {
    if(end-first<5u)return false;
    const unsigned last=end-1u,middle=(first+last)/2u;
    const auto a=aperture_witness<Width>(p,receiver,first,aperture,shift,x,row,columns,rows);
    const auto b=aperture_witness<Width>(p,receiver,middle,aperture,shift,x,row,columns,rows);
    const auto c=aperture_witness<Width>(p,receiver,last,aperture,shift,x,row,columns,rows);
    if(!a.valid || !b.valid || !c.valid)return false;
    const float ab=a.luminance-b.luminance,bc=b.luminance-c.luminance,ac=a.luminance-c.luminance;
    if(ab*ab>a.variance+b.variance || bc*bc>b.variance+c.variance || ac*ac>a.variance+c.variance)return false;

    // Sum the piecewise-linear quadrature weights on the original discrete
    // directions. Its endpoints retain the original trapezoid half weights.
    const float left=.5f*float(middle-first-1u),right=.5f*float(last-middle-1u);
    const float wa=step*(1.f+left-(first==0u?.5f:0.f));
    const float wb=step*(1.f+left+right);
    const float wc=step*(1.f+right-(end==p.static_view_count?.5f:0.f));
    const auto rgb=a.rgb*wa+b.rgb*wb+c.rgb*wc;
    sums=make_float3(rgb.x,rgb.y,rgb.z);
    mass=wa+wb+wc;
    square=fmaf(wa,a.luminance*a.luminance,fmaf(wb,b.luminance*b.luminance,wc*c.luminance*c.luminance));
    observed=fmaf(wa,a.variance,fmaf(wb,b.variance,wc*c.variance));
    sigma=fmaf(wa,sqrtf(a.variance),fmaf(wb,sqrtf(b.variance),wc*sqrtf(c.variance)));
    return true;
}

template<int Width,int Height>
__device__ void integrate_aperture_rows(const EncodedAtrousNativeParams &p,unsigned dispatch_index) {
    __shared__ float4 rgb_mass[256],statistics[256];
    const unsigned column=threadIdx.x&31u,band=threadIdx.x>>5u;
    const unsigned cell=(dispatch_index>>8u)*32u+column;
    const int side=p.atlas_side,columns=(Width+3*side-1)/(3*side),rows=(Height+side-1)/side;
    constexpr int row_step=Width==11520?2:1;
    const unsigned plane=unsigned(columns*((rows+row_step-1)/row_step));
    const auto *sites=reinterpret_cast<const unsigned*>(p.atlas);
    const auto *obs=reinterpret_cast<const AtlasObservation*>(p.packed_observation);
    const auto *means=reinterpret_cast<const Half4*>(p.field_measurements);
    const auto *origins=reinterpret_cast<const float*>(p.static_view_x);
    const unsigned aperture=cell<plane?p.static_view_count/4u:3u*p.static_view_count/4u;
    unsigned i=~0u;
    if(cell<2u*plane) {
        if constexpr(row_step==1) i=sites[aperture*plane+cell%plane];
        else {
            // The calibrated aperture origins differ only in x. Preserve each
            // horizontal cell and retain the actual ray nearest the pair's
            // vertical center from the two calibrated raster rows.
            const unsigned local=cell<plane?cell:cell-plane;
            const unsigned first_row=(local/columns)*row_step;
            const unsigned address=(aperture*rows+first_row)*columns+local%columns;
            const unsigned first=sites[address];
            const unsigned second=first_row+1u<unsigned(rows)?sites[address+columns]:~0u;
            i=first<p.count?first:second;
            if(first<p.count && second<p.count) {
                constexpr unsigned diffuse=1u|(kSurfaceDiffuse<<2u);
                const bool first_diffuse=(obs[first].metadata&511u)==diffuse;
                const bool second_diffuse=(obs[second].metadata&511u)==diffuse;
                const float center=float(first_row*side)+float(side)-.5f;
                if(second_diffuse!=first_diffuse?second_diffuse:
                   fabsf(float(second/Width)-center)<fabsf(float(first/Width)-center))i=second;
            }
        }
    }
    AngularFieldSample record{};record.site=i;
    float3 sums=make_float3(0.f,0.f,0.f);
    float mass=0.f,square=0.f,observed=0.f,band_sigma=0.f;
    AtlasObservation a{};
    if(i<p.count) a=obs[i];
    const bool diffuse=i<p.count && (a.metadata&511u)==(1u|(kSurfaceDiffuse<<2u));
    if(!__syncthreads_or(diffuse)) {
        if(!band && cell<2u*plane) {
            auto &d=reinterpret_cast<AngularFieldSample*>(p.motion_cell_values)[p.motion_cell_count+cell];
            d.site=i;d.support=0;
        }
        return;
    }
    if(diffuse) {
        record.surface=a.surface_key;record.normal=a.packed_geometric_normal_oct16;record.metadata=a.metadata;
        const float depth=a.hit_distance*reinterpret_cast<const float*>(p.static_inverse_ray_length)[i];
        if(depth>0.f) {
            const int x=i%Width,y=i/Width;
            const auto n=unpack_normal(a.packed_geometric_normal_oct16);
            const float shift=(1.f-1.f/depth)*p.atlas_inverse_dx;
            const unsigned last=p.static_view_count-1u;
            const float step=2.f/float(last);
            const unsigned first=(band*p.static_view_count+7u)/8u,end=((band+1u)*p.static_view_count+7u)/8u;
            bool measured=false;
            if constexpr(Width==11520) {
                measured=integrate_coherent_aperture_band<Width>(p,a,first,end,origins[aperture],shift,
                    x,unsigned(y/side),columns,rows,step,sums,mass,square,observed,band_sigma);
            }
            for(unsigned v=measured?end:first;v<end;++v) {
                const float xx=fmaf(origins[v]-origins[aperture],shift,float(x));
                if(!(xx>=-.5f && xx<float(Width)-.5f))continue;
                const unsigned col=__float2uint_rd((xx+.5f)/float(3*side));
                const unsigned j=sites[(v*rows+unsigned(y/side))*columns+col];
                if(j>=p.count)continue;
                const float g=atlas_geometry(a,obs[j],n);
                if(!(g>0.f))continue;
                const float w=(v==0 || v==last?.5f:1.f)*step*g;
                const auto m=unpack(load_signal(means,j));const float lum=luminance({m.x,m.y,m.z}),var=fabsf(m.w);
                sums.x=fmaf(m.x,w,sums.x);sums.y=fmaf(m.y,w,sums.y);sums.z=fmaf(m.z,w,sums.z);
                mass+=w;square=fmaf(w,lum*lum,square);observed=fmaf(w,var,observed);
                band_sigma=fmaf(w,sqrtf(var),band_sigma);
            }
        }
    }
    rgb_mass[threadIdx.x]=make_float4(sums.x,sums.y,sums.z,mass);
    statistics[threadIdx.x]=make_float4(square,observed,band_sigma*band_sigma,0.f);
    __syncthreads();
    if(band || cell>=2u*plane)return;
    auto &d=reinterpret_cast<AngularFieldSample*>(p.motion_cell_values)[p.motion_cell_count+cell];
    if(!diffuse){d.site=i;d.support=0;return;}
    mass=aperture_band_sum(rgb_mass,column,3u);
    if(mass>1.e-10f) {
        const float inv=1.f/mass;
        const Vec3 color{aperture_band_sum(rgb_mass,column,0u)*inv,aperture_band_sum(rgb_mass,column,1u)*inv,
                         aperture_band_sum(rgb_mass,column,2u)*inv};
        const float mean_noise=aperture_band_sum(statistics,column,2u);
        square=aperture_band_sum(statistics,column,0u);observed=aperture_band_sum(statistics,column,1u);
        record.radiance_and_sigma=pack({color.x,color.y,color.z,sqrtf(mean_noise)*inv});
        const float z=luminance(color);
        record.structural_sigma=to_half(sqrtf(fmaxf((square-observed)*inv-z*z,0.f)));
        record.support=to_half(mass);
        finish_field_record(record,i,a.metadata);
    }
    d=record;
}
