// A diffuse receiver is independent of outgoing direction. Query the same
// transported surface point on both sides of its previous physical aperture.
// The pair is used only when the ordinary registered history has no valid tap.
struct ProjectedDiffuseHistory {
    float4 mean{};
    float variance{};
    float weight{};
};

__device__ ProjectedDiffuseHistory previous_diffuse_side(
        const MrParams &params, const QueryState<0,false> &original,
        const Observation &a, int center_view, int side) {
    const auto &g=params.geometry;
    const float3 point=original.predicted_previous_position_local;
    const int view=center_view+side;
    const float u=view_position(g,view),inv_z=g.focal_distance/point.z;
    const float fx=fmaf(point.x-u,inv_z,u),fy=point.y*inv_z;
    auto q=original;
    q.query_lattice_x=(fx+.5f*g.focal_width)*g.inverse_focal_lattice_width-.5f;
    q.query_lattice_y=(.5f*g.focal_height-fy)*g.inverse_focal_lattice_height-.5f;
    q.query_u_x=u;q.continuous_view=float(view);q.inverse_u_spacing=inverse_view_spacing(g,view);
    q.query_direction_previous_local=normalize3(subtract(point,make_float3(u,0.f,0.f)));
    q.exact_identity_query=false;
    const auto four=project_rgb_manifold_taps<4,true,true>(g,q,
        {params.rgb_b0x,params.rgb_b0y,params.rgb_b1x,params.rgb_b1y,params.rgb_basis_valid!=0u});
    const PhaseTap taps[]{four.tap0,four.tap1,four.tap2,four.tap3};
    const float source_area=g.focal_lattice_width*g.focal_lattice_height*
        (point.z*g.inverse_focal_distance)*(point.z*g.inverse_focal_distance)*float(g.view_count);
    const float radius_squared=fmaxf(source_area*.318309886f,1.e-20f);
    const float plane_limit=g.base_position_tolerance*fmaxf(point.z*g.inverse_focal_distance,1.f);
    ProjectedDiffuseHistory best{};
    for(const auto &tap:taps) {
        if(!(tap.weight>0.f) || tap.index>=3u*g.output_width*g.output_height)continue;
        const int actual=int(tap.site>>kPhaseSiteViewShift);
        if((actual-center_view)*side<=0)continue;
        const auto old=pointer<const native::TemporalSample>(g.history_read)[tap.index];
        if(!surface_valid(old.metadata) || surface_class(old.metadata)!=kSurfaceDiffuse ||
           old.surface_key!=a.surface_key || history_age(old.metadata)<2u || !(old.depth>0.f))continue;
        const auto b=old.signal_physical_mean_rms,c=a.linear_albedo_rgb_diffuse_factor;
        if(b.x!=c.x || b.y!=c.y || b.z!=c.z || b.w!=c.w)continue;
        const float3 normal=transform_vector(g.previous_world_to_local,unpack_normal(old.packed_normal));
        const float orientation=dot3(original.predicted_previous_normal_local,normal);
        if(orientation<g.minimum_normal_similarity)continue;
        const auto ray=analytical_local_ray(g,tap.index%(3u*g.output_width),tap.index/(3u*g.output_width),actual);
        const float3 offset=subtract(add(ray.origin,multiply(ray.direction,old.depth)),point);
        const float d2=dot3(offset,offset);
        if(d2>radius_squared || fabsf(dot3(offset,normal))>plane_limit)continue;
        const float score=orientation*__expf(-.5f*d2/radius_squared);
        if(score<=best.weight)continue;
        const auto m=pointer<const native::Half4>(params.measurement_read)[tap.index];
        const float sigma=half_to_float(m.w);
        best={make_float4(half_to_float(m.x),half_to_float(m.y),half_to_float(m.z),0.f),sigma*sigma,score};
    }
    return best;
}

__device__ void recover_diffuse_pair(const MrParams &p,const QueryState<0,false> &state,const Observation &a,
                                     float4 sample,float4 &mean,float &s2,unsigned &age) {
    if(!state.query_valid || state.exact_identity_query || !(state.predicted_previous_position_local.z>0.f))return;
    const int v=__float2int_rn(state.continuous_view);
    if(v<=0 || v+1>=int(p.geometry.view_count))return;
    const auto left=previous_diffuse_side(p,state,a,v,-1);
    if(!(left.weight>0.f))return;
    const auto right=previous_diffuse_side(p,state,a,v,1);
    if(!(right.weight>0.f))return;
    const float diff=mr_luma(left.mean)-mr_luma(right.mean),noise=left.variance+right.variance;
    if(diff*diff>noise)return;
    // Both previous means can contain shared temporal sources. Together they
    // contribute at most one effective previous observation to the new ray.
    const float support=fminf(left.weight,right.weight);
    const float beta=support/(1.f+support),alpha=1.f-beta;
    const float4 hint=make_float4(.5f*(left.mean.x+right.mean.x),.5f*(left.mean.y+right.mean.y),
                                  .5f*(left.mean.z+right.mean.z),0.f);
    const float change=mr_luma(sample)-mr_luma(hint);
    mean=make_float4(fmaf(beta,hint.x-sample.x,sample.x),fmaf(beta,hint.y-sample.y,sample.y),
                     fmaf(beta,hint.z-sample.z,sample.z),beta*(.5f*noise+.25f*diff*diff+alpha*change*change));
    s2=fmaf(alpha,alpha,beta*beta);age=2u;
}
