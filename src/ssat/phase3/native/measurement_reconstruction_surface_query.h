#pragma once
// A diffuse lighting query through the reprojected surface, on the legal
// previous horizontal aperture. No additional history reads or path work.
template<bool Calibrated=false,typename Geometry,typename State>
__device__ void mr_surface_query(const Geometry &g,State &state) {
    if(!state.query_valid || state.exact_identity_query || !state.current_valid ||
       !(state.current_class&kSurfaceDiffuse) || (state.current_class&kResponsiveSurfaceMask))return;
    const float3 surface=state.predicted_previous_position_local;
    if(surface.z<=1.e-6f)return;
    const float a=view_position(g,0u),b=view_position(g,g.view_count-1u);
    const float u=clampf(state.query_u_x,fminf(a,b),fmaxf(a,b));
    const float scale=g.focal_distance/surface.z;
    state.query_lattice_x=(u+(surface.x-u)*scale+g.focal_width*.5f)*g.inverse_focal_lattice_width-.5f;
    state.query_lattice_y=(g.focal_height*.5f-surface.y*scale)*g.inverse_focal_lattice_height-.5f;
    if(!Calibrated || state.query_u_x!=u) {
        const auto angular=angular_coordinate<Calibrated>(g,u);
        state.continuous_view=angular.continuous_view;state.inverse_u_spacing=angular.inverse_spacing;
    }
    state.query_u_x=u;
    state.query_direction_previous_local=normalize3(subtract(surface,make_float3(u,0,0)));
}
