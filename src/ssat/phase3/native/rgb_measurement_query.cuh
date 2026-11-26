#pragma once
// RGB measurement query. Derived from the scalar LF query, with encoded-x
// grid spacing one instead of same-carrier logical-pixel spacing three.
// Physical phase = x + 3*y*slant + offset is unchanged. Exactly four radiance taps.
namespace {

struct CompactRgbFootprint {
    // Every supported encoded coordinate, including the certified basis
    // halo, fits a signed 16-bit word. A positive weight owns a real site.
    unsigned corner{},views{};
    float w0{},w1{},w2{},w3{};

    __device__ __forceinline__ bool common_scale() const {
        return (w0==0.f || w0>=0x1p-32f) && (w1==0.f || w1>=0x1p-32f) &&
               (w2==0.f || w2>=0x1p-32f) && (w3==0.f || w3>=0x1p-32f);
    }

    __device__ __forceinline__ PhaseTap take(const lfs::denoise::mr_native::Params &p,unsigned slot) const {
        PhaseTap result{};
        result.weight=slot==0u?w0:(slot==1u?w1:(slot==2u?w2:w3));
        if(!(result.weight>0.f))return result;
        const int x=int(static_cast<short>(corner))+(slot&1u?p.rgb_b0x:0)+(slot&2u?p.rgb_b1x:0);
        const int y=int(static_cast<short>(corner>>16u))+(slot&1u?p.rgb_b0y:0)+(slot&2u?p.rgb_b1y:0);
        const unsigned view=(views>>(8u*slot))&255u;
        result.index=unsigned(y)*(3u*p.geometry.output_width)+unsigned(x);
        result.site=pack_phase_site(unsigned(x)/3u,unsigned(y),view);
        return result;
    }

    __device__ __forceinline__ void set_corner(int x,int y) {
        corner=unsigned(static_cast<unsigned short>(x))|(unsigned(static_cast<unsigned short>(y))<<16u);
    }
};

__device__ __forceinline__ ContinuousPoint rgb_continuous_point(
    const native::Phase3Params &params,
    float pixel_x,
    float lattice_y,
    std::uint32_t channel,
    int branch) {
    ContinuousPoint result{};
    result.lattice_x = pixel_x;
    result.lattice_y = lattice_y;
    result.wrapped_phase = result.lattice_x +
        3.0f * lattice_y * params.slant_tangent + params.offset -
        static_cast<float>(branch) * params.pitch;
    if (params.view_count > 1u) {
        const float angle = result.wrapped_phase *
            params.phase_to_angle_scale + params.phase_angle_bias;
        const float tangent = calibrated_tangent(angle);
        if (!finitef(tangent)) return result;
        result.u_x = params.focal_distance * tangent;
        const float du_dphase = params.focal_distance *
            (1.0f + tangent * tangent) * params.phase_to_angle_scale;
        result.du_dp = du_dphase;
        result.du_dy = du_dphase * 3.0f * params.slant_tangent;
    }
    result.valid = finitef(result.lattice_x) && finitef(result.lattice_y) &&
        finitef(result.u_x) && finitef(result.du_dp) &&
        finitef(result.du_dy);
    return result;
}

__device__ __forceinline__ ContinuousMetric rgb_continuous_metric(
    const ContinuousPoint &point,
    float inverse_sigma_x,
    float inverse_sigma_u) {
    ContinuousMetric result{};
    if (!point.valid || !(inverse_sigma_x > 0.0f) ||
        !(inverse_sigma_u > 0.0f)) {
        return result;
    }
    const float j00 = inverse_sigma_x;
    const float j11 = inverse_sigma_x;
    const float j20 = point.du_dp * inverse_sigma_u;
    const float j21 = point.du_dy * inverse_sigma_u;
    result.a00 = j00 * j00 + j20 * j20;
    result.a01 = j20 * j21;
    result.a11 = j11 * j11 + j21 * j21;
    result.determinant = result.a00 * result.a11 -
        result.a01 * result.a01;
    const float trace = result.a00 + result.a11;
    if (!finitef(result.determinant) || !finitef(trace) ||
        result.determinant <= 1.0e-8f * fmaxf(trace * trace, 1.0f)) {
        return result;
    }
    result.valid = true;
    return result;
}

template<int Radius, bool StaticEntry>
__device__ __forceinline__ PhaseTap make_rgb_manifold_tap(
    const native::Phase3Params &params,
    const QueryState<Radius, StaticEntry> &state,
    int pixel,
    int row,
    float basis,
    float inverse_sigma_x,
    float inverse_sigma_u) {
    PhaseTap tap{};
    if (pixel < 0 || row < 0 ||
        pixel >= static_cast<int>(params.output_width * 3u) ||
        row >= static_cast<int>(params.output_height) || basis <= 0.0f) {
        return tap;
    }
    const std::uint32_t encoded_x =
        static_cast<std::uint32_t>(pixel);
    tap.index = static_cast<std::uint32_t>(row) *
        (params.output_width * 3u) + encoded_x;
    const std::uint32_t view = producer_view_id(
        params, encoded_x, static_cast<std::uint32_t>(row),
        params.slant_tangent);
    tap.site = pack_phase_site(
        encoded_x / 3u,
        static_cast<std::uint32_t>(row), view);
    const float dx =
        (static_cast<float>(encoded_x) - state.query_lattice_x) *
        inverse_sigma_x;
    const float dy =
        (static_cast<float>(row) - state.query_lattice_y) * inverse_sigma_x;
    const float du =
        (view_position(params, view) - state.query_u_x) * inverse_sigma_u;
    tap.distance_squared = dx * dx + dy * dy + du * du;
    tap.weight = basis * __expf(-tap.distance_squared);
    if (!finitef(tap.distance_squared) || !finitef(tap.weight) ||
        tap.weight <= 0.0f) {
        tap = {};
    }
    return tap;
}

template<int Radius, bool StaticEntry>
__device__ __forceinline__ PhaseTap make_reduced_rgb_manifold_tap(
    const native::Phase3Params &params,
    const QueryState<Radius, StaticEntry> &state,
    const ReducedLatticeBasis &basis,
    int coordinate0,
    int coordinate1,
    float weight,
    float inverse_sigma_x,
    float inverse_sigma_u) {
    const long long pixel =
        static_cast<long long>(basis.b0x) * coordinate0 +
        static_cast<long long>(basis.b1x) * coordinate1;
    const long long row =
        static_cast<long long>(basis.b0y) * coordinate0 +
        static_cast<long long>(basis.b1y) * coordinate1;
    if (pixel < INT_MIN || pixel > INT_MAX || row < INT_MIN || row > INT_MAX) {
        return {};
    }
    return make_rgb_manifold_tap(
        params, state, static_cast<int>(pixel), static_cast<int>(row), weight,
        inverse_sigma_x, inverse_sigma_u);
}

template<int Capacity, bool CertifiedRegular = false, bool GatherOnly = false, bool PrevalidatedRgb = false, int Radius, bool StaticEntry>
__device__ __forceinline__ FixedPhaseTaps project_rgb_manifold_taps(
    const native::Phase3Params &params,
    QueryState<Radius, StaticEntry> &state, ReducedLatticeBasis calibrated = {},
    CompactRgbFootprint *compact = nullptr) {
    FixedPhaseTaps taps{};
    if (!state.query_valid || !(params.pitch > 1.0e-6f) ||
        !(params.inverse_pitch > 0.0f)) {
        state.projection_invalid = true;
        return taps;
    }
    if constexpr (StaticEntry) {
        if (3u != 0u && state.exact_identity_query) {
            const std::uint32_t logical_x = state.encoded_x / 3u;
            taps.tap0.index = state.index;
            taps.tap0.site = pack_phase_site(
                logical_x, state.encoded_y, state.current_view);
            taps.tap0.weight = 1.0f;
            taps.tap0.distance_squared = 0.0f;
            taps.dominant_index = state.index;
            taps.dominant_weight = 1.0f;
            state.projection_residual_squared = 0.0f;
            state.jacobian_condition = 1.0f;
            state.generated_taps = 1u;
            state.support_empty = false;
            return taps;
        }
    }
    const float spatial_pixel =
        state.query_lattice_x;
    const float views = static_cast<float>(max(params.view_count, 1u));
    const float target_phase = params.view_count > 1u
        ? (views - 0.5f - state.continuous_view) * params.pitch_per_view
        : 0.5f * params.pitch;
    const float raw_branch =
        (spatial_pixel +
         3.0f * state.query_lattice_y * params.slant_tangent + params.offset -
         target_phase) * params.inverse_pitch;
    if (!finitef(raw_branch) || raw_branch < -2147483520.0f ||
        raw_branch > 2147483520.0f) {
        state.projection_invalid = true;
        return taps;
    }
    int phase_branch = nearest_branch(raw_branch);
    float pixel_x = spatial_pixel;
    float lattice_y = state.query_lattice_y;
    const float slant3 = 3.0f * params.slant_tangent;
    const float spatial_denominator = fmaf(slant3, slant3, 1.0f);
    const float target_phase_coordinate =
        state.query_lattice_x + slant3 * state.query_lattice_y +
        params.offset - static_cast<float>(phase_branch) * params.pitch;
    float phase_coordinate = target_phase_coordinate;
    const float inverse_sigma_x = params.inverse_sigma_x_lattice;
    const float inverse_sigma_u =
        state.inverse_u_spacing * params.inverse_sigma_u_views;
    const int anchor_pixel = nearest_lattice_integer(spatial_pixel);
    const int anchor_row = nearest_lattice_integer(state.query_lattice_y);
    bool exact_projection = false;
    if (anchor_pixel >= 0 && anchor_row >= 0 &&
        anchor_pixel < static_cast<int>(params.output_width * 3u) &&
        anchor_row < static_cast<int>(params.output_height)) {
        const std::uint32_t anchor_x =
            static_cast<std::uint32_t>(anchor_pixel);
        const std::uint32_t anchor_y =
            static_cast<std::uint32_t>(anchor_row);
        const std::uint32_t anchor_view = producer_view_id(
            params, anchor_x, anchor_y);
        const float x_tolerance = 32.0f * FLT_EPSILON *
            fmaxf(fabsf(state.query_lattice_x), 1.0f);
        const float y_tolerance = 32.0f * FLT_EPSILON *
            fmaxf(fabsf(state.query_lattice_y), 1.0f);
        const float view_tolerance = 32.0f * FLT_EPSILON *
            static_cast<float>(max(params.view_count, 1u));
        const float u_error_views = fabsf(
            view_position(params, anchor_view) - state.query_u_x) *
            state.inverse_u_spacing;
        if (fabsf(static_cast<float>(anchor_x) - state.query_lattice_x) <=
                x_tolerance &&
            fabsf(static_cast<float>(anchor_y) - state.query_lattice_y) <=
                y_tolerance &&
            u_error_views <= view_tolerance) {
            const float anchor_phase_unwrapped = static_cast<float>(anchor_x) +
                3.0f * static_cast<float>(anchor_y) *
                    params.slant_tangent + params.offset;
            phase_branch = static_cast<int>(floorf(
                anchor_phase_unwrapped * params.inverse_pitch));
            state.projection_residual_squared = 0.0f;
            state.jacobian_condition = 1.0f;
            pixel_x = static_cast<float>(anchor_pixel);
            lattice_y = static_cast<float>(anchor_row);
            phase_coordinate = static_cast<float>(anchor_x) +
                slant3 * static_cast<float>(anchor_y) + params.offset -
                static_cast<float>(phase_branch) * params.pitch;
            exact_projection = true;
        }
    }
    // The aperture coordinate depends on a single wrapped phase.  Eliminate
    // pixel/row analytically for a fixed phase, then solve only that scalar.
    // This is the Schur complement of the previous 2x2 Gauss-Newton system;
    // it preserves the same two iterations and calibrated tangent while
    // removing the per-target matrix construction and inverse.
    #pragma unroll
    for (int iteration = 0; iteration < 2; ++iteration) {
        if (exact_projection) break;
        const float angle = phase_coordinate * params.phase_to_angle_scale +
            params.phase_angle_bias;
        const float tangent = calibrated_tangent(angle);
        const float u_x = params.focal_distance * tangent;
        const float du_dphase = params.focal_distance *
            (1.0f + tangent * tangent) * params.phase_to_angle_scale;
        if (!finitef(tangent) || !finitef(u_x) || !finitef(du_dphase)) {
            state.projection_invalid = true;
            return taps;
        }
        const float spatial_precision =
            inverse_sigma_x * inverse_sigma_x / spatial_denominator;
        const float angular_precision = inverse_sigma_u * inverse_sigma_u;
        const float gradient = fmaf(
            spatial_precision,
            phase_coordinate - target_phase_coordinate,
            (u_x - state.query_u_x) * du_dphase * angular_precision);
        const float hessian = fmaf(
            du_dphase * du_dphase, angular_precision, spatial_precision);
        if (!finitef(gradient) || !finitef(hessian) ||
            !(hessian > 1.0e-20f)) {
            state.projection_invalid = true;
            return taps;
        }
        phase_coordinate -= gradient / hessian;
    }
    if (!exact_projection) {
        const float lattice_line = phase_coordinate - params.offset +
            static_cast<float>(phase_branch) * params.pitch;
        lattice_y = (fmaf(
            slant3, lattice_line - state.query_lattice_x,
            state.query_lattice_y)) / spatial_denominator;
        const float lattice_x = lattice_line - slant3 * lattice_y;
        pixel_x = lattice_x;
    }
    const float maximum_pixel = static_cast<float>(params.output_width * 3u - 1u);
    const float maximum_row = static_cast<float>(params.output_height - 1u);
    if (!finitef(pixel_x) || !finitef(lattice_y) || pixel_x < -0.25f ||
        lattice_y < -0.25f || pixel_x > maximum_pixel + 0.25f ||
        lattice_y > maximum_row + 0.25f) {
        state.projection_invalid = true;
        return taps;
    }
    pixel_x = clampf(pixel_x, 0.0f, maximum_pixel);
    lattice_y = clampf(lattice_y, 0.0f, maximum_row);
    ReducedLatticeBasis reduced{};
    static_assert(!PrevalidatedRgb || (Capacity==4 && CertifiedRegular && GatherOnly));
    const bool certified=PrevalidatedRgb || (GatherOnly && CertifiedRegular && calibrated.valid);
    if(certified) {
        const float wrapped=pixel_x+3.f*lattice_y*params.slant_tangent+params.offset-
                            static_cast<float>(phase_branch)*params.pitch;
        if(!finitef(wrapped) || wrapped< -1.e-4f || wrapped>params.pitch+1.e-4f) {
            state.projection_invalid=true;return taps;
        }
        reduced=calibrated;
    } else {
    const ContinuousPoint projected = rgb_continuous_point(
        params, pixel_x, lattice_y, state.channel, phase_branch);
    const ContinuousMetric projected_metric = rgb_continuous_metric(
        projected, inverse_sigma_x, inverse_sigma_u);
    if (!projected.valid || !projected_metric.valid) {
        state.projection_invalid = true;
        return taps;
    }

    if (projected.wrapped_phase < -1.0e-4f ||
        projected.wrapped_phase > params.pitch + 1.0e-4f) {
        state.projection_invalid = true;
        return taps;
    }
    const float pdx =
        (projected.lattice_x - state.query_lattice_x) * inverse_sigma_x;
    const float pdy =
        (projected.lattice_y - state.query_lattice_y) * inverse_sigma_x;
    const float pdu =
        (projected.u_x - state.query_u_x) * inverse_sigma_u;
    state.projection_residual_squared = fmaxf(
        pdx * pdx + pdy * pdy + pdu * pdu, 0.0f);

    reduced = CertifiedRegular && calibrated.valid
        ? calibrated : reduce_lattice_basis<CertifiedRegular>(projected_metric);
    }
    if (!reduced.valid) {
        state.projection_invalid = true;
        return taps;
    }
    const long long determinant =
        static_cast<long long>(reduced.b0x) * reduced.b1y -
        static_cast<long long>(reduced.b0y) * reduced.b1x;
    if (determinant != 1ll && determinant != -1ll) {
        state.projection_invalid = true;
        return taps;
    }
    const float coordinate0 =
        (static_cast<float>(reduced.b1y) * pixel_x -
         static_cast<float>(reduced.b1x) * lattice_y) /
        static_cast<float>(determinant);
    const float coordinate1 =
        (-static_cast<float>(reduced.b0y) * pixel_x +
         static_cast<float>(reduced.b0x) * lattice_y) /
        static_cast<float>(determinant);
    if (!finitef(coordinate0) || !finitef(coordinate1)) {
        state.projection_invalid = true;
        return taps;
    }
    const int base0 = static_cast<int>(floorf(coordinate0));
    const float transition = clampf(
        coordinate0 - static_cast<float>(base0), 0.0f, 1.0f);
    const int base1 = static_cast<int>(floorf(coordinate1));
    const float row_transition = clampf(
        coordinate1 - static_cast<float>(base1), 0.0f, 1.0f);
    const float row0_weight = 1.0f - row_transition;
    const float row1_weight = row_transition;
    if constexpr (Capacity == 4) {
        if(certified) {
            const int x=base0*reduced.b0x+base1*reduced.b1x;
            const int y=base0*reduced.b0y+base1*reduced.b1y;
            if constexpr(PrevalidatedRgb) {
                compact->set_corner(x,y);
                const auto a=make_rgb_manifold_tap(params,state,x,y,row0_weight*(1.f-transition),inverse_sigma_x,inverse_sigma_u);
                compact->w0=a.weight;
                const auto b=make_rgb_manifold_tap(params,state,x+reduced.b0x,y+reduced.b0y,row0_weight*transition,inverse_sigma_x,inverse_sigma_u);
                compact->w1=b.weight;
                const auto c=make_rgb_manifold_tap(params,state,x+reduced.b1x,y+reduced.b1y,row1_weight*(1.f-transition),inverse_sigma_x,inverse_sigma_u);
                compact->w2=c.weight;
                const auto d=make_rgb_manifold_tap(params,state,x+reduced.b0x+reduced.b1x,y+reduced.b0y+reduced.b1y,
                                                    row1_weight*transition,inverse_sigma_x,inverse_sigma_u);
                compact->w3=d.weight;
                compact->views=((a.site>>kPhaseSiteViewShift)&255u) | (((b.site>>kPhaseSiteViewShift)&255u)<<8u) |
                    (((c.site>>kPhaseSiteViewShift)&255u)<<16u) | (((d.site>>kPhaseSiteViewShift)&255u)<<24u);
                return taps;
            }
            taps.tap0=make_rgb_manifold_tap(params,state,x,y,row0_weight*(1.f-transition),inverse_sigma_x,inverse_sigma_u);
            taps.tap1=make_rgb_manifold_tap(params,state,x+reduced.b0x,y+reduced.b0y,row0_weight*transition,inverse_sigma_x,inverse_sigma_u);
            taps.tap2=make_rgb_manifold_tap(params,state,x+reduced.b1x,y+reduced.b1y,row1_weight*(1.f-transition),inverse_sigma_x,inverse_sigma_u);
            taps.tap3=make_rgb_manifold_tap(params,state,x+reduced.b0x+reduced.b1x,y+reduced.b0y+reduced.b1y,
                                            row1_weight*transition,inverse_sigma_x,inverse_sigma_u);
            return taps;
        }
        taps.tap0 = make_reduced_rgb_manifold_tap(
            params, state, reduced, base0, base1,
            row0_weight * (1.0f - transition),
            inverse_sigma_x, inverse_sigma_u);
        taps.tap1 = make_reduced_rgb_manifold_tap(
            params, state, reduced, base0 + 1, base1,
            row0_weight * transition,
            inverse_sigma_x, inverse_sigma_u);
        taps.tap2 = make_reduced_rgb_manifold_tap(
            params, state, reduced, base0, base1 + 1,
            row1_weight * (1.0f - transition),
            inverse_sigma_x, inverse_sigma_u);
        taps.tap3 = make_reduced_rgb_manifold_tap(
            params, state, reduced, base0 + 1, base1 + 1,
            row1_weight * transition,
            inverse_sigma_x, inverse_sigma_u);
    } else {
        const float outgoing_weight = 1.0f - transition;
        const float incoming_weight = transition;
        constexpr float kInverseRowPrecision = 1.0f / 3.0f;
        taps.tap0 = make_reduced_rgb_manifold_tap(
            params, state, reduced, base0, base1,
            row0_weight * kInverseRowPrecision,
            inverse_sigma_x, inverse_sigma_u);
        taps.tap1 = make_reduced_rgb_manifold_tap(
            params, state, reduced, base0 - 1, base1,
            row0_weight * outgoing_weight * kInverseRowPrecision,
            inverse_sigma_x, inverse_sigma_u);
        taps.tap2 = make_reduced_rgb_manifold_tap(
            params, state, reduced, base0 + 1, base1,
            row0_weight * kInverseRowPrecision,
            inverse_sigma_x, inverse_sigma_u);
        taps.tap3 = make_reduced_rgb_manifold_tap(
            params, state, reduced, base0 + 2, base1,
            row0_weight * incoming_weight * kInverseRowPrecision,
            inverse_sigma_x, inverse_sigma_u);
        taps.tap4 = make_reduced_rgb_manifold_tap(
            params, state, reduced, base0, base1 + 1,
            row1_weight * kInverseRowPrecision,
            inverse_sigma_x, inverse_sigma_u);
        taps.tap5 = make_reduced_rgb_manifold_tap(
            params, state, reduced, base0 - 1, base1 + 1,
            row1_weight * outgoing_weight * kInverseRowPrecision,
            inverse_sigma_x, inverse_sigma_u);
        taps.tap6 = make_reduced_rgb_manifold_tap(
            params, state, reduced, base0 + 1, base1 + 1,
            row1_weight * kInverseRowPrecision,
            inverse_sigma_x, inverse_sigma_u);
        taps.tap7 = make_reduced_rgb_manifold_tap(
            params, state, reduced, base0 + 2, base1 + 1,
            row1_weight * incoming_weight * kInverseRowPrecision,
            inverse_sigma_x, inverse_sigma_u);
    }
    state.generated_taps = 0u;
    #pragma unroll
    for (int slot = 0; slot < Capacity; ++slot) {
        // Explicit reads below keep the fixed footprint register-resident.
        const PhaseTap tap = slot == 0 ? taps.tap0 :
            (slot == 1 ? taps.tap1 :
             (slot == 2 ? taps.tap2 :
              (slot == 3 ? taps.tap3 :
               (slot == 4 ? taps.tap4 :
                (slot == 5 ? taps.tap5 :
                 (slot == 6 ? taps.tap6 : taps.tap7))))));
        state.generated_taps += tap.index != 0xffffffffu ? 1u : 0u;
        if (tap.index != 0xffffffffu && tap.weight > 0.0f &&
            (tap.weight > taps.dominant_weight ||
             (tap.weight == taps.dominant_weight &&
              tap.index < taps.dominant_index))) {
            taps.dominant_index = tap.index;
            taps.dominant_weight = tap.weight;
        }
    }
    state.support_empty = state.generated_taps == 0u;
    return taps;
}

} // namespace
