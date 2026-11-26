#pragma once

#include "dsl/dsl.h"
#include "math/base.h"
#include "math/basic_types.h"
#include "math/transform.h"

namespace lfs::lightfield::gpu {

struct LocalPhaseCoordinate {
    ocarina::Float3 x;
    ocarina::Float3 u;
    ocarina::Uint view_id;
    ocarina::Uint channel;
};

struct LocalViewCoordinate {
    ocarina::Float u_x;
    ocarina::Uint view_id;
};

struct WorldPhaseCoordinate {
    LocalPhaseCoordinate local;
    ocarina::Float3 origin;
    ocarina::Float3 direction;
};

template<typename Param>
[[nodiscard]] ocarina::Uint local_view_id(
    const Param &param,
    const ocarina::Uint2 &encoded,
    const ocarina::Float &slant_tangent) noexcept {
    using namespace ocarina;

    const Float d = cast<float>(encoded.x) +
        3.f * cast<float>(encoded.y) * slant_tangent + param.offset;
    const Float period = max(param.pitch, 0.001f);
    const Float wrapped = d - floor(d / period) * period;
    const Float view_width = period /
        max(cast<float>(param.view_count), 1.f);
    Uint view_id = min(
        cast<uint>(floor(wrapped / view_width)),
        max(param.view_count, 1u) - 1u);
    return max(param.view_count, 1u) - 1u - view_id;
}

template<typename Param>
[[nodiscard]] ocarina::Float local_view_position(
    const Param &param,
    const ocarina::Uint &view_id) noexcept {
    using namespace ocarina;
    constexpr float kPi = 3.14159265358979323846f;

    const Float view_u = select(
        param.view_count > 1u,
        cast<float>(view_id) /
            cast<float>(max(param.view_count, 2u) - 1u),
        0.5f);
    const Float view_angle = (view_u - 0.5f) *
        param.array_angle_degrees * (kPi / 180.f);
    return param.focal_distance * tan(view_angle);
}

template<typename Param>
[[nodiscard]] LocalViewCoordinate local_view_coordinate(
    const Param &param,
    const ocarina::Uint2 &encoded,
    const ocarina::Float &slant_tangent) noexcept {
    const ocarina::Uint view_id = local_view_id(
        param, encoded, slant_tangent);
    return {
        local_view_position(param, view_id),
        view_id};
}

template<typename Param>
[[nodiscard]] LocalViewCoordinate local_view_coordinate(
    const Param &param,
    const ocarina::Uint2 &encoded) noexcept {
    return local_view_coordinate(param, encoded, ocarina::tan(param.slant));
}

template<typename Param>
[[nodiscard]] LocalPhaseCoordinate local_phi_with_view(
    const Param &param,
    const ocarina::Uint2 &encoded,
    const LocalViewCoordinate &view) noexcept {
    using namespace ocarina;

    const Uint pixel_x = encoded.x / 3u;
    const Uint channel = encoded.x % 3u;

    const Float pixel_width = param.focal_width /
        cast<float>(param.output_width);
    const Float pixel_height = param.focal_height /
        cast<float>(param.output_height);
    const Float subpixel_offset =
        (cast<float>(channel) + 0.5f) * (pixel_width / 3.f);
    const Float3 x = make_float3(
        -param.focal_width * 0.5f +
            cast<float>(pixel_x) * pixel_width + subpixel_offset,
        param.focal_height * 0.5f -
            (cast<float>(encoded.y) + 0.5f) * pixel_height,
        param.focal_distance);
    return {
        x,
        make_float3(view.u_x, 0.f, 0.f),
        view.view_id,
        channel};
}

template<typename Param>
[[nodiscard]] LocalPhaseCoordinate local_phi(
    const Param &param,
    const ocarina::Uint2 &encoded,
    const ocarina::Float &slant_tangent) noexcept {
    return local_phi_with_view(
        param,
        encoded,
        local_view_coordinate(param, encoded, slant_tangent));
}

template<typename Param>
[[nodiscard]] LocalPhaseCoordinate local_phi(
    const Param &param,
    const ocarina::Uint2 &encoded) noexcept {
    return local_phi(param, encoded, ocarina::tan(param.slant));
}

template<typename Param, typename Transform>
[[nodiscard]] WorldPhaseCoordinate world_phi(
    const Param &param,
    const Transform &local_to_world,
    const ocarina::Uint2 &encoded) noexcept {
    using namespace ocarina;
    LocalPhaseCoordinate local = local_phi(param, encoded);
    const Float3 origin = transform_point(local_to_world, local.u);
    const Float3 focal = transform_point(local_to_world, local.x);
    return {local, origin, normalize(focal - origin)};
}

template<typename Param>
[[nodiscard]] ocarina::Float2 project_x_to_lattice(
    const Param &param,
    const ocarina::Float3 &x) noexcept {
    using namespace ocarina;
    const Float lattice_width = param.focal_width /
        cast<float>(param.output_width * 3u);
    const Float lattice_height = param.focal_height /
        cast<float>(param.output_height);
    return make_float2(
        (x.x + param.focal_width * 0.5f) / lattice_width - 0.5f,
        (param.focal_height * 0.5f - x.y) / lattice_height - 0.5f);
}

} // namespace lfs::lightfield::gpu
