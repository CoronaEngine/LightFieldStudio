// Geometry shared by per-view reconstruction and the cross-aperture gather.
// Radiance never enters the cached weights.
namespace {
using AtlasObservation = denoise_native::PackedSurfaceObservationV1;

__device__ __forceinline__ float atlas_geometry(const AtlasObservation &a, const AtlasObservation &b,
                                                Vec3 n) {
    if (a.surface_key != b.surface_key || ((a.metadata ^ b.metadata) & 511u))
        return 0.f;
    if (a.packed_geometric_normal_oct16 == b.packed_geometric_normal_oct16)
        return 1.f;
    float v = fmaxf(dot(n, unpack_normal(b.packed_geometric_normal_oct16)), 0.f);
    v *= v;
    v *= v;
    v *= v;
    v *= v;
    return v * v;
}

} // namespace

extern "C" __global__ void lfs_stationary_check_geometry(EncodedAtrousNativeParams p) {
    constexpr unsigned valid_identity = 1u | (1u << 1u);
    // A public 64-byte RigidInstanceMotionRecord ends with a uint4 of metadata.
    bool changed = p.rebuild_weights || !p.rigid_count;
    const auto *records = reinterpret_cast<const uint4 *>(p.rigid_records);
    for (unsigned j = threadIdx.x; j < p.rigid_count; j += blockDim.x) {
        if ((records[4 * j + 3].x & valid_identity) != valid_identity)
            changed = true;
    }
    const int any = __syncthreads_or(changed);
    if (threadIdx.x == 0)
        *reinterpret_cast<unsigned *>(p.weights_refresh) = unsigned(any);
}
