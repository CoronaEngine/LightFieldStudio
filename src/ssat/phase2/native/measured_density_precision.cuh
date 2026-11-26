// M records the sum of squared normalized observation weights on this ray.
__device__ __forceinline__ float measured_density_information(const EncodedAtrousNativeParams &p,unsigned i) {
    const auto *geometry=reinterpret_cast<const denoise_native::TemporalSample32*>(p.regular_measurement_geometry);
    const float squared_weights=from_half(geometry[i].sigma_albedo.x);
    // For nu=N_eff-1 the unbiased precision fraction is (nu-2)/nu. The
    // inverse variance has no finite mean at <=2 effective degrees of freedom.
    return fminf(1.f,fmaxf(0.f,(1.f-3.f*squared_weights)/fmaxf(1.f-squared_weights,1.e-20f)));
}
__device__ __forceinline__ float calibrated_density_ratio(float pair_variance,float inverse_twice_center,float information) {
    const float relative_deviation=sqrtf(pair_variance*inverse_twice_center);
    // Estimate the relative standard deviation before taking its reciprocal.
    return 1.f/fmaf(information,relative_deviation-1.f,1.f);
}
