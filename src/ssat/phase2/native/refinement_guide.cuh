struct RefinementGuide { Vec3 p,n;Half4 a;unsigned instance,valid,flags; };
static_assert(sizeof(RefinementGuide)==44u);
