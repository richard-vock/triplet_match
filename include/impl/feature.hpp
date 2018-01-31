namespace {

using triplet_match::vec3f_t;

inline float
angle(const vec3f_t& a, const vec3f_t& b) {
    return atan2f(a.cross(b).norm(), fabs(a.dot(b)));
}

} // anon namespace

namespace triplet_match {

template <typename Point>
inline std::optional<feature_t>
feature(const Point& pnt_0,
        const Point& pnt_1, const Point& pnt_2, const curv_info_t<Point>& crv_0, const curv_info_t<Point>& crv_1, const curv_info_t<Point>& crv_2) {
    // planes use the same features as cylinders (or all "pseudo-2d-embeddings" for that matter)
    // i.e. intrinsic distance (in this case simple euclidean L2) and intrinsic tangent angles (with w=0)
    vec3f_t p0 = pnt_0.getVector3fMap();
    vec3f_t p1 = pnt_1.getVector3fMap();
    vec3f_t p2 = pnt_2.getVector3fMap();
    vec3f_t tgt_0 = tangent(pnt_0);
    vec3f_t tgt_1 = tangent(pnt_1);
    vec3f_t tgt_2 = tangent(pnt_2);

    feature_t feat;
    vec3f_t d0 = p1 - p0;
    vec3f_t d1 = p2 - p1;
    vec3f_t d2 = p0 - p2;
    feat[0] = d0.norm();
    feat[1] = d2.norm();
    feat[2] = d1.norm();
    feat[3] = angle(d0, tgt_1);
    feat[4] = angle(d1, tgt_2);
    feat[5] = angle(d2, tgt_0);
    vec3f_t nrm = d1.cross(d2);
    feat[6] = angle(nrm, vec3f_t::UnitZ());
        //feat[6] = crv_0.pc_max < Eigen::NumTraits<float>::dummy_precision() ? 1.f : crv_0.pc_min / crv_0.pc_max;
        //feat[7] = crv_1.pc_max < Eigen::NumTraits<float>::dummy_precision() ? 1.f : crv_1.pc_min / crv_1.pc_max;
        //feat[8] = crv_2.pc_max < Eigen::NumTraits<float>::dummy_precision() ? 1.f : crv_2.pc_min / crv_2.pc_max;

    return feat;
}

template <typename Point>
inline discrete_feature_t
discretize_feature(const feature_t& f, const feature_bounds_t& bounds, const discretization_params& params) {
    discrete_feature_t df;
    df <<
        discretize(f[0], bounds.min()[0], bounds.diagonal()[0], params.distance_step_count),
        discretize(f[1], bounds.min()[1], bounds.diagonal()[1], params.distance_step_count),
        discretize(f[2], bounds.min()[2], bounds.diagonal()[2], params.distance_step_count),
        discretize(f[3], params.angle_step),
        discretize(f[4], params.angle_step),
        discretize(f[5], params.angle_step),
        discretize(f[6], params.angle_step);
    return df;
}

template <typename Point>
inline bool
valid(const feature_t& f, const feature_bounds_t& bounds) {
    if (f[0] < bounds.min()[0] || f[0] > bounds.max()[0]) {
        return false;
    }
    //if constexpr (!scale_invariant) {
        if (f[1] < bounds.min()[1] || f[1] > bounds.max()[1]) {
            return false;
        }
        if (f[2] < bounds.min()[2] || f[2] > bounds.max()[2]) {
            return false;
        }
    //}

    //if (f[6] < bounds.min()[6] || f[6] > bounds.max()[6]) {
        //return false;
    //}
    //if (f[7] < bounds.min()[7] || f[7] > bounds.max()[7]) {
        //return false;
    //}
    //if (f[8] < bounds.min()[8] || f[8] > bounds.max()[8]) {
        //return false;
    //}

    float pi = static_cast<float>(M_PI);
    bool angle_ok = true;
    //if constexpr (scale_invariant) {
        //angle_ok = (f[1] >= 0.f && f[1] <= pi) && (f[2] >= 0.f && f[2] <= pi) && (f[3] >= 0.f && f[3] <= pi);
    //} else {
        //angle_ok = f[3] >= 0.f && f[3] <= pi;
        angle_ok = (f[3] >= 0.f && f[3] <= pi) && (f[4] >= 0.f && f[4] <= pi) && (f[5] >= 0.f && f[5] <= pi) && (f[6] >= 0.f && f[6] <= pi);
    if (!angle_ok) pdebug("not valid angle");
    //}

    //if (!angle_ok) pdebug("angles {} {} {}", f[1], f[2], f[3]);
    return angle_ok;
}

feature_bounds_t
valid_bounds(const feature_bounds_t& bounds, float /*min_angle*/, float /*max_angle*/, float min_rel_dist, float max_rel_dist) {
    feature_bounds_t new_bounds = bounds;

    // distance is adjusted to relative bounds
    new_bounds.min()[0] = bounds.min()[0] + min_rel_dist * bounds.diagonal()[0];
    new_bounds.max()[0] = bounds.min()[0] + max_rel_dist * bounds.diagonal()[0];
    //if constexpr (!scale_invariant) {
        new_bounds.min()[1] = bounds.min()[1] + min_rel_dist * bounds.diagonal()[1];
        new_bounds.max()[1] = bounds.min()[1] + max_rel_dist * bounds.diagonal()[1];
        new_bounds.min()[2] = bounds.min()[2] + min_rel_dist * bounds.diagonal()[2];
        new_bounds.max()[2] = bounds.min()[2] + max_rel_dist * bounds.diagonal()[2];
    //}

    //new_bounds.min()[1] = min_angle;
    //new_bounds.min()[2] = min_angle;
    //new_bounds.min()[3] = min_angle;
    //new_bounds.max()[1] = max_angle;
    //new_bounds.max()[2] = max_angle;
    //new_bounds.max()[3] = max_angle;

    return new_bounds;
}

} // triplet_match
