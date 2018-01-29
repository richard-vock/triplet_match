namespace {

using triplet_match::vec3f_t;

inline float
angle(const vec3f_t& a, const vec3f_t& b) {
    return atan2f(a.cross(b).norm(), fabs(a.dot(b)));
}

} // anon namespace

namespace triplet_match {

template <typename Proj, typename Point>
inline std::optional<feature_t>
feature(typename Proj::const_handle_t handle, const Point& pnt_0,
                             const Point& pnt_1, const Point& pnt_2, const curv_info_t<Point>& crv_0, const curv_info_t<Point>& crv_1, const curv_info_t<Point>& crv_2) {
    // planes use the same features as cylinders (or all "pseudo-2d-embeddings" for that matter)
    // i.e. intrinsic distance (in this case simple euclidean L2) and intrinsic tangent angles (with w=0)
    std::optional<vec3f_t> uvw_0 = Proj::project(handle, pnt_0.getVector3fMap());
    std::optional<vec3f_t> uvw_1 = Proj::project(handle, pnt_1.getVector3fMap());
    std::optional<vec3f_t> uvw_2 = Proj::project(handle, pnt_2.getVector3fMap());

    if (!uvw_0 || !uvw_1 || !uvw_2) return std::nullopt;

    vec3f_t tgt_0 = Proj::tangent(handle, pnt_0);
    vec3f_t tgt_1 = Proj::tangent(handle, pnt_1);
    vec3f_t tgt_2 = Proj::tangent(handle, pnt_2);

    feature_t feat;
    //if constexpr (scale_invariant) {
        //feat[0] = Proj::intrinsic_distance(handle, *uvw_0, *uvw_2) / Proj::intrinsic_distance(handle, *uvw_0, *uvw_1);

        //vec3f_t d1 = (*uvw_1) - (*uvw_0);
        //vec3f_t d2 = (*uvw_2) - (*uvw_0);
        //vec3f_t d3 = (*uvw_2) - (*uvw_1);
        //vec3f_t c1 = d1.cross(d2);
        //vec3f_t c2 = (-d1).cross(d3);
        //vec3f_t c3 = (-d2).cross(-d3);
        //feat[1] = atan2f(c1.norm(), d1.dot(d2));
        //feat[2] = atan2f(c2.norm(), (-d1).dot(d3));
        //feat[3] = atan2f(c3.norm(), (-d2).dot(-d3));
    //} else {
        vec3f_t d1 = (*uvw_1) - (*uvw_0);
        vec3f_t d2 = (*uvw_2) - (*uvw_1);
        vec3f_t d3 = (*uvw_0) - (*uvw_2);
        feat[0] = Proj::intrinsic_distance(handle, *uvw_0, *uvw_1);
        feat[1] = Proj::intrinsic_distance(handle, *uvw_0, *uvw_2);
        feat[2] = Proj::intrinsic_distance(handle, *uvw_1, *uvw_2);
        feat[3] = angle(d1, tgt_1);
        feat[4] = angle(d2, tgt_2);
        feat[5] = angle(d3, tgt_0);
        vec3f_t nrm = d1.cross(d3);
        feat[6] = angle(nrm, vec3f_t::UnitZ());
        //feat[6] = crv_0.pc_max < Eigen::NumTraits<float>::dummy_precision() ? 1.f : crv_0.pc_min / crv_0.pc_max;
        //feat[7] = crv_1.pc_max < Eigen::NumTraits<float>::dummy_precision() ? 1.f : crv_1.pc_min / crv_1.pc_max;
        //feat[8] = crv_2.pc_max < Eigen::NumTraits<float>::dummy_precision() ? 1.f : crv_2.pc_min / crv_2.pc_max;
    //}

    // this changes norm which is irrelevant for the following
    // angle computation thanks to atan2(y, x) ~ atan(y/x) = atan(ky/kx)
    /* plane2
    tgt_0[2] = 0.f;
    tgt_1[2] = 0.f;
    tgt_2[2] = 0.f;
    vec3f_t c1 = tgt_0.cross(tgt_1);
    vec3f_t c2 = tgt_0.cross(tgt_2);
    vec3f_t c3 = tgt_1.cross(tgt_2);
    feat[1] = atan2f(c1.norm(), tgt_0.dot(tgt_1));
    feat[2] = atan2f(c2.norm(), tgt_0.dot(tgt_2));
    feat[3] = atan2f(c3.norm(), tgt_1.dot(tgt_2));
    */

    return feat;
}

template <typename Proj, typename Point>
inline discrete_feature_t
discretize_feature(typename Proj::const_handle_t handle, const feature_t& f, const feature_bounds_t& bounds, const discretization_params& params) {
    discrete_feature_t df;
    //if constexpr (scale_invariant) {
        //df <<
            //discretize(f[0], bounds.min()[0], bounds.diagonal()[0], params.distance_step_count),
            //discretize(f[1], params.angle_step),
            //discretize(f[2], params.angle_step),
            //discretize(f[3], params.angle_step);
    //} else {
        df <<
            discretize(f[0], bounds.min()[0], bounds.diagonal()[0], params.distance_step_count),
            discretize(f[1], bounds.min()[1], bounds.diagonal()[1], params.distance_step_count),
            discretize(f[2], bounds.min()[2], bounds.diagonal()[2], params.distance_step_count),
            discretize(f[3], params.angle_step),
            discretize(f[4], params.angle_step),
            discretize(f[5], params.angle_step),
            discretize(f[6], params.angle_step);
            //discretize(f[6], bounds.min()[6], bounds.diagonal()[6], params.curvature_ratio_step_count),
            //discretize(f[7], bounds.min()[7], bounds.diagonal()[7], params.curvature_ratio_step_count),
            //discretize(f[8], bounds.min()[8], bounds.diagonal()[8], params.curvature_ratio_step_count);
    //}
    return df;
}

template <typename Proj, typename Point>
inline bool
valid(typename Proj::const_handle_t handle, const feature_t& f, const feature_bounds_t& bounds) {
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
