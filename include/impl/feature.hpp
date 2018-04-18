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
        const Point& pnt_1, const curv_info_t<Point>& crv_0, const curv_info_t<Point>& crv_1) {
    // planes use the same features as cylinders (or all "pseudo-2d-embeddings" for that matter)
    // i.e. intrinsic distance (in this case simple euclidean L2) and intrinsic tangent angles (with w=0)
    vec3f_t p0 = pnt_0.getVector3fMap();
    vec3f_t p1 = pnt_1.getVector3fMap();
    vec3f_t tgt_0 = tangent(pnt_0);
    vec3f_t tgt_1 = tangent(pnt_1);

    feature_t feat;
    vec3f_t d0 = p1 - p0;
    feat[0] = d0.norm();
    feat[1] = angle(d0, tgt_0);
    feat[2] = angle(d0, tgt_1);
    //vec3f_t nrm = tgt_0.cross(tgt_1);
    feat[3] = feat[0];//crv_0.pc_min / crv_0.pc_max; //angle(nrm, vec3f_t::UnitZ());

    return feat;
}

template <typename Point>
inline discrete_feature_t
discretize_feature(const feature_t& f, const feature_bounds_t& bounds, const discretization_params& params) {
    discrete_feature_t df;
    df <<
        discretize(f[0], bounds.min()[0], bounds.diagonal()[0], params.distance_step_count),
        discretize(f[1], params.angle_step),
        discretize(f[2], params.angle_step),
        discretize(f[3], bounds.min()[0], bounds.diagonal()[0], params.distance_step_count);//discretize(f[3], bounds.min()[3], bounds.diagonal()[3], params.curvature_ratio_step_count);
    return df;
}

template <typename Point>
inline bool
valid(const feature_t& f, const feature_bounds_t& bounds) {
    if (f[0] < bounds.min()[0] || f[0] > bounds.max()[0]) {
        return false;
    }
    //if (f[3] < bounds.min()[3] || f[3] > bounds.max()[3]) {
        //return false;
    //}
    //if constexpr (!scale_invariant) {
        //if (f[1] < bounds.min()[1] || f[1] > bounds.max()[1]) {
            //return false;
        //}
        //if (f[2] < bounds.min()[2] || f[2] > bounds.max()[2]) {
            //return false;
        //}
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
        angle_ok = (f[1] >= 0.f && f[1] <= pi) && (f[2] >= 0.f && f[2] <= pi);// && (f[3] >= 0.f && f[3] <= pi);
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
    new_bounds.min()[3] = bounds.min()[3] + min_rel_dist * bounds.diagonal()[3];
    new_bounds.max()[3] = bounds.min()[3] + max_rel_dist * bounds.diagonal()[3];
    //if constexpr (!scale_invariant) {
        //new_bounds.min()[1] = bounds.min()[1] + min_rel_dist * bounds.diagonal()[1];
        //new_bounds.max()[1] = bounds.min()[1] + max_rel_dist * bounds.diagonal()[1];
        //new_bounds.min()[2] = bounds.min()[2] + min_rel_dist * bounds.diagonal()[2];
        //new_bounds.max()[2] = bounds.min()[2] + max_rel_dist * bounds.diagonal()[2];
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
