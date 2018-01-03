namespace triplet_match {

template <typename Point>
inline typename identity_traits<Point>::handle_t
identity_traits<Point>::init_from_model(typename cloud_t::ConstPtr cloud,
                                     const subset_t& subset) {
    subset_t indices(subset.begin(), subset.end());
    if (indices.empty()) {
        indices.resize(cloud->size());
        std::iota(indices.begin(), indices.end(), 0);
    }

    auto handle = std::make_shared<state_t>();

    // compute uvw bounds
    for (auto idx : indices) {
        handle->uvw_bounds.extend(cloud->points[idx].getVector3fMap());
    }

    return handle;
}

template <typename Point>
inline typename identity_traits<Point>::handle_t
identity_traits<Point>::init_from_samples(
    const std::array<Point, sample_count>& samples) {
    auto handle = std::make_shared<state_t>();
    return handle;
}

template <typename Point>
inline vec3f_t
identity_traits<Point>::project(const_handle_t handle, const vec3f_t& xyz) {
    return xyz;
}

template <typename Point>
inline vec3f_t
identity_traits<Point>::unproject(const_handle_t handle, const vec3f_t& uvw) {
    return uvw;
}

template <typename Point>
inline vec3f_t
identity_traits<Point>::tangent(const_handle_t handle, const Point& pnt) {
    return vec3f_t(pnt.data_c[1], pnt.data_c[2], pnt.data_c[3]);
}

template <typename Point>
inline float
identity_traits<Point>::intrinsic_distance(const_handle_t handle,
                                        const vec3f_t& p0,
                                        const vec3f_t& p1) {
    return (p1 - p0).norm();
}

template <typename Point>
inline typename identity_traits<Point>::feature_t
identity_traits<Point>::feature(const_handle_t handle, const Point& pnt_0,
                             const Point& pnt_1) {
    // planes use the same features as cylinders (or all "pseudo-2d-embeddings" for that matter)
    // i.e. intrinsic distance (in this case simple euclidean L2) and intrinsic tangent angles (with w=0)
    vec3f_t uvw_0 = pnt_0.getVector3fMap();
    vec3f_t uvw_1 = pnt_1.getVector3fMap();
    vec3f_t tgt_0 = vec3f_t(pnt_0.data_c[1], pnt_0.data_c[2], pnt_0.data_c[3]);
    vec3f_t tgt_1 = vec3f_t(pnt_1.data_c[1], pnt_1.data_c[2], pnt_1.data_c[3]);

    vec2f_t feat;
    feat[0] = (uvw_1 - uvw_0).norm();

    // this changes norm which is irrelevant for the following
    // angle computation thanks to atan2(y, x) ~ atan(y/x) = atan(ky/kx)
    tgt_0[2] = 0.f;
    tgt_1[2] = 0.f;
    vec3f_t c = tgt_0.cross(tgt_1);
    feat[1] = atan2f(c.norm(), tgt_0.dot(tgt_1));
    return feat;
}

template <typename Point>
inline typename identity_traits<Point>::discrete_feature_t
identity_traits<Point>::discretize_feature(const_handle_t handle, const feature_t& f, const feature_bounds_t& bounds, const discretization_params& params) {
    discrete_feature_t df;
    df <<
        discretize(f[0], bounds.min()[0], bounds.diagonal()[0], params.distance_step_count),
        discretize(f[1], params.angle_step);
    return df;
}

template <typename Point>
inline bool
identity_traits<Point>::valid(const_handle_t handle, const feature_t& f, const feature_bounds_t& bounds, float min_angle, float max_angle, float min_rel_dist, float max_rel_dist) {
    // map distance into [0, 1] range of occuring distances
    float dist = f[0] - bounds.min()[0];
    if (bounds.diagonal()[0] > Eigen::NumTraits<float>::dummy_precision()) {
        dist /= bounds.diagonal()[0];
    }
    if (dist < min_rel_dist || dist > max_rel_dist) {
        return false;
    }
    return (f[1] >= min_angle && f[1] <= max_angle);
}

//template <typename Point>
//inline std::string
//identity_traits<Point>::opencl_source() {
    //return "plane.cl";
//}

}  // namespace triplet_match
