namespace triplet_match {

template <typename Point>
inline typename plane_traits<Point>::handle_t
plane_traits<Point>::init_from_model(typename cloud_t::ConstPtr cloud,
                                     const subset_t& subset) {
    subset_t indices(subset.begin(), subset.end());
    if (indices.empty()) {
        indices.resize(cloud->size());
        std::iota(indices.begin(), indices.end(), 0);
    }

    // compute centroid
    vec3f_t centroid = vec3f_t::Zero();
    for (auto && [count, idx] : vw::zip(vw::ints(1u), indices)) {
        centroid += (cloud->points[idx].getVector3fMap() - centroid) / count;
    }

    // compute scatter matrix
    matf_t scatter(indices.size(), 3);
    for (auto && [row, idx] : vw::zip(vw::ints(0u), indices)) {
        scatter.row(row) = (cloud->points[idx].getVector3fMap() - centroid).transpose();
    }

    Eigen::JacobiSVD<matf_t> svd(scatter, Eigen::ComputeThinU | Eigen::ComputeThinV);

    auto handle = std::make_shared<state_t>();
    handle->origin = centroid;
    handle->g2l = mat4f_t::Identity();
    handle->g2l.template topLeftCorner<3, 3>() = svd.matrixV().transpose();
    handle->g2l.template block<3,1>(0, 3) = handle->g2l.template topLeftCorner<3, 3>() * (-handle->origin);
    handle->l2g = handle->g2l.inverse();

    // compute uvw bounds
    for (auto idx : indices) {
        handle->uvw_bounds.extend(project(handle, cloud->points[idx].getVector3fMap()));
    }

    return handle;
}

template <typename Point>
inline typename plane_traits<Point>::handle_t
plane_traits<Point>::init_from_samples(
    const std::array<Point, sample_count>& samples) {
    // given normals 1 point fully determines the plane - however rotation around normal is a DOF
    // since we correct for (translations and) rotations in uvw space we don't bother though
    Point sample1 = std::get<0>(samples);
    auto handle = std::make_shared<state_t>();
    handle->origin = sample1.getVector3fMap();
    handle->g2l = mat4f_t::Identity();
    handle->g2l.template block<1,3>(2, 0) = sample1.getNormalVector3fMap();
    handle->g2l.template block<1,3>(1, 0) = sample1.getNormalVector3fMap().unitOrthogonal();
    handle->g2l.template block<1,3>(0, 0) = handle->g2l.template block<1,3>(1, 0).cross(handle->g2l.template block<1,3>(2, 0)).normalized();
    handle->g2l.template block<3,1>(0, 3) = handle->g2l.template topLeftCorner<3, 3>() * (-handle->origin);
    handle->l2g = handle->g2l.inverse();

    return handle;
}

template <typename Point>
inline vec3f_t
plane_traits<Point>::project(const_handle_t handle, const vec3f_t& xyz) {
    return (handle->g2l * xyz.homogeneous()).head(3);
}

template <typename Point>
inline vec3f_t
plane_traits<Point>::unproject(const_handle_t handle, const vec3f_t& uvw) {
    return (handle->l2g * uvw.homogeneous()).head(3);
}

template <typename Point>
inline vec3f_t
plane_traits<Point>::tangent(const_handle_t handle, const Point& pnt) {
    return handle->g2l.template topLeftCorner<3, 3>() *
           vec3f_t(pnt.data_c[1], pnt.data_c[2], pnt.data_c[3]);
}

template <typename Point>
inline float
plane_traits<Point>::intrinsic_distance(const_handle_t handle,
                                        const vec3f_t& p0,
                                        const vec3f_t& p1) {
    return (p1 - p0).norm();
}

template <typename Point>
inline typename plane_traits<Point>::feature_t
plane_traits<Point>::feature(const_handle_t handle, const Point& pnt_0,
                             const Point& pnt_1) {
    // planes use the same features as cylinders (or all "pseudo-2d-embeddings" for that matter)
    // i.e. intrinsic distance (in this case simple euclidean L2) and intrinsic tangent angles (with w=0)
    vec3f_t uvw_0 = project(handle, pnt_0.getVector3fMap());
    vec3f_t uvw_1 = project(handle, pnt_1.getVector3fMap());
    vec3f_t tgt_0 = tangent(handle, pnt_0);
    vec3f_t tgt_1 = tangent(handle, pnt_1);

    vec2f_t feat;
    feat[0] = intrinsic_distance(handle, uvw_0, uvw_1);

    // this changes norm which is irrelevant for the following
    // angle computation thanks to atan2(y, x) ~ atan(y/x) = atan(ky/kx)
    tgt_0[2] = 0.f;
    tgt_1[2] = 0.f;
    vec3f_t c = tgt_0.cross(tgt_1);
    feat[1] = atan2f(c.norm(), tgt_0.dot(tgt_1));
    return feat;
}

template <typename Point>
inline typename plane_traits<Point>::discrete_feature_t
plane_traits<Point>::discretize_feature(const_handle_t handle, const feature_t& f, const feature_bounds_t& bounds, const discretization_params& params) {
    discrete_feature_t df;
    df <<
        discretize(f[0], bounds.min()[0], bounds.diagonal()[0], params.distance_step_count),
        discretize(f[1], params.angle_step);
    return df;
}

template <typename Point>
inline bool
plane_traits<Point>::valid(const_handle_t handle, const feature_t& f, const feature_bounds_t& bounds, float min_angle, float max_angle, float min_rel_dist, float max_rel_dist) {
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
//plane_traits<Point>::opencl_source() {
    //return "plane.cl";
//}

}  // namespace triplet_match
