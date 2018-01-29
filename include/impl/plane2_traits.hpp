namespace triplet_match {

template <typename Point>
inline typename plane2_traits<Point>::handle_t
plane2_traits<Point>::init_from_model(typename cloud_t::ConstPtr cloud,
                                      const subset_t& subset) {
    subset_t indices(subset.begin(), subset.end());
    if (indices.empty()) {
        indices.resize(cloud->size());
        std::iota(indices.begin(), indices.end(), 0);
    }

    // compute centroid
    vec3f_t centroid = vec3f_t::Zero();
    for (auto&& [count, idx] : vw::zip(vw::ints(1u), indices)) {
        centroid += (cloud->points[idx].getVector3fMap() - centroid) / count;
    }

    // compute scatter matrix
    matf_t scatter(indices.size(), 3);
    for (auto&& [row, idx] : vw::zip(vw::ints(0u), indices)) {
        scatter.row(row) =
            (cloud->points[idx].getVector3fMap() - centroid).transpose();
    }

    Eigen::JacobiSVD<matf_t> svd(scatter,
                                 Eigen::ComputeThinU | Eigen::ComputeThinV);

    auto handle = std::make_shared<state_t>();
    handle->origin = centroid;
    handle->g2l = mat4f_t::Identity();
    handle->g2l.template topLeftCorner<3, 3>() = svd.matrixV().transpose();
    handle->g2l.template block<3, 1>(0, 3) =
        handle->g2l.template topLeftCorner<3, 3>() * (-handle->origin);
    handle->l2g = handle->g2l.inverse();

    handle->threshold = 0.f;
    for (auto idx : indices) {
        vec4f_t xyz = cloud->points[idx].getVector3fMap().homogeneous();
        handle->threshold =
            std::max(handle->threshold, fabs(handle->g2l.row(2).dot(xyz)));
    }
    handle->threshold *= 2.f;

    return handle;
}

template <typename Point>
inline typename plane2_traits<Point>::handle_t
plane2_traits<Point>::init_from_samples(const const_handle_t& model_handle,
                                        const samples_t& samples) {
    vec3f_t sample1 = std::get<0>(samples).getVector3fMap();
    vec3f_t sample2 = std::get<1>(samples).getVector3fMap();
    vec3f_t sample3 = std::get<2>(samples).getVector3fMap();
    auto handle = std::make_shared<state_t>();
    handle->threshold = model_handle->threshold;
    handle->origin = sample1;

    vec3f_t d1 = sample2 - sample1;
    vec3f_t d2 = sample3 - sample1;
    vec3f_t nrm = d1.cross(d2).normalized();

    float cosa1 = std::get<0>(samples).getNormalVector3fMap().dot(nrm);
    float cosa2 = std::get<1>(samples).getNormalVector3fMap().dot(nrm);
    float cosa3 = std::get<2>(samples).getNormalVector3fMap().dot(nrm);
    if (1.f - fabs(cosa1) > 0.1f) return nullptr;
    if (1.f - fabs(cosa2) > 0.1f) return nullptr;
    if (1.f - fabs(cosa3) > 0.1f) return nullptr;

    d1 -= d1.dot(nrm) * nrm;
    d1.normalize();
    d2 = nrm.cross(d1).normalized();
    handle->g2l = mat4f_t::Identity();
    handle->g2l.template block<1, 3>(2, 0) = nrm;
    handle->g2l.template block<1, 3>(1, 0) = d2;
    handle->g2l.template block<1, 3>(0, 0) = d1;
    handle->g2l.template block<3, 1>(0, 3) =
        handle->g2l.template topLeftCorner<3, 3>() * (-handle->origin);
    handle->l2g = handle->g2l.inverse();

    return handle;
}

template <typename Point>
inline std::optional<vec3f_t>
plane2_traits<Point>::project(const const_handle_t& handle, const vec3f_t& xyz) {
    return (handle->g2l * xyz.homogeneous()).head(3);
}

template <typename Point>
inline vec3f_t
plane2_traits<Point>::unproject(const const_handle_t& handle, const vec3f_t& uvw) {
    return (handle->l2g * uvw.homogeneous()).head(3);
}

template <typename Point>
inline vec3f_t
plane2_traits<Point>::tangent(const const_handle_t& handle, const Point& pnt) {
    return (handle->g2l.template topLeftCorner<3, 3>() *
            vec3f_t(pnt.data_c[1], pnt.data_c[2], pnt.data_c[3]))
        .normalized();
}

template <typename Point>
inline vec3f_t
plane2_traits<Point>::normal(const const_handle_t& handle, const Point& pnt) {
    return (handle->g2l.template topLeftCorner<3, 3>() *
            pnt.getNormalVector3fMap()).normalized();
}

template <typename Point>
inline float
plane2_traits<Point>::intrinsic_distance(const const_handle_t& handle,
                                         const vec3f_t& p0, const vec3f_t& p1) {
    return (p1 - p0).head(2).norm();
}

}  // namespace triplet_match
