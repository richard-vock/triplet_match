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
    //for (auto idx : indices) {
        //handle->uvw_bounds.extend(cloud->points[idx].getVector3fMap());
    //}

    return handle;
}

template <typename Point>
inline typename identity_traits<Point>::handle_t
identity_traits<Point>::init_from_samples(
    const_handle_t, const samples_t& samples) {
    auto handle = std::make_shared<state_t>();
    return handle;
}

template <typename Point>
inline std::optional<vec3f_t>
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
inline vec3f_t
identity_traits<Point>::normal(const_handle_t handle, const Point& pnt) {
    return pnt.getNormalVector3fMap();
}

template <typename Point>
inline float
identity_traits<Point>::intrinsic_distance(const_handle_t handle,
                                        const vec3f_t& p0,
                                        const vec3f_t& p1) {
    return (p1 - p0).norm();
}

}  // namespace triplet_match
