namespace triplet_match {

template <typename Point>
inline typename pointcloud<Point>::curvature_info_t
principal_curvatures(const pointcloud<Point>& cloud, uint32_t p_idx, const std::vector<int> &indices) {
    mat3f_t I = mat3f_t::Identity();
    vec3f_t n_idx  = cloud.points[p_idx].getNormalVector3fMap();;
    mat3f_t M = I - n_idx * n_idx.transpose();

    std::vector<vec3f_t> proj_normals(indices.size());
    vec3f_t centroid = vec3f_t::Zero();
    for (size_t idx = 0; idx < indices.size(); ++idx) {
        proj_normals[idx] = M * cloud.points[indices[idx]].getNormalVector3fMap();
        centroid += (proj_normals[idx] - centroid) / (idx+1);
    }

    mat3f_t cov = mat3f_t::Zero();
    for (size_t idx = 0; idx < indices.size (); ++idx) {
        vec3f_t demean = proj_normals[idx] - centroid;

        double demean_xy = demean[0] * demean[1];
        double demean_xz = demean[0] * demean[2];
        double demean_yz = demean[1] * demean[2];

        cov(0, 0) += demean[0] * demean[0];
        cov(0, 1) += static_cast<float> (demean_xy);
        cov(0, 2) += static_cast<float> (demean_xz);

        cov(1, 0) += static_cast<float> (demean_xy);
        cov(1, 1) += demean[1] * demean[1];
        cov(1, 2) += static_cast<float> (demean_yz);

        cov(2, 0) += static_cast<float> (demean_xz);
        cov(2, 1) += static_cast<float> (demean_yz);
        cov(2, 2) += demean[2] * demean[2];
    }

    vec3f_t evs, evec;
    pcl::eigen33 (cov, evs);
    pcl::computeCorrespondingEigenVector (cov, evs [2], evec);

    float area_inv = 1.0f / static_cast<float>(indices.size());
    return {evec, evs[1] * area_inv, evs[2] * area_inv};
}

template <typename Point>
inline
pointcloud<Point>::~pointcloud() {
}

template <typename Point>
inline typename pointcloud<Point>::Ptr
pointcloud<Point>::empty() {
    Ptr self(new pointcloud<Point>());
    return self;
}

template <typename Point>
inline typename pointcloud<Point>::Ptr
pointcloud<Point>::from_pcd(const std::string& filename) {
    Ptr self = empty();
    pcl::io::loadPCDFile(filename, *self);
    return self;
}

template <typename Point>
inline float
pointcloud<Point>::resolution() const {
    auto tree = kdtree();
    mutex_.lock();
    if (resolution_ < 0.f) {
        constexpr uint32_t n = 30;
        resolution_ = average(vw::sample(this->points, n) | vw::transform([&](const Point& pnt) {
            std::vector<int> is(2);
            std::vector<float> ds(2);
            tree->nearestKSearch(pnt, 2, is, ds);
            return sqrtf(ds[1]);
        }), 0.f);
    }
    mutex_.unlock();
    return resolution_;
}

template <typename Point>
inline float
pointcloud<Point>::diameter() const {
    mutex_.lock();
    if (diameter_ < 0.f) {
        bbox3_t bbox;
        for (const auto& pnt : this->points) {
            bbox.extend(pnt.getVector3fMap());
        }
        diameter_ = bbox.diagonal().norm();
    }
    mutex_.unlock();
    return diameter_;
}

template <typename Point>
inline void
pointcloud<Point>::set_indices(subset_t subset) {
    mutex_.lock();
    subset_ = std::move(subset);
    tree_.reset();
    mutex_.unlock();
}

template <typename Point>
void
pointcloud<Point>::reset_tree() {
    mutex_.lock();
    tree_.reset();
    mutex_.unlock();
}

template <typename Point>
inline typename pointcloud<Point>::tree_t::Ptr
pointcloud<Point>::kdtree() {
    mutex_.lock();
    if (!tree_) {
        build_tree_();
    }
    mutex_.unlock();
    return tree_;
}

template <typename Point>
inline typename pointcloud<Point>::tree_t::ConstPtr
pointcloud<Point>::kdtree() const {
    mutex_.lock();
    if (!tree_) {
        build_tree_();
    }
    mutex_.unlock();
    return tree_;
}

template <typename Point>
inline std::pair<std::vector<int>, std::vector<float>>
pointcloud<Point>::knn_inclusive(uint32_t k, const Point& point) const {
    auto tree = kdtree();
    std::vector<int> is(k);
    std::vector<float> ds(k);
    tree->nearestKSearch(point, k, is, ds);
    return {is, ds};
}

template <typename Point>
inline std::pair<std::vector<int>, std::vector<float>>
pointcloud<Point>::knn_inclusive(uint32_t k, uint32_t idx) const {
    return knn_inclusive(k, this->points[idx]);
}

template <typename Point>
inline std::pair<std::vector<int>, std::vector<float>>
pointcloud<Point>::knn_exclusive(uint32_t k, const Point& point) const {
    auto && [is, ds] = knn_inclusive(k+1, point);
    std::vector<int> is_ex = vw::tail(is) | ranges::to_vector;
    std::vector<float> ds_ex = vw::tail(ds) | ranges::to_vector;
    return {is_ex, ds_ex};
}

template <typename Point>
inline std::pair<std::vector<int>, std::vector<float>>
pointcloud<Point>::knn_exclusive(uint32_t k, uint32_t idx) const {
    return knn_exclusive(k, this->points[idx]);
}

template <typename Point>
inline std::pair<std::vector<int>, std::vector<float>>
pointcloud<Point>::radius_search_inclusive(float r, const Point& point) const {
    auto tree = kdtree();
    std::vector<int> is;
    std::vector<float> ds;
    tree->radiusSearch(point, r, is, ds);
    return {is, ds};
}

template <typename Point>
inline std::pair<std::vector<int>, std::vector<float>>
pointcloud<Point>::radius_search_inclusive(float r, uint32_t idx) const {
    return radius_search_inclusive(r, this->points[idx]);
}

template <typename Point>
inline std::pair<std::vector<int>, std::vector<float>>
pointcloud<Point>::radius_search_exclusive(float r, const Point& point) const {
    auto && [is, ds] = radius_search_inclusive(r, point);
    std::vector<int> is_ex = vw::tail(is) | ranges::to_vector;
    std::vector<float> ds_ex = vw::tail(ds) | ranges::to_vector;
    return {is_ex, ds_ex};
}

template <typename Point>
inline std::pair<std::vector<int>, std::vector<float>>
pointcloud<Point>::radius_search_exclusive(float r, uint32_t idx) const {
    return radius_search_exclusive(r, this->points[idx]);
}

template <typename Point>
inline typename pointcloud<Point>::curvature_info_t
pointcloud<Point>::curvature(uint32_t k, uint32_t idx) const {
    return principal_curvatures(*this, idx, knn_inclusive(k, idx).first);
}

template <typename Point>
inline typename pointcloud<Point>::curvature_info_t
pointcloud<Point>::curvature(float r, uint32_t idx) const {
    return principal_curvatures(*this, idx, radius_search_inclusive(r, idx).first);
}

template <typename Point>
inline typename pointcloud<Point>::base_t::Ptr
pointcloud<Point>::native() {
    return boost::dynamic_pointer_cast<base_t>(this->shared_from_this());
}

template <typename Point>
inline typename pointcloud<Point>::base_t::ConstPtr
pointcloud<Point>::native() const {
    return boost::dynamic_pointer_cast<const base_t>(this->shared_from_this());
}

template <typename Point>
inline
pointcloud<Point>::pointcloud() : base_t(), resolution_(-1.f), diameter_(-1.f), tree_(nullptr) {
}

template <typename Point>
void
pointcloud<Point>::build_tree_() const {
    pcl::IndicesPtr indices;
    if (subset_.empty()) {
        indices = pcl::IndicesPtr(new std::vector<int>(this->points.size()));
        std::iota(indices->begin(), indices->end(), 0);
    } else {
        indices = pcl::IndicesPtr(
            new std::vector<int>(subset_.begin(), subset_.end()));
    }
    tree_ = typename tree_t::Ptr(new tree_t(false));
    tree_->setInputCloud(this->shared_from_this(), indices);
}

template <typename Point>
inline const std::pair<int, float>&
knn_range<Point>::read() {
    return ns_[crt_];
}

template <typename Point>
inline bool
knn_range<Point>::equal(ranges::default_sentinel) const {
    return crt_ >= ns_.size();
}

template <typename Point>
inline void
knn_range<Point>::next() {
    ++crt_;
}

template <typename Point>
inline
knn_range<Point>::knn_range(const pointcloud<Point>& cloud, uint32_t k, const Point& pnt) {
    auto ns = cloud.knn_inclusive(k, pnt);
    ns_.resize(ns.first.size());
    for (uint32_t i = 0; i < ns.first.size(); ++i) {
        ns_[i] = {ns.first[i], ns.second[i]};
    }
}

template <typename Point>
inline
knn_range<Point>::knn_range(typename pointcloud<Point>::ConstPtr cloud, uint32_t k, const Point& pnt) : knn_range(*cloud, k, pnt) {
}

template <typename Point>
inline int
knn_index_range<Point>::read() {
    return ns_[crt_];
}

template <typename Point>
inline bool
knn_index_range<Point>::equal(ranges::default_sentinel) const {
    return crt_ >= ns_.size();
}

template <typename Point>
inline void
knn_index_range<Point>::next() {
    ++crt_;
}

template <typename Point>
inline
knn_index_range<Point>::knn_index_range(const pointcloud<Point>& cloud, uint32_t k, const Point& pnt) {
    auto ns = cloud.knn_inclusive(k, pnt);
    ns_.resize(ns.first.size());
    for (uint32_t i = 0; i < ns.first.size(); ++i) {
        ns_[i] = ns.first[i];
    }
}

template <typename Point>
inline
knn_index_range<Point>::knn_index_range(typename pointcloud<Point>::ConstPtr cloud, uint32_t k, const Point& pnt) : knn_index_range(*cloud, k, pnt) {
}

template <typename Point>
inline float
knn_dist_range<Point>::read() {
    return sqrtf(ns_[crt_]);
}

template <typename Point>
inline bool
knn_dist_range<Point>::equal(ranges::default_sentinel) const {
    return crt_ >= ns_.size();
}

template <typename Point>
inline void
knn_dist_range<Point>::next() {
    ++crt_;
}

template <typename Point>
inline
knn_dist_range<Point>::knn_dist_range(const pointcloud<Point>& cloud, uint32_t k, const Point& pnt) {
    auto ns = cloud.knn_inclusive(k, pnt);
    ns_.resize(ns.first.size());
    for (uint32_t i = 0; i < ns.first.size(); ++i) {
        ns_[i] = ns.second[i];
    }
}

template <typename Point>
inline
knn_dist_range<Point>::knn_dist_range(typename pointcloud<Point>::ConstPtr cloud, uint32_t k, const Point& pnt) : knn_dist_range(*cloud, k, pnt) {
}

template <typename Point>
inline float
knn_sqr_dist_range<Point>::read() {
    return ns_[crt_];
}

template <typename Point>
inline bool
knn_sqr_dist_range<Point>::equal(ranges::default_sentinel) const {
    return crt_ >= ns_.size();
}

template <typename Point>
inline void
knn_sqr_dist_range<Point>::next() {
    ++crt_;
}

template <typename Point>
inline
knn_sqr_dist_range<Point>::knn_sqr_dist_range(const pointcloud<Point>& cloud, uint32_t k, const Point& pnt) {
    auto ns = cloud.knn_inclusive(k, pnt);
    ns_.resize(ns.first.size());
    for (uint32_t i = 0; i < ns.first.size(); ++i) {
        ns_[i] = ns.second[i];
    }
}

template <typename Point>
inline
knn_sqr_dist_range<Point>::knn_sqr_dist_range(typename pointcloud<Point>::ConstPtr cloud, uint32_t k, const Point& pnt) : knn_sqr_dist_range(*cloud, k, pnt) {
}

} // triplet_match
