namespace triplet_match {

template <typename Point>
struct model<Point>::impl {
    impl(typename cloud_t::ConstPtr cloud, discretization_params params) : cloud_(cloud), params_(params), init_(false) {
    }

    ~impl() {}

    std::future<void>
    init(const subset_t& subset) {
        if (subset.empty()) {
            subset_.resize(cloud_->size());
            std::iota(subset_.begin(), subset_.end(), 0u);
        } else {
            subset_ = subset;
        }

        return std::async(std::launch::async, [&] () {
            this->build_model_();
        });
    }

    template <typename PointQuery>
    std::pair<triplet_iter_t, triplet_iter_t>
    query(const PointQuery& p1, const PointQuery& p2, const PointQuery& p3) {
        if (!init_) {
            throw std::runtime_error("Cannot query uninitialized model");
        }

        discrete_feature df = compute_discrete<PointQuery>(p1, p2, p3, params_);

        return map_.equal_range(df);
    }

    float diameter() const {
        return diameter_;
    }

    typename cloud_t::ConstPtr cloud() const {
        return cloud_;
    }

    void
    build_model_() {
        bbox3_t bbox;
        for (uint32_t i : subset_) {
            const Point& p1 = cloud_->points[i];
            if (!pcl::isFinite(p1)) {
                continue;
            }

            bbox.extend(p1.getVector3fMap());

            for (uint32_t j : subset_) {
                const Point& p2 = cloud_->points[j];
                if (i == j || !pcl::isFinite(p2)) {
                    continue;
                }

                for (uint32_t k : subset_) {
                    const Point& p3 = cloud_->points[k];
                    if (i == k || j == k || !pcl::isFinite(p3)) {
                        continue;
                    }

                    discrete_feature df = compute_discrete<Point>(p1, p2, p3, params_);

                    map_.insert({df, triplet_t{i, j, k}});
                }
            }
        }

        diameter_ = (bbox.max() - bbox.min()).norm();
        init_ = true;
    }

    typename cloud_t::ConstPtr cloud_;
    discretization_params params_;
    bool init_;

    subset_t subset_;
    hash_map_t map_;
    float diameter_;
};


template <typename Point>
inline
model<Point>::model(typename cloud_t::ConstPtr cloud, discretization_params params) {
    impl_ = std::make_unique<impl>(cloud, params);
}

template <typename Point>
inline
model<Point>::~model() {
}

template <typename Point>
inline std::future<void>
model<Point>::init() {
    return impl_->init(subset_t());
}

template <typename Point>
inline std::future<void>
model<Point>::init(const subset_t& subset) {
    return impl_->init(subset);
}

template <typename Point>
template <typename PointQuery>
inline range<typename model<Point>::triplet_iter_t>
model<Point>::query(const PointQuery& p1, const PointQuery& p2, const PointQuery& p3) {
    return range<triplet_iter_t>(impl_->template query<PointQuery>(p1, p2, p3));
}

template <typename Point>
inline float
model<Point>::diameter() const {
    return impl_->diameter();
}

template <typename Point>
inline typename model<Point>::cloud_t::ConstPtr
model<Point>::cloud() const {
    return impl_->cloud();
}

}  // namespace triplet_match
