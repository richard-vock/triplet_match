namespace triplet_match {

template <typename Point>
struct model<Point>::impl {
    impl(typename cloud_t::ConstPtr cloud, discretization_params params) : cloud_(cloud), params_(params), init_(false) {
    }

    ~impl() {}

    std::future<void>
    init(const subset_t& subset, const sample_parameters& params) {
        if (subset.empty()) {
            subset_.resize(cloud_->size());
            std::iota(subset_.begin(), subset_.end(), 0u);
        } else {
            subset_ = subset;
        }

        s_params_ = params;

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

    uint32_t point_count() const {
        return point_count_;
    }

    uint64_t triplet_count() const {
        return triplet_count_;
    }

    typename cloud_t::ConstPtr cloud() const {
        return cloud_;
    }

    void
    build_model_() {
        bbox3_t bbox;
        std::set<uint32_t> valid;
        for (uint32_t i : subset_) {
            const Point& p = cloud_->points[i];
            if (pcl::isFinite(p)) {
                valid.insert(i);
                bbox.extend(p.getVector3fMap());
            }
        }
        diameter_ = (bbox.max() - bbox.min()).norm();

        float lower = diameter_ * s_params_.min_diameter_factor;
        float upper = diameter_ * s_params_.max_diameter_factor;
        triplet_count_ = 0;
        for (uint32_t i : valid) {
            const Point& p1 = cloud_->points[i];

            for (uint32_t j : valid) {
                const Point& p2 = cloud_->points[j];
                if (i == j) {
                    continue;
                }

                vec3f_t d1 = p2.getVector3fMap() - p1.getVector3fMap();
                float dist1 = d1.norm();
                d1 /= dist1;
                if (dist1 < lower || dist1 > upper) {
                    continue;
                }

                for (uint32_t k : valid) {
                    const Point& p3 = cloud_->points[k];
                    if (i == k || j == k) {
                        continue;
                    }

                    vec3f_t d2 = (p3.getVector3fMap() - p1.getVector3fMap()).normalized();
                    float orthogonality = 1.f - fabs(d2.dot(d1));
                    if (orthogonality < s_params_.min_orthogonality) {
                        continue;
                    }

                    ++triplet_count_;

                    discrete_feature df = compute_discrete<Point>(p1, p2, p3, params_);

                    map_.insert({df, triplet_t{i, j, k}});
                }
            }
        }

        point_count_ = valid.size();
        init_ = true;
    }

    typename cloud_t::ConstPtr cloud_;
    discretization_params params_;
    bool init_;

    subset_t subset_;
    sample_parameters s_params_;
    hash_map_t map_;
    float diameter_;
    uint32_t point_count_;
    uint64_t triplet_count_;
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
model<Point>::init(const sample_parameters& sample_params) {
    return impl_->init(subset_t(), sample_params);
}

template <typename Point>
inline std::future<void>
model<Point>::init(const subset_t& subset, const sample_parameters& sample_params) {
    return impl_->init(subset, sample_params);
}

template <typename Point>
template <typename PointQuery>
inline std::pair<typename model<Point>::triplet_iter_t, typename model<Point>::triplet_iter_t>
model<Point>::query(const PointQuery& p1, const PointQuery& p2, const PointQuery& p3) {
    return range<triplet_iter_t>(impl_->template query<PointQuery>(p1, p2, p3));
}

template <typename Point>
inline float
model<Point>::diameter() const {
    return impl_->diameter();
}

template <typename Point>
inline uint32_t
model<Point>::point_count() const {
    return impl_->point_count();
}

template <typename Point>
inline uint64_t
model<Point>::triplet_count() const {
    return impl_->triplet_count();
}

template <typename Point>
inline typename model<Point>::cloud_t::ConstPtr
model<Point>::cloud() const {
    return impl_->cloud();
}

}  // namespace triplet_match
