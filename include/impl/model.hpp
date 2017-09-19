#include <fstream>
#include <range/v3/all.hpp>

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

        //float lower = diameter_ * s_params_.min_diameter_factor;
        //float upper = diameter_ * s_params_.max_diameter_factor;
        //discrete_feature df = compute_discrete<PointQuery>(p1, p2, p3, params_, lower, upper-lower);
        float lower_ratio = s_params_.min_triplet_ratio;
        float upper_ratio = s_params_.max_triplet_ratio;
        discrete_feature df = compute_discrete<PointQuery>(p1, p2, p3, params_, lower_ratio, upper_ratio-lower_ratio);

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
        //float range = upper-lower;
        float lower_ratio = s_params_.min_triplet_ratio;
        float upper_ratio = s_params_.max_triplet_ratio;
        float range_ratio = upper_ratio - lower_ratio;
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
                if (dist1 < lower || dist1 > upper) {
                    continue;
                }

                for (uint32_t k : valid) {
                    const Point& p3 = cloud_->points[k];
                    if (i == k || j == k) {
                        continue;
                    }

                    vec3f_t d2 = (p3.getVector3fMap() - p2.getVector3fMap());
                    vec3f_t d3 = (p3.getVector3fMap() - p1.getVector3fMap());
                    float dist2 = d2.norm();
                    float dist3 = d3.norm();
                    if (dist2 < lower || dist2 > upper || dist3 < lower || dist3 > upper) {
                        continue;
                    }
                    float rat0 = dist2 / dist1;
                    float rat1 = dist3 / dist1;
                    if (rat0 < lower_ratio || rat0 > upper_ratio || rat1 < lower_ratio || rat1 > upper_ratio) {
                        continue;
                    }

                    ++triplet_count_;

                    used_points_.insert(i);
                    used_points_.insert(j);
                    used_points_.insert(k);
                    discrete_feature df = compute_discrete<Point>(p1, p2, p3, params_, lower_ratio, range_ratio);

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
    std::set<uint32_t> used_points_;
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

template <typename Point>
inline std::set<uint32_t>
model<Point>::used_points() const {
    return impl_->used_points_;
}

template <typename Point>
void
model<Point>::write_octave_density_maps(const std::string& folder, const std::string& data_file_prefix, const std::string& script_file) const {
    const discretization_params& ps = impl_->params_;
    const hash_map_t& map = impl_->map_;

    const float max_angle = static_cast<float>(M_PI);
    const uint32_t angle_count = static_cast<uint32_t>(max_angle / ps.angle_step) + 1;
    const uint32_t dist_count = static_cast<uint32_t>(ps.distance_step_count);

    std::vector<std::vector<uint32_t>> hists(5);
    hists[0] = std::vector<uint32_t>(dist_count, 0);
    hists[1] = std::vector<uint32_t>(dist_count, 0);
    hists[2] = std::vector<uint32_t>(angle_count, 0);
    hists[3] = std::vector<uint32_t>(angle_count, 0);
    hists[4] = std::vector<uint32_t>(angle_count, 0);
    auto missing_dims = ranges::view::cartesian_product(
        ranges::view::ints(0u, dist_count),
        ranges::view::ints(0u, dist_count),
        ranges::view::ints(0u, angle_count),
        ranges::view::ints(0u, angle_count),
        ranges::view::ints(0u, angle_count)
    );
    for (auto && [a,b,c,d,e] : missing_dims) {
        discrete_feature feat;
        feat[0] = a;
        feat[1] = b;
        feat[2] = c;
        feat[3] = d;
        feat[4] = e;
        auto && [fst,lst] = map.equal_range(feat);
        uint32_t count = std::distance(fst, lst);
        hists[0][a] += count;
        hists[1][b] += count;
        hists[2][c] += count;
        hists[3][d] += count;
        hists[4][e] += count;
    }

    auto name = [&] (uint32_t i) { return data_file_prefix + "_" + std::to_string(i); };

    std::ofstream out;
    for (uint32_t i = 0; i < 5; ++i) {
        out.open(folder + "/" + name(i) + ".dat");
        for (const auto& count : hists[i]) {
            out << count << "\n";
        }
        out.close();
    }

    out.open(folder + "/" + script_file);
    for (uint32_t i = 0; i < 5; ++i) {
        out << "load " << name(i) << ".dat" << "\n";
        out << "figure(" << (i+1) << ")" << "\n";
        out << "plot(0:numel(" << name(i) << ")-1, " << name(i) << ", '-')" << "\n";
    }
    out.close();
}

}  // namespace triplet_match
