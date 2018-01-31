#include <fstream>
#include <range/v3/all.hpp>

//#include "debug.hpp"

namespace triplet_match {

template <typename Point>
struct model<Point>::impl {
    impl(typename cloud_t::ConstPtr cloud, discretization_params params) : cloud_(cloud), params_(params), init_(false) {
    }

    ~impl() {}

    void
    init(const subset_t& subset, const sample_parameters& params) {
        if (subset.empty()) {
            subset_.resize(cloud_->size());
            std::iota(subset_.begin(), subset_.end(), 0u);
        } else {
            subset_ = subset;
        }

        subset_ = vw::filter(subset_, [&] (uint32_t idx) {
            vec3f_t pos = cloud_->points[idx].getVector3fMap();
            vec3f_t nrm = cloud_->points[idx].getNormalVector3fMap();
            vec3f_t tgt = vec3f_t(cloud_->points[idx].data_c[1], cloud_->points[idx].data_c[2], cloud_->points[idx].data_c[3]);
            bool finite = pos.allFinite() && nrm.allFinite() && tgt.allFinite();
            return finite;
        }) | ranges::to_vector;

        s_params_ = params;

        bbox3_t bbox;
        for (uint32_t i : subset_) {
            const Point& p = cloud_->points[i];
            bbox.extend(p.getVector3fMap());
        }
        diameter_ = (bbox.max() - bbox.min()).norm();

        vec3f_t lower = bbox.min();
        vec3f_t upper = bbox.max();
        vec3f_t range = upper-lower;

        vec3f_t ext = (bbox.diagonal() / cloud_->resolution())
            .cwiseMax(vec3f_t::Constant(1.f));

        margin_ = 5;

        extents_ = (ext + vec3f_t::Constant(2.f*margin_)).template cast<int>();

        vec3f_t scale(
            range[0] < Eigen::NumTraits<float>::dummy_precision() ? 1.f : ext[0] / range[0],
            range[1] < Eigen::NumTraits<float>::dummy_precision() ? 1.f : ext[1] / range[1],
            range[2] < Eigen::NumTraits<float>::dummy_precision() ? 1.f : ext[2] / range[2]);
        to_voxel_ = mat4f_t::Identity();
        to_voxel_.block<3, 3>(0, 0) = scale.asDiagonal();
        to_voxel_.block<3, 1>(0, 3) = to_voxel_.template block<3, 3>(0, 0) * (-bbox.min())
                                + vec3f_t::Constant(static_cast<float>(margin_));
        // subvoxel-shift
        to_voxel_.template block<3, 1>(0, 3) -= vec3f_t::Constant(0.5f);

        mat4f_t inv = to_voxel_.inverse();

        uint32_t voxel_count = extents_[0] * extents_[1] * extents_[2];
        voxel_data_.resize(voxel_count);

        auto voxels = vw::cartesian_product(
            vw::ints(0, extents_[0]),
            vw::ints(0, extents_[1]),
            vw::ints(0, extents_[2])
        );

        std::vector<typename pointcloud<Point>::curvature_info_t> curvs(cloud_->size());
        for (uint32_t i = 0; i < cloud_->size(); ++i) {
            curvs[i] = cloud_->curvature(30u, i);
        }

        for (auto && [i,j,k] : voxels) {
            int lin = k * extents_[0] * extents_[1] + j * extents_[0] + i;
            Point uvw;
            uvw.getVector3fMap() = (inv * vec4f_t(i, j, k, 1.f)).head(3);
            auto is = cloud_->knn_inclusive(1, uvw).first;
            voxel_data_[lin] = is[0];
        }

        subset_ = vw::filter(subset_, [&] (uint32_t idx) {
            vec3f_t tgt = vec3f_t(cloud_->points[idx].data_c[1], cloud_->points[idx].data_c[2], cloud_->points[idx].data_c[3]);
            return tgt.norm() > 0.7f && (curvs[idx].pc_min / curvs[idx].pc_max) < 0.2f;
        }) | ranges::to_vector;
        auto pairs = vw::cartesian_product(subset_, subset_);
        float lower_bound = diameter_ * params.min_diameter_factor;
        float upper_bound = diameter_ * params.max_diameter_factor;
        uint32_t valid_count = 0;
        for (auto && [i, j] : pairs) {
            if (i==j) continue;

            vec3f_t d1 = cloud_->points[j].getVector3fMap() - cloud_->points[i].getVector3fMap();
            float dist1 = d1.norm();
            d1 /= dist1;
            if (dist1 < lower_bound || dist1 > upper_bound) continue;
            if (1.f - fabs(d1.dot(tangent(cloud_->points[i]))) < 0.01f) continue;

            auto f = feature<Point>(cloud_->points[i], cloud_->points[j], curvs[i], curvs[j]);
            if (!f) continue;
            feat_bounds_.extend(*f);

            ++valid_count;
        }

        double valid_ratio = static_cast<double>(valid_count) / ranges::distance(pairs);
        pdebug("valid pair ratio: {}", valid_ratio);
        feat_bounds_ = valid_bounds(feat_bounds_, M_PI / 10.f, 2.0*M_PI, 0.0f, 1.f);

        std::map<int, uint32_t> hist_0, hist_1;
        for (auto && [i, j] : pairs) {
            if (i==j) continue;

            vec3f_t d1 = cloud_->points[j].getVector3fMap() - cloud_->points[i].getVector3fMap();
            float dist1 = d1.norm();
            d1 /= dist1;
            if (dist1 < lower_bound || dist1 > upper_bound) continue;
            if (1.f - fabs(d1.dot(tangent(cloud_->points[i]))) < 0.01f) continue;
            //bool debug = i == 54 && j == 313;
            auto f = feature<Point>(cloud_->points[i], cloud_->points[j], curvs[i], curvs[j]);
            if (!f) continue;
            auto df = discretize_feature<Point>(*f, feat_bounds_, params_);

            if (valid<Point>(*f, feat_bounds_)) {
                if (hist_0.find(df[0]) == hist_0.end()) {
                    hist_0[df[0]] = 0;
                }
                if (hist_1.find(df[1]) == hist_1.end()) {
                    hist_1[df[1]] = 0;
                }
                hist_0[df[0]] += 1;
                hist_1[df[1]] += 1;
                map_.insert({df, result_t{i,j}});
                used_points_.insert(i);
                used_points_.insert(j);
            }
        }

        int max_key_0 = std::max_element(hist_0.begin(), hist_0.end(), [&] (auto a, auto b) { return a.first < b.first; })->first;
        int max_key_1 = std::max_element(hist_1.begin(), hist_1.end(), [&] (auto a, auto b) { return a.first < b.first; })->first;
        std::ofstream out("/tmp/out0.dat");
        for (int i = 0; i < max_key_0; ++i) {
            if (i) out << " ";
            out << (hist_0.find(i) == hist_0.end() ? 0u : hist_0[i]);
        }
        out << "\n";
        for (int i = 0; i < max_key_1; ++i) {
            if (i) out << " ";
            out << (hist_1.find(i) == hist_1.end() ? 0u : hist_1[i]);
        }
        out << "\n";
        out.close();

        init_ = true;
    }

    std::pair<pair_iter_t, pair_iter_t>
    query(const feature_t& f, bool debug) {
        if (!init_) {
            throw std::runtime_error("Cannot query uninitialized model");
        }

        auto df = discretize_feature<Point>(f, feat_bounds_, params_);

        return map_.equal_range(df);
    }

    std::optional<uint32_t>
    voxel_query(const vec4f_t& pos, bool debug = false) const {
        vec4i_t ijk = (to_voxel_ * pos).template cast<int>();
        int i = ijk[0];
        int j = ijk[1];
        int k = ijk[2];
        if (i < 0 || j < 0 || k < 0 || i >= extents_[0] || j >= extents_[1] ||
            k >= extents_[2]) {
            return std::nullopt;
        }
        int lin = k * extents_[0] * extents_[1] + j * extents_[0] + i;
        return voxel_data_[lin];
    }

    //std::optional<float>
    //voxel_distance(const vec3f_t& local) const {
        //std::optional<vec3f_t> uvw = Proj::project(proj_, local);
        //if (!uvw) return std::nullopt;
        //std::optional<uint32_t> uvw_n = voxel_query(*uvw);
        //if (!uvw_n) return std::nullopt;
        //return Proj::intrinsic_distance(proj_, *uvw, uvw_cloud_->points[uvw_n.value()].getVector3fMap());
    //}

    float diameter() const {
        return diameter_;
    }

    const vec3i_t& extents() const {
        return extents_;
    }

    int margin() const {
        return margin_;
    }

    uint32_t point_count() const {
        return subset_.size();
    }

    uint64_t pair_count() const {
        return pair_count_;
    }

    typename cloud_t::ConstPtr cloud() const {
        return cloud_;
    }

    const feature_bounds_t&
    feature_bounds() const {
        return feat_bounds_;
    }

    typename cloud_t::ConstPtr cloud_;
    discretization_params params_;
    bool init_;

    subset_t subset_;
    sample_parameters s_params_;
    hash_map_t map_;
    float diameter_;
    vec3i_t extents_;
    int margin_;
    feature_bounds_t feat_bounds_;
    mat4f_t to_voxel_;
    uint64_t pair_count_;
    subset_t voxel_data_;
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
inline void
model<Point>::init(const sample_parameters& sample_params) {
    impl_->init(subset_t(), sample_params);
}

template <typename Point>
inline void
model<Point>::init(const subset_t& subset, const sample_parameters& sample_params) {
    impl_->init(subset, sample_params);
}

template <typename Point>
inline std::pair<typename model<Point>::pair_iter_t, typename model<Point>::pair_iter_t>
model<Point>::query(const feature_t& f, bool debug) {
    return range<pair_iter_t>(impl_->query(f, debug));
}

//template <typename Point>
//inline typename Proj::handle_t
//model<Proj,Point>::projector() {
    //return impl_->projector();
//}

//template <typename Point>
//inline typename Proj::const_handle_t
//model<Proj,Point>::projector() const {
    //return impl_->projector();
//}

//template <typename Point>
//inline typename model<Proj,Point>::cloud_t::Ptr
//model<Proj,Point>::uvw_cloud() {
    //return impl_->uvw_cloud();
//}

//template <typename Point>
//inline typename model<Proj,Point>::cloud_t::ConstPtr
//model<Proj,Point>::uvw_cloud() const {
    //return impl_->uvw_cloud();
//}

template <typename Point>
inline std::optional<uint32_t>
model<Point>::voxel_query(const vec4f_t& uvw, bool debug) const {
    return impl_->voxel_query(uvw, debug);
}

//template <typename Point>
//inline std::optional<float>
//model<Proj,Point>::voxel_distance(const vec3f_t& local) const {
    //return impl_->voxel_distance(local);
//}

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
inline const vec3i_t&
model<Point>::extents() const {
    return impl_->extents();
}

template <typename Point>
inline int
model<Point>::margin() const {
    return impl_->margin();
}

template <typename Point>
inline uint64_t
model<Point>::pair_count() const {
    return impl_->pair_count();
}

template <typename Point>
inline typename model<Point>::cloud_t::ConstPtr
model<Point>::cloud() const {
    return impl_->cloud();
}

template <typename Point>
inline const feature_bounds_t&
model<Point>::feature_bounds() const {
    return impl_->feature_bounds();
}

}  // namespace triplet_match
