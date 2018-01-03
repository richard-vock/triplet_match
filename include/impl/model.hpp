#include <fstream>
#include <range/v3/all.hpp>

//#include "debug.hpp"

namespace triplet_match {

template <typename Proj, typename Point>
struct model<Proj,Point>::impl {
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

        proj_ = Proj::init_from_model(cloud_, subset);

        s_params_ = params;

        bbox3_t bbox;
        for (uint32_t i : subset_) {
            const Point& p = cloud_->points[i];
            bbox.extend(p.getVector3fMap());
        }
        diameter_ = (bbox.max() - bbox.min()).norm();

        uvw_cloud_ = cloud_t::empty();
        for (auto idx : subset_) {
            Point uvw_pnt;
            uvw_pnt.getVector3fMap() = traits_t::project(proj_, cloud_->points[idx].getVector3fMap());
            uvw_cloud_->push_back(uvw_pnt);
            uvw_bounds_.extend(uvw_cloud_->points[idx].getVector3fMap());
        }
        uvw_res_ = uvw_cloud_->resolution();
        vec3f_t lower = uvw_bounds_.min();
        vec3f_t upper = uvw_bounds_.max();
        vec3f_t range = upper-lower;

        vec3f_t ext = (uvw_bounds_.diagonal() / uvw_res_)
            .cwiseMax(vec3f_t::Constant(1.f));

        margin_ = 5;

        extents_ = (ext + vec3f_t::Constant(2.f*margin_)).template cast<int>();

        vec3f_t scale(
            range[0] < Eigen::NumTraits<float>::dummy_precision() ? 1.f : ext[0] / range[0],
            range[1] < Eigen::NumTraits<float>::dummy_precision() ? 1.f : ext[1] / range[1],
            range[2] < Eigen::NumTraits<float>::dummy_precision() ? 1.f : ext[2] / range[2]);
        uvw_to_voxel_ = mat4f_t::Identity();
        uvw_to_voxel_.block<3, 3>(0, 0) = scale.asDiagonal();
        uvw_to_voxel_.block<3, 1>(0, 3) = uvw_to_voxel_.template block<3, 3>(0, 0) * (-uvw_bounds_.min())
                                + vec3f_t::Constant(static_cast<float>(margin_));
        // subvoxel-shift
        uvw_to_voxel_.template block<3, 1>(0, 3) -= vec3f_t::Constant(0.5f);

        mat4f_t inv = uvw_to_voxel_.inverse();

        uint32_t voxel_count = extents_[0] * extents_[1] * extents_[2];
        voxel_data_.resize(voxel_count);

        auto voxels = vw::cartesian_product(
            vw::ints(0, extents_[0]),
            vw::ints(0, extents_[1]),
            vw::ints(0, extents_[2])
        );

        for (auto && [i,j,k] : voxels) {
            int lin = k * extents_[0] * extents_[1] + j * extents_[0] + i;
            Point uvw;
            uvw.getVector3fMap() = (inv * vec4f_t(i, j, k, 1.f)).head(3);
            auto is = uvw_cloud_->knn_inclusive(1, uvw).first;
            voxel_data_[lin] = is[0];
        }

        subset_ = vw::filter(subset_, [&] (uint32_t idx) {
            vec3f_t tgt = vec3f_t(cloud_->points[idx].data_c[1], cloud_->points[idx].data_c[2], cloud_->points[idx].data_c[3]);
            return tgt.norm() > 0.7f;
        }) | ranges::to_vector;
        for (auto && [i, j] : vw::cartesian_product(subset_, subset_)) {
            if (i==j) continue;

            auto f = traits_t::feature(proj_, cloud_->points[i], cloud_->points[j]);
            feat_bounds_.extend(f);
        }

        std::map<int, uint32_t> hist_0, hist_1;
        for (auto && [i, j] : vw::cartesian_product(subset_, subset_)) {
            if (i==j) continue;

            //bool debug = i == 54 && j == 313;
            auto f = traits_t::feature(proj_, cloud_->points[i], cloud_->points[j]);
            auto df = traits_t::discretize_feature(proj_, f, feat_bounds_, params_);

            if (traits_t::valid(proj_, f, feat_bounds_, M_PI / 10.f, 2.0*M_PI, 0.4f, 1.f)) {
                //vec3f_t ti(cloud_->points[i].data_c[1], cloud_->points[i].data_c[2], cloud_->points[i].data_c[3]);
                //vec3f_t tj(cloud_->points[j].data_c[1], cloud_->points[j].data_c[2], cloud_->points[j].data_c[3]);
                //pdebug("model ti: {}, ||t_i|| = {}", ti.transpose(), ti.norm());
                //pdebug("model tj: {}, ||t_j|| = {}", tj.transpose(), tj.norm());
                //pdebug("model feat 1: {}", f[1]);
                //if (debug) {
                    //pdebug("model feat: {}", f.transpose());
                    //pdebug("model dfeat: {}", df.transpose());
                //}
                map_.insert({df, pair_t{i,j}});
                used_points_.insert(i);
                used_points_.insert(j);
            }
        }

        init_ = true;
    }

    std::pair<pair_iter_t, pair_iter_t>
    query(const typename traits_t::feature_t& f, bool debug) {
        if (!init_) {
            throw std::runtime_error("Cannot query uninitialized model");
        }

        auto df = traits_t::discretize_feature(proj_, f, feat_bounds_, params_);
        if (debug) {
            pdebug("scene dfeat: {}", df.transpose());
        }

        return map_.equal_range(df);
    }

    typename Proj::handle_t projector() {
        return proj_;
    }

    typename Proj::const_handle_t projector() const {
        return proj_;
    }

    typename cloud_t::Ptr uvw_cloud() {
        return uvw_cloud_;
    }

    typename cloud_t::ConstPtr uvw_cloud() const {
        return uvw_cloud_;
    }

    std::optional<uint32_t>
    voxel_query(const vec3f_t& uvw, bool debug = false) const {
        vec4i_t ijk = (uvw_to_voxel_ * uvw.homogeneous()).template cast<int>();
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

    std::optional<float>
    voxel_distance(const vec3f_t& local) const {
        vec3f_t uvw = Proj::project(proj_, local);
        std::optional<uint32_t> uvw_n = voxel_query(uvw);
        if (!uvw_n) return std::nullopt;
        return Proj::intrinsic_distance(proj_, uvw, uvw_cloud_->points[uvw_n.value()].getVector3fMap());
    }

    float diameter() const {
        return diameter_;
    }

    const vec3i_t& extents() const {
        return extents_;
    }

    int margin() const {
        return margin_;
    }

    float uvw_resolution() const {
        return uvw_res_;
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

    const typename Proj::feature_bounds_t&
    feature_bounds() const {
        return feat_bounds_;
    }

    //vec3i_t from_linear(const vec3i_t& ijk) {
        //return ijk[2] * extents_[0] * extents_[1] + ijk[1] * extents_[0] + ijk[0];
    //}

    //typename cloud_t::Ptr
    //instantiate(const mat4f_t& rigid, const mat4f_t& uvw_transform) const {
        //mat4f_t inv_norm = normalize_.inverse();

        //typename cloud_t::Ptr result(new cloud_t());
        //result->resize(cloud_->size());
        //for (uint32_t idx = 0; idx < cloud_->size(); ++idx) {
            //vec4f_t p = cloud_->points[idx].getVector3fMap().homogeneous();

            //// project into normalized uvw space
            //p.head(3) = projector_.project(p.head(3));
            //p = normalize_ * p;

            //// distort
            //p = uvw_transform * p;

            //vec3f_t unnorm = (inv_norm * p).head(3);

            //// unproject
            //p.head(3) = projector_.unproject(unnorm);

            //// rigid transform into scene
            //p = rigid * p;

            //result->points[idx].getVector3fMap() = p.head(3);
        //}

        //return result;
    //}

    typename cloud_t::ConstPtr cloud_;
    typename cloud_t::Ptr uvw_cloud_;
    typename traits_t::handle_t proj_;
    discretization_params params_;
    bool init_;

    subset_t subset_;
    sample_parameters s_params_;
    hash_map_t map_;
    float diameter_;
    vec3i_t extents_;
    int margin_;
    bbox3_t uvw_bounds_;
    typename Proj::feature_bounds_t feat_bounds_;
    mat4f_t uvw_to_voxel_;
    float uvw_res_;
    uint64_t pair_count_;
    subset_t voxel_data_;
    std::set<uint32_t> used_points_;
};


template <typename Proj, typename Point>
inline
model<Proj,Point>::model(typename cloud_t::ConstPtr cloud, discretization_params params) {
    impl_ = std::make_unique<impl>(cloud, params);
}

template <typename Proj, typename Point>
inline
model<Proj,Point>::~model() {
}

template <typename Proj, typename Point>
inline void
model<Proj,Point>::init(const sample_parameters& sample_params) {
    impl_->init(subset_t(), sample_params);
}

template <typename Proj, typename Point>
inline void
model<Proj,Point>::init(const subset_t& subset, const sample_parameters& sample_params) {
    impl_->init(subset, sample_params);
}

template <typename Proj, typename Point>
inline std::pair<typename model<Proj,Point>::pair_iter_t, typename model<Proj,Point>::pair_iter_t>
model<Proj,Point>::query(const typename traits_t::feature_t& f, bool debug) {
    return range<pair_iter_t>(impl_->query(f, debug));
}

//template <typename Proj, typename Point>
//inline typename model<Proj,Point>::cloud_t::Ptr
//model<Proj,Point>::instantiate(const mat4f_t& rigid, const mat4f_t& uvw_transform) const {
    //return impl_->instantiate(rigid, uvw_transform);
//}

template <typename Proj, typename Point>
inline typename Proj::handle_t
model<Proj,Point>::projector() {
    return impl_->projector();
}

template <typename Proj, typename Point>
inline typename Proj::const_handle_t
model<Proj,Point>::projector() const {
    return impl_->projector();
}

template <typename Proj, typename Point>
inline typename model<Proj,Point>::cloud_t::Ptr
model<Proj,Point>::uvw_cloud() {
    return impl_->uvw_cloud();
}

template <typename Proj, typename Point>
inline typename model<Proj,Point>::cloud_t::ConstPtr
model<Proj,Point>::uvw_cloud() const {
    return impl_->uvw_cloud();
}

//template <typename Proj, typename Point>
//inline const mat4f_t&
//model<Proj,Point>::uvw_to_voxel() const {
    //return impl_->uvw_to_voxel();
//}

template <typename Proj, typename Point>
inline std::optional<uint32_t>
model<Proj,Point>::voxel_query(const vec3f_t& uvw, bool debug) const {
    return impl_->voxel_query(uvw, debug);
}

template <typename Proj, typename Point>
inline std::optional<float>
model<Proj,Point>::voxel_distance(const vec3f_t& local) const {
    return impl_->voxel_distance(local);
}

template <typename Proj, typename Point>
inline float
model<Proj,Point>::diameter() const {
    return impl_->diameter();
}

template <typename Proj, typename Point>
inline uint32_t
model<Proj,Point>::point_count() const {
    return impl_->point_count();
}

template <typename Proj, typename Point>
inline const vec3i_t&
model<Proj,Point>::extents() const {
    return impl_->extents();
}

template <typename Proj, typename Point>
inline int
model<Proj,Point>::margin() const {
    return impl_->margin();
}

template <typename Proj, typename Point>
inline float
model<Proj,Point>::uvw_resolution() const {
    return impl_->uvw_resolution();
}

template <typename Proj, typename Point>
inline uint64_t
model<Proj,Point>::pair_count() const {
    return impl_->pair_count();
}

template <typename Proj, typename Point>
inline typename model<Proj,Point>::cloud_t::ConstPtr
model<Proj,Point>::cloud() const {
    return impl_->cloud();
}

template <typename Proj, typename Point>
inline const typename Proj::feature_bounds_t&
model<Proj,Point>::feature_bounds() const {
    return impl_->feature_bounds();
}

//template <typename Proj, typename Point>
//inline typename model<Proj,Point>::gpu_data_t&
//model<Proj,Point>::device_data() {
    //return impl_->device_data();
//}

//template <typename Proj, typename Point>
//inline const typename model<Proj,Point>::gpu_data_t&
//model<Proj,Point>::device_data() const {
    //return impl_->device_data();
//}

//template <typename Proj, typename Point>
//inline const typename model<Proj,Point>::cpu_data_t&
//model<Proj,Point>::host_data() const {
    //return impl_->host_data();
//}

//template <typename Proj, typename Point>
//void
//model<Proj,Point>::write_octave_density_maps(const std::string& folder, const std::string& data_file_prefix, const std::string& script_file) const {
//    const discretization_params& ps = impl_->params_;
//    const hash_map_t& map = impl_->map_;
//
//    const float max_angle = static_cast<float>(M_PI);
//    const uint32_t angle_count = static_cast<uint32_t>(max_angle / ps.angle_step) + 1;
//    const uint32_t dist_count = static_cast<uint32_t>(ps.distance_step_count);
//
//    std::vector<std::vector<uint32_t>> hists(5);
//    hists[0] = std::vector<uint32_t>(dist_count, 0);
//    hists[1] = std::vector<uint32_t>(dist_count, 0);
//    hists[2] = std::vector<uint32_t>(angle_count, 0);
//    hists[3] = std::vector<uint32_t>(angle_count, 0);
//    hists[4] = std::vector<uint32_t>(angle_count, 0);
//    auto missing_dims = ranges::view::cartesian_product(
//        ranges::view::ints(0u, dist_count),
//        ranges::view::ints(0u, dist_count),
//        ranges::view::ints(0u, angle_count),
//        ranges::view::ints(0u, angle_count),
//        ranges::view::ints(0u, angle_count)
//    );
//    for (auto && [a,b,c,d,e] : missing_dims) {
//        discrete_feature feat;
//        feat[0] = a;
//        feat[1] = b;
//        feat[2] = c;
//        feat[3] = d;
//        feat[4] = e;
//        auto && [fst,lst] = map.equal_range(feat);
//        uint32_t count = std::distance(fst, lst);
//        hists[0][a] += count;
//        hists[1][b] += count;
//        hists[2][c] += count;
//        hists[3][d] += count;
//        hists[4][e] += count;
//    }
//
//    auto name = [&] (uint32_t i) { return data_file_prefix + "_" + std::to_string(i); };
//
//    std::ofstream out;
//    for (uint32_t i = 0; i < 5; ++i) {
//        out.open(folder + "/" + name(i) + ".dat");
//        for (const auto& count : hists[i]) {
//            out << count << "\n";
//        }
//        out.close();
//    }
//
//    out.open(folder + "/" + script_file);
//    for (uint32_t i = 0; i < 5; ++i) {
//        out << "load " << name(i) << ".dat" << "\n";
//        out << "figure(" << (i+1) << ")" << "\n";
//        out << "plot(0:numel(" << name(i) << ")-1, " << name(i) << ", '-')" << "\n";
//    }
//    out.close();
//}

}  // namespace triplet_match
