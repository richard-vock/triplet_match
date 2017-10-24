#include <fstream>
#include <range/v3/all.hpp>

#include "debug.hpp"

namespace triplet_match {

namespace detail {

template <typename SearchT>
float
estimate_resolution(const SearchT& search) {
    std::vector<float> dists(2);
    std::vector<int> nns(2);
    double res = 0.0;
    uint32_t idx = 0;
    for (const auto& pnt : *search.getInputCloud()) {
        search.nearestKSearch(pnt, 2, nns, dists);
        res += (static_cast<double>(dists[1]) - res) / (++idx);
    }

    return static_cast<float>(res);
}

} // detail

template <typename Point>
struct model<Point>::impl {
    impl(typename cloud_t::ConstPtr cloud, projector_t projector, discretization_params params) : cloud_(cloud), projector_(std::move(projector)), params_(params), init_(false) {
    }

    ~impl() {}

    std::future<void>
    init(const subset_t& subset, const sample_parameters& params, gpu_state::sptr_t state) {
        if (subset.empty()) {
            subset_.resize(cloud_->size());
            std::iota(subset_.begin(), subset_.end(), 0u);
        } else {
            subset_ = subset;
        }

        s_params_ = params;

        state_ = state;

        return std::async(std::launch::async, [&] () {
            this->build_model_();
        });
    }

    std::pair<pair_iter_t, pair_iter_t>
    query(const Point& p1, const Point& p2, const vec2f_t& uv1, const vec2f_t& uv2) {
        if (!init_) {
            throw std::runtime_error("Cannot query uninitialized model");
        }

        float lower = s_params_.min_diameter_factor;
        float upper = s_params_.max_diameter_factor;
        discrete_feature df = compute_discrete<Point>(p1, p2, uv1, uv2, params_, lower, upper-lower);

        return map_.equal_range(df);
    }

    const projector_t& projector() const {
        return projector_;
    }

    const mat4f_t& normalization() const {
        return normalize_;
    }

    float diameter() const {
        return diameter_;
    }

    float resolution() const {
        return resolution_;
    }

    const vec2i_t& extents() const {
        return extents_;
    }

    vec3f_t centroid() const {
        return centroid_;
    }

    uint32_t point_count() const {
        return point_count_;
    }

    uint64_t pair_count() const {
        return pair_count_;
    }

    typename cloud_t::ConstPtr cloud() const {
        return cloud_;
    }

    gpu_data_t& device_data() {
        return gpu_data_;
    }

    const gpu_data_t& device_data() const {
        return gpu_data_;
    }

    const cpu_data_t& host_data() const {
        return cpu_data_;
    }

    void
    build_model_() {
        pcl::search::KdTree<Point> kdtree;
        kdtree.setInputCloud(cloud_);

        bbox3_t bbox;
        std::set<uint32_t> valid;
        centroid_ = vec3f_t::Zero();
        for (uint32_t i : subset_) {
            const Point& p = cloud_->points[i];
            if (pcl::isFinite(p)) {
                valid.insert(i);
                bbox.extend(p.getVector3fMap());
                centroid_ += (cloud_->points[i].getVector3fMap() - centroid_) / valid.size();
            }
        }
        point_count_ = valid.size();
        diameter_ = (bbox.max() - bbox.min()).norm();
        resolution_ = detail::estimate_resolution(kdtree);
        extents_ = vec2i_t(2000, 500);
        cpu_data_.resize(extents_[0] * extents_[1]);

        bbox2_t bbox_uv;
        for (const auto& pnt : *cloud_) {
            bbox_uv.extend(projector_.project(pnt.getVector3fMap()).head(2));
        }

        float lower = bbox_uv.min()[1];
        float upper = bbox_uv.max()[1];
        float range = upper - lower;
        // add margin to v
        //float margin = margin_factor * range;
        //range = upper-lower + 2.f * margin;
        // add as correction to projection matrix
        normalize_ = mat4f_t::Identity();
        float scale = 1.f / range;
        //std::cout << "normalize" << "\n";
        //std::cout << lower << "\n";
        //std::cout << upper << "\n";
        //std::cout << range << "\n";
        normalize_(1,3) = -lower;
        normalize_.row(1) *= scale;
        //unnormalize_ = normalize_.inverse();

        //std::vector<vec3f_t> debug;
        //std::map<int, std::vector<std::pair<float, vec3f_t>>> debug_data;
        Eigen::MatrixXf img = -Eigen::MatrixXf::Ones(extents_[1], extents_[0]);
        for (int j = 0; j < extents_[1]; ++j) {
            float v = lower + range * static_cast<float>(j) / extents_[1];

            int x = j * extents_[0];
            for (int i = 0; i < extents_[0]; ++i) {
                float u = static_cast<float>(i) / extents_[0];
                Point query;
                query.getVector3fMap() = projector_.unproject(vec2f_t(u, v));

                std::vector<int> nns(1);
                std::vector<float> dists(1);
                kdtree.nearestKSearch(query, 1, nns, dists);
                //distance_data_[x + y + k] =
                    //max_integer_dist * (sqrtf(dists[0]) / max_dist);
                vec3f_t neigh = cloud_->points[nns[0]].getVector3fMap();
                cpu_data_[x + i][0] = neigh[0];
                cpu_data_[x + i][1] = neigh[1];
                cpu_data_[x + i][2] = neigh[2];
                cpu_data_[x + i][3] = 1.f;

                if (sqrtf(dists[0] < 0.0005f)) {
                    //if (nns[0] == 100) {
                        //fmt::print("model point, uv: ({}, {}), idx: {}, dist: {}, px: ({}, {})\n", u, v, x + j, dists[0], i, j);
                    //}
                    img(j, i) = sqrtf(dists[0]);
                    //if (debug_data.find(nns[0]) == debug_data.end()) {
                        //debug_data[nns[0]] = std::vector<std::pair<float, vec3f_t>>();
                    //}
                    //debug_data[nns[0]].push_back(std::make_pair(sqrtf(dists[0]), vec3f_t(u, v, 0.f)));
                }
            }
        }

        //for (uint32_t i = 0; i < cloud_->size(); ++i) {
            //if (i%30) continue;
            //auto it = debug_data.find(i);
            //if (it == debug_data.end()) continue;

            //std::vector<std::pair<float, vec3f_t>> copy(it->second.begin(), it->second.end());
            //std::sort(copy.begin(), copy.end(), [&] (auto a, auto b) { return a.first < b.first; });
            //debug.push_back(copy.front().second);
            //debug.push_back(cloud_->points[i].getVector3fMap());
        //}

        to_grayscale_image("/tmp/model.pgm", img);

        lower = /* diameter_ */ s_params_.min_diameter_factor;
        upper = /* diameter_ */ s_params_.max_diameter_factor;
        //float range = upper-lower;
        pair_count_ = 0;
        for (uint32_t i : valid) {
            const Point& p1 = cloud_->points[i];
            vec2f_t uv1 = projector_.project(p1.getVector3fMap()).head(2);

            for (uint32_t j : valid) {
                if (i == j) {
                    continue;
                }
                const Point& p2 = cloud_->points[j];
                vec2f_t uv2 = projector_.project(p2.getVector3fMap()).head(2);

                vec2f_t d = uv2 - uv1;
                float dist = d.norm();
                if (dist < lower || dist > upper) {
                    continue;
                }

                ++pair_count_;

                //used_points_.insert(i);
                //used_points_.insert(j);
                discrete_feature df = compute_discrete<Point>(p1, p2, uv1, uv2, params_, lower, upper-lower);

                map_.insert({df, pair_t{i, j}});
            }
        }


        gpu_data_ =
            gpu::vector<gpu::float4_>(cpu_data_.size(), state_->context);
        gpu::copy(cpu_data_.begin(), cpu_data_.end(), gpu_data_.begin(), state_->queue);

        init_ = true;
    }

    typename cloud_t::ConstPtr cloud_;
    projector_t projector_;
    discretization_params params_;
    bool init_;

    gpu_state::sptr_t state_;
    subset_t subset_;
    sample_parameters s_params_;
    hash_map_t map_;
    float diameter_;
    float resolution_;
    vec3f_t centroid_;
    uint32_t point_count_;
    vec2i_t extents_;
    uint64_t pair_count_;
    cpu_data_t cpu_data_;
    gpu_data_t gpu_data_;
    mat4f_t normalize_;
    //mat4f_t unnormalize_;
};


template <typename Point>
inline
model<Point>::model(typename cloud_t::ConstPtr cloud, projector_t projector, discretization_params params) {
    impl_ = std::make_unique<impl>(cloud, std::move(projector), params);
}

template <typename Point>
inline
model<Point>::~model() {
}

template <typename Point>
inline std::future<void>
model<Point>::init(const sample_parameters& sample_params, gpu_state::sptr_t state) {
    return impl_->init(subset_t(), sample_params, state);
}

template <typename Point>
inline std::future<void>
model<Point>::init(const subset_t& subset, const sample_parameters& sample_params, gpu_state::sptr_t state) {
    return impl_->init(subset, sample_params, state);
}

template <typename Point>
inline std::pair<typename model<Point>::pair_iter_t, typename model<Point>::pair_iter_t>
model<Point>::query(const Point& p1, const Point& p2, const vec2f_t& uv1, const vec2f_t& uv2) {
    return range<pair_iter_t>(impl_->query(p1, p2, uv1, uv2));
}

template <typename Point>
inline const typename model<Point>::projector_t&
model<Point>::projector() const {
    return impl_->projector();
}

template <typename Point>
inline const mat4f_t&
model<Point>::normalization() const {
    return impl_->normalization();
}

template <typename Point>
inline float
model<Point>::diameter() const {
    return impl_->diameter();
}

template <typename Point>
inline float
model<Point>::resolution() const {
    return impl_->resolution();
}

template <typename Point>
inline vec3f_t
model<Point>::centroid() const {
    return impl_->centroid();
}

template <typename Point>
inline uint32_t
model<Point>::point_count() const {
    return impl_->point_count();
}

template <typename Point>
inline const vec2i_t&
model<Point>::extents() const {
    return impl_->extents();
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
inline typename model<Point>::gpu_data_t&
model<Point>::device_data() {
    return impl_->device_data();
}

template <typename Point>
inline const typename model<Point>::gpu_data_t&
model<Point>::device_data() const {
    return impl_->device_data();
}

template <typename Point>
inline const typename model<Point>::cpu_data_t&
model<Point>::host_data() const {
    return impl_->host_data();
}

//template <typename Point>
//void
//model<Point>::write_octave_density_maps(const std::string& folder, const std::string& data_file_prefix, const std::string& script_file) const {
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
