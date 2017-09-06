#include <random>
#include <chrono>
#include "timer.hpp"

namespace triplet_match {

namespace detail {

//constexpr uint32_t random_kernel_size = 100;
constexpr bool early_out = true;
constexpr bool deterministic = true;
constexpr double match_probability = 0.99;
constexpr float max_dist_factor = 1.0;

uint64_t
estimate_sample_count(uint32_t model_size, uint32_t scene_size, double neighborhood_size) {
    double prob = std::pow(static_cast<double>(model_size), 3.0) / (static_cast<double>(scene_size) * neighborhood_size * neighborhood_size);
    //std::cout << "probability: " << prob << "\n";
    uint64_t cand_bound = static_cast<uint64_t>(-std::log(1.0 - detail::match_probability) / prob);
    std::cout << "estimated sample count: " << cand_bound << "\n";
    return cand_bound;
}

template <typename Point>
inline mat3f_t
make_base(typename pcl::PointCloud<Point>::ConstPtr cloud, int i, int j, int k) {
    vec3f_t p0 = cloud->points[i].getVector3fMap();
    vec3f_t p1 = cloud->points[j].getVector3fMap();
    vec3f_t p2 = cloud->points[k].getVector3fMap();
    vec3f_t e0 = (p1 - p0).normalized();
    vec3f_t e1 = p2 - p0;
    vec3f_t e2 = e0.cross(e1).normalized();
    e1 = e2.cross(e0).normalized();
    mat3f_t base;
    base << e0, e1, e2;
    return base;
}

inline quat_t
base_rotation(const mat3f_t& b0, const mat3f_t& b1) {
    quat_t q1, q2;
    vec3f_t b0x = b0.col(0);
    vec3f_t b0y = b0.col(1);
    vec3f_t b1x = b1.col(0);
    vec3f_t b1y = b1.col(1);
    q1.setFromTwoVectors(b0x, b1x);
    vec3f_t b0y_trans = q1._transformVector(b0y);
    q2.setFromTwoVectors(b0y_trans, b1y);
    return q2 * q1;
}

template <typename Point0, typename Point1>
inline mat4f_t
base_transformation(typename pcl::PointCloud<Point0>::ConstPtr c0, typename pcl::PointCloud<Point1>::ConstPtr c1, uint32_t i0, uint32_t j0,
                     uint32_t k0, uint32_t i1, uint32_t j1, uint32_t k1) {
    auto b0 = make_base<Point0>(c0, i0, j0, k0);
    auto b1 = make_base<Point1>(c1, i1, j1, k1);
    mat4f_t r = mat4f_t::Identity();
    mat4f_t t = mat4f_t::Identity();
    r.topLeftCorner<3,3>() = base_rotation(b0, b1).toRotationMatrix();
    t.block<3,1>(0,3) = -c0->points[i0].getVector3fMap();
    r = r * t;
    t.block<3,1>(0,3) = c1->points[i1].getVector3fMap();
    r = t * r;

    return r;
}

}  // namespace detail

template <typename Point>
inline
scene<Point>::impl::impl(typename cloud_t::ConstPtr cloud) : cloud_(cloud) {}

template <typename Point>
inline
scene<Point>::impl::~impl() {}

template <typename Point>
template <typename PointModel>
inline std::pair<mat4f_t, uint32_t>
scene<Point>::impl::find(model<PointModel>& m, std::function<uint32_t (const mat4f_t&)> score_func, std::function<bool (uint32_t)> early_out_func, const sample_parameters& params, subset_t subset) {
    pcl::IndicesPtr indices;
    if (subset.empty()) {
        indices = pcl::IndicesPtr(new std::vector<int>(cloud_->size()));
        std::iota(indices->begin(), indices->end(), 0);
    } else {
        indices = pcl::IndicesPtr(
            new std::vector<int>(subset.begin(), subset.end()));
    }
    kdtree_.setInputCloud(cloud_, indices);
    float lower = m.diameter() * params.min_diameter_factor;
    float upper = m.diameter() * params.max_diameter_factor;
    //float lower_ratio = params.min_triplet_ratio;
    //float upper_ratio = params.max_triplet_ratio;
    
    std::mt19937 rng;
    uint32_t seed = 13;
    if (!detail::deterministic) {
        auto now = std::chrono::system_clock::now();
        seed = now.time_since_epoch().count();
    }
    rng.seed(seed);
    std::shuffle(indices->begin(), indices->end(), rng);

    mat4f_t best_transform = mat4f_t::Identity();
    uint32_t best_score = 0;
    uint64_t valid_sample_count = 0, sample_count = 0;

    uint32_t n_model = m.cloud()->size();
    uint32_t n_scene = indices->size();
    uint64_t outer_bound = static_cast<uint64_t>(std::log(1.0 - detail::match_probability) / std::log(1.0 - static_cast<double>(n_model) / n_scene));

    for (uint64_t outer = 0; outer < std::min(indices->size(), outer_bound); ++outer) {
        int i = (*indices)[outer];
        const Point& p1 = cloud_->points[i];

        std::vector<int> nn;
        std::vector<float> dist;
        kdtree_.radiusSearch(p1, upper, nn,
                             dist);
        if (nn.empty()) continue;
        std::shuffle(nn.begin(), nn.end(), rng);

        double prob = static_cast<double>(n_model) / nn.size();
        uint64_t inner_bound = static_cast<uint64_t>(-std::log(1.0 - detail::match_probability) / prob);
        inner_bound = std::min(inner_bound, nn.size());

        for (uint64_t inner0 = 0; inner0 < inner_bound; ++inner0) {
            int j = nn[inner0];
            if (j == i) continue;

            const Point& p2 = cloud_->points[j];

            vec3f_t d1 = p2.getVector3fMap() - p1.getVector3fMap();
            float dist1 = d1.norm();
            if (dist1 < lower) {
                continue;
            }

            for (uint64_t inner1 = 0; inner1 < inner_bound; ++inner1) {
                int k = nn[inner1];
                if (k == j || k == i) continue;

                const Point& p3 = cloud_->points[k];
                vec3f_t d2 = (p3.getVector3fMap() - p2.getVector3fMap());
                vec3f_t d3 = (p3.getVector3fMap() - p1.getVector3fMap());
                float dist2 = d2.norm();
                float dist3 = d3.norm();
                if (dist2 < lower || dist3 < lower) {
                    continue;
                }
                //float rat0 = dist2 / dist1;
                //float rat1 = dist3 / dist1;
                //if (rat0 < lower || rat0 > upper || rat1 < lower || rat1 > upper) {
                    //continue;
                //}

                auto && [q_first, q_last] = m.query(p1, p2, p3);
                if (q_first != q_last) {
                    ++valid_sample_count;
                }

                ++sample_count;
                for (auto it = q_first; it != q_last; ++it) {
                    auto&& [m_i, m_j, m_k] = it->second;

                    mat4f_t transform = detail::base_transformation<PointModel, Point>(
                        cloud_, m.cloud(),
                        static_cast<uint32_t>(i),
                        static_cast<uint32_t>(j),
                        static_cast<uint32_t>(k), m_i, m_j, m_k);
                    uint32_t score = std::invoke(score_func, transform);
                    if (score > best_score) {
                        best_transform = transform;
                        best_score = score;

                        if (detail::early_out && early_out_func(best_score)) {
                            return {transform, best_score};
                        }
                    }
                }
            }
        }
    }

    return {best_transform, best_score};
}

template <typename Point>
inline
scene<Point>::scene(typename cloud_t::ConstPtr cloud) : impl_(new impl(cloud)) {
}

template <typename Point>
inline
scene<Point>::~scene() {
}

template <typename Point>
inline typename scene<Point>::cloud_t::ConstPtr
scene<Point>::cloud() const {
    return impl_->cloud();
}

}  // namespace triplet_match
