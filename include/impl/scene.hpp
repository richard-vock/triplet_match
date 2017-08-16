#include <random>
#include <chrono>
namespace triplet_match {

namespace detail {

constexpr bool deterministic = true;
constexpr double match_probability = 0.99;

uint64_t
estimate_sample_count(uint32_t model_size, uint32_t scene_size, double neighborhood_size) {
    double prob = std::pow(static_cast<double>(model_size), 3.0) / (static_cast<double>(scene_size) * 0.25 * neighborhood_size * neighborhood_size);
    std::cout << "probability: " << prob << "\n";
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
inline mat4f_t
scene<Point>::impl::find(model<PointModel>& m, std::function<float (const mat4f_t&)> score_func, const sample_parameters& params, subset_t subset) {
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
    lower = lower * lower;

    std::mt19937 rng;
    uint32_t seed = 13;
    if (!detail::deterministic) {
        auto now = std::chrono::system_clock::now();
        seed = now.time_since_epoch().count();
    }
    rng.seed(seed);
    std::shuffle(indices->begin(), indices->end(), rng);

    mat4f_t best_transform = mat4f_t::Identity();
    float best_score = 0.f;
    uint64_t valid_sample_count = 0, sample_count = 0, last_important_sample = 0, last_important_valid_sample = 0;

    uint32_t n_model = m.cloud()->size();
    uint32_t n_scene = indices->size();
    //double neighborhood_size = static_cast<double>(n_scene);//3529;
    std::cout << "model size: " << n_model << "\n";
    std::cout << "scene size: " << n_scene << "\n";
    uint64_t cand_bound = 0;
    double neighborhood_size = 0.0;
    uint32_t first_points = 0;

    bool stop = false;
    for (int i : *indices) {
        if (stop) {
            break;
        }
        const Point& p1 = cloud_->points[i];

        std::vector<int> nn;
        std::vector<float> dist;
        kdtree_.radiusSearch(p1, upper, nn,
                             dist);

        // online update for average neighborhood size
        neighborhood_size += (static_cast<double>(nn.size()) - neighborhood_size) / (++first_points);
        cand_bound = detail::estimate_sample_count(n_model, n_scene, neighborhood_size);

        for (int j : nn) {
            if (stop) {
                break;
            }
            if (j <= i) continue;

            const Point& p2 = cloud_->points[j];

            vec3f_t d1 = p2.getVector3fMap() - p1.getVector3fMap();
            if (d1.squaredNorm() < lower) {
                continue;
            }

            for (int k : nn) {
                if (stop) {
                    break;
                }
                if (k <= j) continue;

                const Point& p3 = cloud_->points[k];
                vec3f_t d2 = (p3.getVector3fMap() - p1.getVector3fMap()).normalized();
                if (1.f - fabs(d2.dot(d1.normalized())) < params.min_orthogonality) {
                    continue;
                }

                auto && [q_first, q_last] = m.query(p1, p2, p3);
                if (q_first != q_last) {
                    ++valid_sample_count;
                }
                stop = (++sample_count) > cand_bound;
                for (auto it = q_first; it != q_last; ++it) {
                    auto&& [m_i, m_j, m_k] = it->second;

                    mat4f_t transform = detail::base_transformation<PointModel, Point>(
                        cloud_, m.cloud(),
                        static_cast<uint32_t>(i),
                        static_cast<uint32_t>(j),
                        static_cast<uint32_t>(k), m_i, m_j, m_k);
                    float score = std::invoke(score_func, transform);
                    if (score > best_score) {
                        last_important_sample = sample_count;
                        last_important_valid_sample = valid_sample_count;
                        best_transform = transform;
                        best_score = score;
                    }
                }
            }
        }
    }

    std::cout << "valid sample count: " << valid_sample_count << "\n";
    std::cout << "tried sample count: " << sample_count << "\n";
    std::cout << "important valid: " << last_important_valid_sample << "\n";
    std::cout << "important tried: " << last_important_sample << "\n";
    std::cout << "validity ratio: " << (static_cast<double>(valid_sample_count) / sample_count) << "\n";
    return best_transform.inverse();
}

template <typename Point>
inline
scene<Point>::scene(typename cloud_t::ConstPtr cloud) : impl_(new impl(cloud)) {
}

template <typename Point>
inline
scene<Point>::~scene() {
}

}  // namespace triplet_match
