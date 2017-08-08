#include <random>
#include <chrono>
namespace triplet_match {

namespace detail {

constexpr float radius_percentage = 1.f;
constexpr bool deterministic = true;

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
struct scene<Point>::impl {
    impl(typename cloud_t::ConstPtr cloud) : cloud_(cloud) {}

    ~impl() {}

    template <typename PointModel, typename Rejector>
    mat4f_t
    find(model<PointModel>& m, Rejector&& reject, subset_t subset) {
        pcl::IndicesPtr indices;
        if (subset.empty()) {
            indices = pcl::IndicesPtr(new std::vector<int>(cloud_->size()));
            std::iota(indices->begin(), indices->end(), 0);
        } else {
            indices = pcl::IndicesPtr(
                new std::vector<int>(subset.begin(), subset.end()));
        }
        kdtree_.setInputCloud(cloud_, indices);

        std::mt19937 rng;
        uint32_t seed = 13;
        if (!detail::deterministic) {
            auto now = std::chrono::system_clock::now();
            seed = now.time_since_epoch().count();
        }
        rng.seed(seed);

        for (int i : *indices) {
            const Point& p1 = cloud_->points[i];

            std::vector<int> nn;
            std::vector<float> dist;
            kdtree_.radiusSearch(p1, detail::radius_percentage * m.diameter(), nn,
                                 dist);

            for (int j : nn) {
                if (j <= i) continue;

                const Point& p2 = cloud_->points[j];

                for (int k : nn) {
                    if (k <= j) continue;

                    const Point& p3 = cloud_->points[k];
                    for (auto&& [feat, triplet] : m.query(p1, p2, p3)) {
                        auto&& [m_i, m_j, m_k] = triplet;

                        mat4f_t transform = detail::base_transformation<PointModel, Point>(
                            m.cloud(), cloud_, m_i, m_j, m_k,
                            static_cast<uint32_t>(i),
                            static_cast<uint32_t>(j),
                            static_cast<uint32_t>(k));
                        if (!std::invoke(reject, transform)) {
                            return transform;
                        }
                    }
                }
            }
        }
    }

    typename cloud_t::ConstPtr cloud_;
    pcl::search::KdTree<Point> kdtree_;
};

template <typename Point>
inline
scene<Point>::scene(typename cloud_t::ConstPtr cloud) : impl_(new impl(cloud)) {
}

template <typename Point>
inline
scene<Point>::~scene() {
}

template <typename Point>
template <typename PointModel, typename Rejector>
inline mat4f_t
scene<Point>::find(model<PointModel>& m, Rejector&& reject) {
    return impl_->find(m, std::forward<Rejector>(reject), subset_t());
}

template <typename Point>
template <typename PointModel, typename Rejector>
inline mat4f_t
scene<Point>::find(model<PointModel>& m, Rejector&& reject, const subset_t& subset) {
    return impl_->find(m, std::forward<Rejector>(reject), subset);
}

}  // namespace triplet_match
