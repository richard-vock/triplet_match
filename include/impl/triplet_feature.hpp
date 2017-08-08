namespace triplet_match {

namespace detail {

static inline uint32_t
rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

static inline uint32_t
murmur6(const uint32_t feat[6]) {
    uint32_t h1 = 42;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    for (int i = 0; i < 6; i++) {
        uint32_t k1 = feat[i];

        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    h1 ^= 16;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    return h1;
}

inline float
angle(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    const auto c = a.cross(b);
    return atan2f(c.norm(), a.dot(b));
}

} // detail

template <typename Point>
discrete_feature
compute_discrete(const Point& p1, const Point& p2, const Point& p3, discretization_params params) {
    vec3f_t d1 = p2.getVector3fMap() - p1.getVector3fMap();
    vec3f_t d2 = p3.getVector3fMap() - p2.getVector3fMap();
    vec3f_t d3 = p3.getVector3fMap() - p1.getVector3fMap();
    const float f1 = d1.norm();
    const float f2 = d3.norm();

    d1 /= f1;
    d2.normalize();
    d3 /= f2;

    const float f3 = detail::angle(d1, d3);

    vec3f_t t1 = p1.getNormalVector3fMap().normalized();
    vec3f_t t2 = p2.getNormalVector3fMap().normalized();
    vec3f_t t3 = p3.getNormalVector3fMap().normalized();
    const float f4 = detail::angle(t1, d1);
    const float f5 = detail::angle(t2, d2);
    const float f6 = detail::angle(t3, -d3);

    discrete_feature feat;
    feat[0] = static_cast<uint32_t>(f1 / params.distance_step);
    feat[1] = static_cast<uint32_t>(f2 / params.distance_step);
    feat[2] = static_cast<uint32_t>(f3 / params.angle_step);
    feat[3] = static_cast<uint32_t>(f4 / params.angle_step);
    feat[4] = static_cast<uint32_t>(f5 / params.angle_step);
    feat[5] = static_cast<uint32_t>(f6 / params.angle_step);
    return feat;
}

} // triplet_match

namespace std {
    template <>
    struct hash<triplet_match::discrete_feature> {
        std::size_t operator()(const triplet_match::discrete_feature& feat) const {
            return triplet_match::detail::murmur6(feat.data());
        }
    }; // struct hash
} // namespace std
