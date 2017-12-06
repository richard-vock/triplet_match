namespace triplet_match {

namespace detail {

static inline uint32_t
rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

//static inline uint32_t
//murmur6(const uint32_t feat[6]) {
    //uint32_t h1 = 42;
    //const uint32_t c1 = 0xcc9e2d51;
    //const uint32_t c2 = 0x1b873593;

    //for (int i = 0; i < 6; i++) {
        //uint32_t k1 = feat[i];

        //k1 *= c1;
        //k1 = rotl32(k1, 15);
        //k1 *= c2;

        //h1 ^= k1;
        //h1 = rotl32(h1, 13);
        //h1 = h1 * 5 + 0xe6546b64;
    //}

    //h1 ^= 16;
    //h1 ^= h1 >> 16;
    //h1 *= 0x85ebca6b;
    //h1 ^= h1 >> 13;
    //h1 *= 0xc2b2ae35;
    //h1 ^= h1 >> 16;

    //return h1;
//}

constexpr uint32_t murmur_seed = 42;

template <int Rows, int Cols, int Options, int MaxRows, int MaxCols>
inline uint32_t
murmur(const Eigen::Matrix<uint32_t, Rows, Cols, Options, MaxRows, MaxCols>& key) {
    uint32_t h1 = murmur_seed;

    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    //----------
    // body

    for (uint32_t i = 0; i < key.size(); ++i) {
        uint32_t k1 = key.coeff(i);

        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }


    h1 ^= 4*key.size();

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

inline float
angle_unsigned(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    return std::acos(fabs(a.dot(b)));
}

inline uint32_t
discretize(float value, float min_value, float range_value, uint32_t steps) {
    float nval = (value - min_value) / range_value;
    if (nval < 0.f) return 0;
    if (nval >= 1.f) return steps-1;
    return static_cast<uint32_t>(nval * steps);
}

inline uint32_t
discretize(float value, float step_size) {
    return static_cast<uint32_t>(value / step_size);
}

} // detail

template <typename Point>
discrete_feature
compute_discrete(const Point& p1, const Point& p2, discretization_params params, float min_dist, float range_dist) {
    vec3f_t d = p2.getVector3fMap() - p1.getVector3fMap();
    const float f1 = d.norm();
    d /= f1;

    vec3f_t t1 = vec3f_t(p1.data_c[1], p1.data_c[2], p1.data_c[3]);
    vec3f_t t2 = vec3f_t(p2.data_c[1], p2.data_c[2], p2.data_c[3]);

    const float f2 = detail::angle(t1, t2);
    const float f3 = detail::angle(t1, d);
    const float f4 = detail::angle(t2, d);

    discrete_feature feat;
    feat[0] = detail::discretize(f1, min_dist, range_dist, params.distance_step_count);
    feat[1] = detail::discretize(f2, params.angle_step);
    feat[2] = detail::discretize(f3, params.angle_step);
    feat[3] = detail::discretize(f4, params.angle_step);
    return feat;
}

} // triplet_match

namespace std {
    template <>
    struct hash<triplet_match::discrete_feature> {
        std::size_t operator()(const triplet_match::discrete_feature& feat) const {
            return triplet_match::detail::murmur(feat);
        }
    }; // struct hash
} // namespace std
