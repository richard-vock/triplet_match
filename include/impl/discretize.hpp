namespace triplet_match {

namespace detail {

static inline uint32_t
rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

constexpr uint32_t murmur_seed = 42;

template <int Dim>
inline uint32_t
murmur(const Eigen::Matrix<uint32_t, Dim, 1>& key) {
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


    h1 ^= 4*Dim;

    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    return h1;
}

} // detail
} // triplet_match

namespace std {

template <int Dim>
inline std::size_t
hash<::Eigen::Matrix<uint32_t, Dim, 1>>::operator()(const ::Eigen::Matrix<uint32_t, Dim, 1>& feat) const {
    return ::triplet_match::detail::murmur(feat);
}

} // std

