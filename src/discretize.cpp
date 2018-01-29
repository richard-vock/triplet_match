#include <discretize>
#include <impl/discretize.hpp>

namespace triplet_match {
namespace detail {

template uint32_t murmur<1>(const Eigen::Matrix<uint32_t, 1, 1>& key);
template uint32_t murmur<2>(const Eigen::Matrix<uint32_t, 2, 1>& key);
template uint32_t murmur<3>(const Eigen::Matrix<uint32_t, 3, 1>& key);
template uint32_t murmur<4>(const Eigen::Matrix<uint32_t, 4, 1>& key);
template uint32_t murmur<5>(const Eigen::Matrix<uint32_t, 5, 1>& key);
template uint32_t murmur<6>(const Eigen::Matrix<uint32_t, 6, 1>& key);
template uint32_t murmur<7>(const Eigen::Matrix<uint32_t, 7, 1>& key);
template uint32_t murmur<8>(const Eigen::Matrix<uint32_t, 8, 1>& key);
template uint32_t murmur<9>(const Eigen::Matrix<uint32_t, 9, 1>& key);

}  // namespace detail

uint32_t
discretize(float value, float min_value, float range_value, uint32_t steps) {
    float nval = (value - min_value) / range_value;
    if (nval < 0.f) return 0;
    if (nval >= 1.f) return steps-1;
    return static_cast<uint32_t>(nval * steps);
}

uint32_t
discretize(float value, float step_size) {
    return static_cast<uint32_t>(value / step_size);
}

}  // namespace triplet_match

namespace std {

template struct hash<::Eigen::Matrix<uint32_t, 1, 1>>;
template struct hash<::Eigen::Matrix<uint32_t, 2, 1>>;
template struct hash<::Eigen::Matrix<uint32_t, 3, 1>>;
template struct hash<::Eigen::Matrix<uint32_t, 4, 1>>;
template struct hash<::Eigen::Matrix<uint32_t, 5, 1>>;
template struct hash<::Eigen::Matrix<uint32_t, 6, 1>>;
template struct hash<::Eigen::Matrix<uint32_t, 7, 1>>;
template struct hash<::Eigen::Matrix<uint32_t, 8, 1>>;
template struct hash<::Eigen::Matrix<uint32_t, 9, 1>>;

} // namespace std
