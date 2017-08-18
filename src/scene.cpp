#include <scene>
#include <impl/scene.hpp>

namespace triplet_match {

namespace detail {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template mat3f_t make_base<type>(typename pcl::PointCloud<type>::ConstPtr cloud, int i, int j, int k); \
    template mat4f_t base_transformation<type, type>(typename pcl::PointCloud<type>::ConstPtr c0, typename pcl::PointCloud<type>::ConstPtr c1, uint32_t i0, uint32_t j0, uint32_t k0, uint32_t i1, uint32_t j1, uint32_t k1);
#include "pcl_point_types.def"

}  // namespace detail

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template struct scene<type>::impl; \
    template mat4f_t scene<type>::impl::find<type>(model<type>& m, std::function<float (const mat4f_t&)> score_func, const sample_parameters& params, subset_t subset); \
    template scene<type>::scene(typename pcl::PointCloud<type>::ConstPtr cloud); \
    template scene<type>::~scene(); \
    template typename pcl::PointCloud<type>::ConstPtr scene<type>::cloud() const;
#include "pcl_point_types.def"

}  // namespace triplet_match
