#include <scene>
#include <impl/scene.hpp>

#include <cylinder_traits>
#include <plane_traits>
#include <plane2_traits>
#include <identity_traits>

namespace triplet_match {

namespace detail {

//#define INSTANTIATE_PCL_POINT_TYPE(type)
    //template mat3f_t make_base<type>(typename pcl::PointCloud<type>::ConstPtr cloud, int i, int j);
    //template mat4f_t base_transformation<type, type>(typename pcl::PointCloud<type>::ConstPtr c0, typename pcl::PointCloud<type>::ConstPtr c1, uint32_t i0, uint32_t j0, uint32_t i1, uint32_t j1);
//#include "pcl_point_types.def"

}  // namespace detail

#define INSTANTIATE_PCL_POINT_TYPE(pnt) \
    template class scene<pnt>;
#include "pcl_point_types.def"

}  // namespace triplet_match
