#include <feature>
#include <impl/feature.hpp>

#include <cylinder_traits>
#include <plane_traits>
#include <plane2_traits>
#include <identity_traits>

namespace triplet_match {

#define INSTANTIATE_PCL_POINT_TYPE(pnt) \
    template std::optional<feature_t> feature<pnt>(const pnt&, const pnt&, const pnt&, const curv_info_t<pnt>&, const curv_info_t<pnt>&, const curv_info_t<pnt>&); \
    template discrete_feature_t discretize_feature<pnt>(const feature_t&, const feature_bounds_t&, const discretization_params&); \
    template bool valid<pnt>(const feature_t& f, const feature_bounds_t& bounds);
#include "pcl_point_types.def"

}  // namespace triplet_match
