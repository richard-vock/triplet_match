#include <feature>
#include <impl/feature.hpp>

#include <cylinder_traits>
#include <plane_traits>
#include <plane2_traits>
#include <identity_traits>

namespace triplet_match {

#define INSTANTIATE_POINT_PROJECTOR_TYPE(proj, pnt) \
    template std::optional<feature_t> feature<proj, pnt>(typename proj::const_handle_t, const pnt&, const pnt&, const pnt&); \
    template discrete_feature_t discretize_feature<proj, pnt>(typename proj::const_handle_t, const feature_t&, const feature_bounds_t&, const discretization_params&); \
    template bool valid<proj, pnt>(typename proj::const_handle_t handle, const feature_t& f, const feature_bounds_t& bounds);
#include "points_projectors_cartesian.def"

}  // namespace triplet_match
