#include <model>
#include <impl/model.hpp>

#include <cylinder_traits>
#include <plane_traits>
#include <plane2_traits>
#include <identity_traits>

namespace triplet_match {

#define INSTANTIATE_POINT_PROJECTOR_TYPE(proj, pnt) \
    template class model<proj, pnt>;
#include "points_projectors_cartesian.def"

}  // namespace triplet_match
