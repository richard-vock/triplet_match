#include <plane2_traits>
#include <impl/plane2_traits.hpp>

namespace triplet_match {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template struct plane2_traits<type>;
#include "pcl_point_types.def"

}  // namespace triplet_match

