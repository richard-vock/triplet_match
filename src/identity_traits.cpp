#include <identity_traits>
#include <impl/identity_traits.hpp>

namespace triplet_match {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template struct identity_traits<type>;
#include "pcl_point_types.def"

}  // namespace triplet_match
