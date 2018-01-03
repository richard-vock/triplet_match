#include <cylinder_traits>
#include <impl/cylinder_traits.hpp>

namespace triplet_match {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template struct cylinder_traits<type>;
#include "pcl_point_types.def"

}  // namespace triplet_match
