#include <octree>
#include <impl/octree.hpp>

namespace triplet_match {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template class octree<type>;
#include "pcl_point_types.def"

} // triplet_match
