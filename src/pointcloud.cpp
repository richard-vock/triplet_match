#include <pointcloud>
#include <impl/pointcloud.hpp>

namespace triplet_match {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template class pointcloud<type>; \
    template class knn_index_range<type>; \
    template class knn_dist_range<type>; \
    template class knn_sqr_dist_range<type>;
#include "pcl_point_types.def"

}  // namespace triplet_match
