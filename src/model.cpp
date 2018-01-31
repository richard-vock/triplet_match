#include <model>
#include <impl/model.hpp>

namespace triplet_match {

#define INSTANTIATE_PCL_POINT_TYPE(pnt) \
    template class model<pnt>;
#include "pcl_point_types.def"

}  // namespace triplet_match
