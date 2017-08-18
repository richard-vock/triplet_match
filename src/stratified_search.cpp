#include <stratified_search>
#include <impl/stratified_search.hpp>
#include <pcl/octree/octree_iterator.h>
#include <pcl/octree/octree_nodes.h>
#include <pcl/octree/octree_container.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/impl/instantiate.hpp>

namespace triplet_match {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template struct stratified_search<type>::impl; \
    template std::pair<mat4f_t, subset_t> stratified_search<type>::impl::find<type>(model<type>& m, float model_match_factor, float score_correspondence_threshold); \
    template stratified_search<type>::stratified_search(typename pcl::PointCloud<type>::ConstPtr cloud, const sample_parameters& sample_params, float model_diameter, float octree_diameter_factor); \
    template stratified_search<type>::~stratified_search(); \
    template void stratified_search<type>::set_model(typename model<type>::sptr_t m); \
    template std::pair<mat4f_t, subset_t> stratified_search<type>::find<type>(model<type>& m, float model_match_factor, float score_correspondence_threshold); \
    template std::pair<std::vector<mat4f_t>, subset_t> stratified_search<type>::find_all<type>(model<type>& m, float model_match_factor, float score_correspondence_threshold);
#include "pcl_point_types.def"

}  // namespace triplet_match

//namespace pcl {
//namespace octree {

//template class OctreeDepthFirstIterator<
    //OctreeBase<
        //OctreeContainerPointIndices,
        //OctreeContainerEmpty
    //>
//>;

//} // octree
//} // pcl

template class PCL_EXPORTS pcl::octree::OctreeBase<pcl::octree::OctreeContainerPointIndices>;


PCL_INSTANTIATE(OctreePointCloudSingleBufferWithLeafDataTVector,
    PCL_XYZ_POINT_TYPES)
PCL_INSTANTIATE(OctreePointCloudDoubleBufferWithLeafDataTVector,
    PCL_XYZ_POINT_TYPES)

PCL_INSTANTIATE(OctreePointCloudSearch, PCL_XYZ_POINT_TYPES)

//#define INSTANTIATE_PCL_POINT_TYPE(type)
//#include "pcl_point_types.def"

//} // octree
//} // pcl
