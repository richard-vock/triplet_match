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
    template stratified_search<type>::stratified_search(typename pcl::PointCloud<type>::ConstPtr cloud, const sample_parameters& sample_params, float model_diameter, float octree_diameter_factor); \
    template stratified_search<type>::~stratified_search(); \
    template void stratified_search<type>::set_model(typename model<type>::sptr_t m); \
    template void stratified_search<type>::reset(); \
    template std::pair<std::vector<mat4f_t>, std::vector<subset_t>> stratified_search<type>::find_all<type>(model<type>&, float, float, float, uint32_t); \
    template typename stratified_search<type>::octree_t::sptr_t stratified_search<type>::get_octree(); \
    template typename stratified_search<type>::octree_t::const_sptr_t stratified_search<type>::get_octree() const; \
    template voxel_score::score_functor<type, type>& stratified_search<type>::get_score_functor();
#include "pcl_point_types.def"

}  // namespace triplet_match

template class PCL_EXPORTS pcl::octree::OctreeBase<pcl::octree::OctreeContainerPointIndices>;
