#include <model>
#include <impl/model.hpp>

namespace triplet_match {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template class model<type>;
    //template model<type>::model(typename pcl::PointCloud<type>::ConstPtr cloud, discretization_params params);
    //template model<type>::~model();
    //template std::future<void> model<type>::init(const sample_parameters& sample_params);
    //template std::future<void> model<type>::init(const subset_t& subset, const sample_parameters& sample_params);
    //template std::pair<typename model<type>::pair_iter_t, typename model<type>::pair_iter_t> model<type>::query<type>(const type& p1, const type& p2);
    //template float model<type>::diameter() const;
    //template uint32_t model<type>::point_count() const;
    //template uint64_t model<type>::pair_count() const;
    //template typename pcl::PointCloud<type>::ConstPtr model<type>::cloud() const;
    //template std::set<uint32_t> model<type>::used_points() const;
    //template void model<type>::write_octave_density_maps(const std::string& folder, const std::string& data_file_prefix, const std::string& script_file) const;
#include "pcl_point_types.def"

}  // namespace triplet_match
