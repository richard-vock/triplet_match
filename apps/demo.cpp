#include <iostream>

#include <pcl/io/pcd_io.h>

#include <voxel_score/score_functor>
namespace vs = voxel_score;
#include <triplet_match/stratified_search>
namespace tr = triplet_match;

typedef pcl::PointSurfel point_t;
typedef pcl::PointCloud<point_t> cloud_t;

int main (int argc, char const* argv[]) {
    std::string model_path = "/path/to/model.pcd";
    std::string scene_path = "/path/to/scene.pcd";

    // load model cloud
    cloud_t::Ptr model_cloud(new cloud_t());
    pcl::io::loadPCDFile(model_path, *model_cloud);
    // only necessary if tangents are stored as normals
    for (auto& pnt : *model_cloud) {
        pnt.data_c[1] = pnt.normal_x;
        pnt.data_c[2] = pnt.normal_y;
        pnt.data_c[3] = pnt.normal_z;
    }

    tr::discretization_params discr_params {
        15, // discretization bins of distances (between min and max)
        12, // discretization bins of angles (in this case 12° steps between 0° and 180°)
    };


    // init search model
    std::shared_ptr<tr::model<point_t>> model;
    model = std::make_shared<tr::model<point_t>>(model_cloud, discr_params);
    tr::sample_parameters sample_params;
    sample_params.min_diameter_factor = 0.4f; // higher means less feasible point pairs (= faster but possibly less correct)
    sample_params.max_diameter_factor = 0.8f; // try to leave as is
    sample_params.min_orthogonality = 0.8f; // never used iirc
    sample_params.redundancy_factor = 1.f; // never used iirc
    auto sync_model = model->init(sample_params);
    sync_model.wait();

    // load scene cloud
    cloud_t::Ptr scene_cloud(new cloud_t());
    pcl::io::loadPCDFile(scene_path, *scene_cloud);
    // only necessary if tangents are stored as normals
    for (auto& pnt : *scene_cloud) {
        pnt.data_c[1] = pnt.normal_x;
        pnt.data_c[2] = pnt.normal_y;
        pnt.data_c[3] = pnt.normal_z;
    }
    std::vector<uint32_t> subset(scene_cloud->size());
    std::iota(subset.begin(), subset.end(), 0);

    // create search object and initialize
    tr::stratified_search<point_t>::uptr_t search;
    search = std::make_unique<tr::stratified_search<point_t>>(scene_cloud, sample_params, model->diameter(), 3.f);
    search->set_model(model);

    ////// find occurrences (if this is done multiple times search->reset() has to be called inbetween)

    // how much point overlap (percentage) is considered "enough" - this needs to be adjusted
    float match_coverage = 0.8f;
    // if eps is the cloud resolution, corr_dist_factor * eps is the maximum distance for correspondences
    // might work out-of-the-box
    float corr_dist_factor = 2.f;
    // how much point overlap (percentage) is considered "enough" for early return
    // notice how >100% might actualy make sense depending on the scene - might work out-of-the-box
    float early_out_factor = 2.f;
    // maximum number if ICP iterations after a valid transform has been found - *cannot* make the result worse
    // just use this as is
    uint32_t max_icp_iterations = 5;

    // actually peform the search
    auto && [ts, subsets] = search->find_all(*model, match_coverage, corr_dist_factor, early_out_factor, max_icp_iterations);

    // ts is a vector of eigen 4x4 transformation matrices from model to scene occurence
    // subsets is a vector of vector of indices - each vector is the matching scene subset (i.e. corrospendence indices)

    std::cout << ts.size() << " " << subsets.size() << "\n"; // prevent "unused variable warning"
}
