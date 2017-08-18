#include <scene>
#include <pcl/octree/octree_pointcloud.h>
#include <voxel_score/score_functor>
namespace vs = voxel_score;

namespace triplet_match {

template <typename Point>
struct stratified_search<Point>::impl {
    impl(typename cloud_t::ConstPtr cloud, const sample_parameters& sample_params, float model_diameter, float octree_diameter_factor) :
        cloud_(cloud),
        sample_params_(sample_params),
        octree_(octree_diameter_factor * model_diameter),
        scene_(new scene<Point>(cloud)) {

        octree_.setInputCloud(cloud_);
        octree_.addPointsFromInputCloud();

        std::cout << "Octree Depth: " << octree_.getTreeDepth() << "\n";

        valid_subset_.resize(cloud_->size());
        std::iota(valid_subset_.begin(), valid_subset_.end(), 0);
    }

    ~impl() {
    }

    void
    set_model(typename model<Point>::sptr_t m) {
        model_ = m;

        // create score functor
        gstate_ = std::make_shared<vs::gpu_state>();
        score_ = std::make_unique<vs::score_functor<Point, Point>>(gstate_);
        score_->set_model(model_->cloud(), 100);
        score_->set_scene(scene_->cloud());
    }

    template <typename PointModel>
    std::pair<mat4f_t, subset_t>
    find(model<PointModel>& m, float model_match_factor, float score_correspondence_threshold) {
        if (!score_) {
            throw std::runtime_error("stratified_search::find(): Model not set");
        }
        for(auto leaf_it = octree_.leaf_begin(); leaf_it != octree_.leaf_end(); ++leaf_it) {
            // get point subset
            std::vector<int> leaf_indices;
            typename pcl::octree::OctreePointCloud<Point>::LeafNode* leaf_node =
                dynamic_cast<typename pcl::octree::OctreePointCloud<Point>::LeafNode*>(*leaf_it);
            leaf_node->getContainer().getPointIndices(leaf_indices);
            std::sort(leaf_indices.begin(), leaf_indices.end());

            // only valid points
            subset_t subset;
            std::set_intersection(leaf_indices.begin(), leaf_indices.end(), valid_subset_.begin(), valid_subset_.end(), std::back_inserter(subset));
            //std::cout << "----     Octree cell with " << subset.size() << " points" << "\n";


            // find best transform
            //mat4f_t t = scene_->find(m, *score_, sample_params_, subset);
            //std::vector<int> matches = score_->correspondences(t, score_correspondence_threshold);
            if (subset.size() < leaf_indices.size()) {
                continue;
            }
            std::cout << "----     Octree cell with " << subset.size() << " points" << "\n";
            subset.clear();
            std::set_difference(valid_subset_.begin(), valid_subset_.end(), leaf_indices.begin(), leaf_indices.end(), std::back_inserter(subset));
            valid_subset_ = subset;
            return {mat4f_t::Identity(), subset_t(leaf_indices.begin(), leaf_indices.end())};
            //if (static_cast<float>(matches.size()) > model_match_factor * model_->cloud()->size()) {
                //subset.clear();
                //std::set_difference(valid_subset_.begin(), valid_subset_.end(), matches.begin(), matches.end(), std::back_inserter(subset));
                //valid_subset_ = subset;
                //return {t, subset_t(matches.begin(), matches.end())};
            //}
        }

        return {mat4f_t::Identity(), subset_t()};
    }

    template <typename PointModel>
    std::pair<std::vector<mat4f_t>, subset_t>
    find_all(model<PointModel>& m, float model_match_factor, float score_correspondence_threshold) {
        if (!score_) {
            throw std::runtime_error("stratified_search::find(): Model not set");
        }
        std::vector<mat4f_t> transforms;
        subset_t all_matches;
        uint32_t leaf_idx = 0;
        for(auto leaf_it = octree_.leaf_begin(); leaf_it != octree_.leaf_end(); ++leaf_it) {
            std::cout << "############## LEAF " << (++leaf_idx) << " #############" << "\n";
            // get point subset
            std::vector<int> leaf_indices;
            typename pcl::octree::OctreePointCloud<Point>::LeafNode* leaf_node =
                dynamic_cast<typename pcl::octree::OctreePointCloud<Point>::LeafNode*>(*leaf_it);
            leaf_node->getContainer().getPointIndices(leaf_indices);
            std::sort(leaf_indices.begin(), leaf_indices.end());

            subset_t subset(leaf_indices.begin(), leaf_indices.end());
            uint32_t before = transforms.size();
            while(true) {
                uint32_t min_points = model_match_factor * model_->cloud()->size();
                if (subset.size() < min_points) {
                    break;
                }
                mat4f_t t = scene_->find(m, *score_, sample_params_, subset);
                std::vector<int> matches = score_->correspondences(t, score_correspondence_threshold);
                if (static_cast<float>(matches.size()) < min_points) {
                    break;
                }

                // transform is good enough
                subset_t new_subset;
                std::set_difference(subset.begin(), subset.end(), matches.begin(), matches.end(), std::back_inserter(new_subset));
                subset = new_subset;
                all_matches.insert(all_matches.end(), matches.begin(), matches.end());
                transforms.push_back(t);
            }
            std::cout << "Found: " << (transforms.size() - before) << " transformations in " << leaf_indices.size() << " points\n";
        }

        std::sort(all_matches.begin(), all_matches.end());
        auto new_end = std::unique(all_matches.begin(), all_matches.end());
        all_matches.resize(std::distance(all_matches.begin(), new_end));

        return {transforms, all_matches};
    }

    typename cloud_t::ConstPtr cloud_;
    sample_parameters sample_params_;
    pcl::octree::OctreePointCloud<Point> octree_;
    typename scene<Point>::uptr_t scene_;
    typename model<Point>::sptr_t model_;
    subset_t valid_subset_;
    vs::gpu_state::sptr_t gstate_;
    typename vs::score_functor<Point, Point>::uptr_t score_;
};

template <typename Point>
inline
stratified_search<Point>::stratified_search(typename cloud_t::ConstPtr cloud, const sample_parameters& sample_params, float model_diameter, float octree_diameter_factor) : impl_(new impl(cloud, sample_params, model_diameter, octree_diameter_factor)) {
}

template <typename Point>
inline
stratified_search<Point>::~stratified_search() {
}

template <typename Point>
inline void
stratified_search<Point>::set_model(typename model<Point>::sptr_t m) {
    impl_->set_model(m);
}

template <typename Point>
template <typename PointModel>
inline
std::pair<mat4f_t, subset_t>
stratified_search<Point>::find(model<PointModel>& m, float model_match_factor, float score_correspondence_threshold) {
    return impl_->find(m, model_match_factor, score_correspondence_threshold);
}

template <typename Point>
template <typename PointModel>
std::pair<std::vector<mat4f_t>, subset_t>
stratified_search<Point>::find_all(model<PointModel>& m, float model_match_factor, float score_correspondence_threshold) {
    return impl_->find_all(m, model_match_factor, score_correspondence_threshold);
}

} // triplet_match
