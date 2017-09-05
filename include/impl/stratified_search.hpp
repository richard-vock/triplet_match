#include <scene>
namespace vs = voxel_score;

namespace triplet_match {

template <typename Point>
struct stratified_search<Point>::impl {
    impl(typename cloud_t::ConstPtr cloud, const sample_parameters& sample_params, float model_diameter, float octree_diameter_factor) :
        cloud_(cloud),
        sample_params_(sample_params),
        scene_(new scene<Point>(cloud)) {
        octree_ = octree_t::from_pointcloud(cloud_, 10, min_voxel_size{octree_diameter_factor * model_diameter});

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
        score_->set_model(model_->cloud(), 100, 0.01f);
        score_->set_scene(scene_->cloud());
    }

    template <typename PointModel>
    std::pair<std::vector<mat4f_t>, std::vector<subset_t>>
    find_all(model<PointModel>& m, float model_match_factor, float score_correspondence_threshold, float early_out_factor) {
        if (!score_) {
            throw std::runtime_error("stratified_search::find(): Model not set");
        }
        std::vector<mat4f_t> transforms;
        std::vector<subset_t> all_matches;
        std::set<uint32_t> considered;

        for (int32_t level = octree_->depth(); level >= 0; --level) {
            std::cout << "############## LEVEL " << level << " #############" << "\n";
            typename octree_t::level_traverse level_rng(octree_->root(), static_cast<uint8_t>(level));
            uint32_t leaf_count = ranges::distance(level_rng);
            uint32_t leaf_idx = 0;
            ranges::for_each(level_rng, [&] (node const& n) {
                std::cout << "####### LEAF " << (++leaf_idx) << "/" << leaf_count << " ######" << "\n";

                // gather subtree indices
                subset_t subset;
                if (leaf_node const* leaf = std::get_if<leaf_node>(&n)) {
                    subset = leaf->points;
                } else {
                    uint32_t subtree_size = 0;
                    ranges::for_each(typename octree_t::leaf_traverse(n), [&] (node const& subtree_leaf) {
                        leaf_node const& leaf = std::get<leaf_node>(subtree_leaf);
                        subset.insert(subset.end(), leaf.points.begin(), leaf.points.end());
                        ++subtree_size;
                    });
                    std::sort(subset.begin(), subset.end());
                    // since by walking bottom-to-top we already considered leaves;
                    // therefore most points must not be considered any longer
                    subset_t residual;
                    std::set_difference(subset.begin(), subset.end(), considered.begin(), considered.end(), std::back_inserter(residual));
                    subset = residual;
                }

                uint32_t subtree_count = subset.size();
                std::cout << "Search in " << subtree_count << " points...   ";
                if (!subtree_count) return;

                //if (leaf_idx) return;
                uint32_t before = transforms.size();
                while(true) {
                    uint32_t min_points = model_match_factor * model_->cloud()->size();
                    if (subset.size() < min_points) {
                        break;
                    }

                    mat4f_t t;
                    uint32_t max_score;
                    std::tie(t, max_score) = scene_->find(m, [&] (const mat4f_t& hyp) { return score_->correspondence_count(hyp, score_correspondence_threshold); }, [&] (uint32_t ccount) { return static_cast<float>(ccount) >= early_out_factor * model_->cloud()->size(); }, sample_params_, subset);

                    if (max_score < min_points) {
                        break;
                    }

                    std::vector<int> matches;
                    std::tie(t, matches) = score_->icp(t, score_correspondence_threshold, 0);

                    // transform is good enough
                    subset_t new_subset;
                    std::set_difference(subset.begin(), subset.end(), matches.begin(), matches.end(), std::back_inserter(new_subset));
                    subset = new_subset;
                    considered.insert(matches.begin(), matches.end());
                    all_matches.push_back(subset_t(matches.begin(), matches.end()));
                    transforms.push_back(t.inverse());
                }
                std::cout << (transforms.size() - before) << " transformations found.\n";
            });
        }

        return {transforms, all_matches};
    }

    typename octree_t::sptr_t get_octree() {
        return octree_;
    }

    typename octree_t::const_sptr_t get_octree() const {
        return octree_;
    }

    vs::score_functor<Point, Point>&
    get_score_functor() {
        return *score_;
    }

    typename cloud_t::ConstPtr cloud_;
    sample_parameters sample_params_;
    typename octree_t::sptr_t octree_;
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
inline std::pair<std::vector<mat4f_t>, std::vector<subset_t>>
stratified_search<Point>::find_all(model<PointModel>& m, float model_match_factor, float score_correspondence_threshold, float early_out_factor) {
    return impl_->find_all(m, model_match_factor, score_correspondence_threshold, early_out_factor);
}

template <typename Point>
inline typename stratified_search<Point>::octree_t::sptr_t
stratified_search<Point>::get_octree() {
    return impl_->get_octree();
}

template <typename Point>
inline typename stratified_search<Point>::octree_t::const_sptr_t
stratified_search<Point>::get_octree() const {
    return impl_->get_octree();
}

template <typename Point>
inline voxel_score::score_functor<Point, Point>&
stratified_search<Point>::get_score_functor() {
    return impl_->get_score_functor();
}

} // triplet_match
