#include <scene>
namespace vs = voxel_score;

namespace triplet_match {

namespace detail {

constexpr bool gather_stats = true;

constexpr bool correspondence_on_all = false;
constexpr bool icp_on_all = false;

} // detail

template <typename Point>
struct stratified_search<Point>::impl {
    impl(typename cloud_t::ConstPtr cloud, const sample_parameters& sample_params, float model_diameter, float octree_diameter_factor) :
        cloud_(cloud),
        sample_params_(sample_params),
        scene_(new scene<Point>(cloud)) {
        // generate subset of points for valid tangents only
        for (uint32_t i = 0; i < cloud_->size(); ++i) {
            if (fabs(1.f - cloud_->points[i].getNormalVector3fMap().norm()) < 0.01f) {
                tangent_indices_.push_back(i);
            }
        }
        std::cout << "tangent indices: " << tangent_indices_.size() << "\n";
        octree_ = octree_t::from_pointcloud(cloud_, 10, min_voxel_size{octree_diameter_factor * model_diameter}, tangent_indices_);

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
        score_->set_model(model_->cloud());
        score_->set_scene(scene_->cloud(), tangent_indices_);
    }

    void
    reset() {
        if (!score_) {
            throw std::runtime_error("stratified_search::reset(): Model not set");
        }
        score_->reset(tangent_indices_);
    }

    template <typename PointModel>
    std::pair<std::vector<mat4f_t>, std::vector<subset_t>>
    find_all(model<PointModel>& m, float model_match_factor, float score_distance_factor, float early_out_factor, uint32_t max_icp_iterations) {
        if (!score_) {
            throw std::runtime_error("stratified_search::find(): Model not set");
        }
        std::vector<mat4f_t> transforms;
        std::vector<subset_t> all_matches;
        std::set<uint32_t> considered;

        typename scene<Point>::statistics stats;
        stats.rejection_rate = 0.0;
        uint32_t stat_count = 0;

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
                std::cout << "Search in " << subtree_count << " points...   \n";
                if (!subtree_count) return;

                //if (leaf_idx) return;
                uint32_t before = transforms.size();
                while(true) {
                    uint32_t n_model = detail::correspondence_on_all ? m.cloud()->size() : m.point_count();
                    uint32_t n_scene = detail::correspondence_on_all ? cloud_->size() : subset.size();
                    uint32_t min_points = model_match_factor * n_model;

                    //pdebug("        n_model: {}, n_scene: {}, min_points: {}", n_model, n_scene, min_points);
                    if (n_scene < min_points) {
                        min_points = model_match_factor * n_scene;
                    }
                    if (!min_points) {
                        break;
                    }

                    mat4f_t t;
                    uint32_t max_score;
                    typename scene<Point>::statistics single_stats;
                    single_stats.rejection_rate = 0.0;
                    std::tie(t, max_score) = scene_->find(
                        m,
                        [&](const mat4f_t& hyp) {
                            return score_->correspondence_count(
                                hyp, score_distance_factor, !detail::correspondence_on_all);
                        },
                        [&](uint32_t ccount) {
                            return static_cast<float>(ccount) >=
                                   early_out_factor * n_model;
                        },
                        sample_params_, subset,
                        detail::gather_stats ? &stats : nullptr);

                    if (detail::gather_stats) {
                        double delta = single_stats.rejection_rate - stats.rejection_rate;
                        stats.rejection_rate += delta / (++stat_count);
                    }

                    //pdebug("score after sampling: {}", max_score);
                    if (max_score < min_points) {
                        //pdebug("stop (non-sufficient score)");
                        break;
                    }

                    //pdebug("final accept at {} of {} necessary points", max_score, min_points);


                    std::vector<int> matches;
                    std::tie(t, matches) = score_->icp(t, score_distance_factor, max_icp_iterations, !detail::icp_on_all, true);

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

        if (detail::gather_stats) {
            std::cout << "Statistics:" << "\n";
            std::cout << "   Rejection Rate: " << std::setprecision(2) << (stats.rejection_rate * 100.0) << "\n";
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
    subset_t tangent_indices_;
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
inline void
stratified_search<Point>::reset() {
    impl_->reset();
}

template <typename Point>
template <typename PointModel>
inline std::pair<std::vector<mat4f_t>, std::vector<subset_t>>
stratified_search<Point>::find_all(model<PointModel>& m, float model_match_factor, float score_distance_factor, float early_out_factor, uint32_t max_icp_iterations) {
    return impl_->find_all(m, model_match_factor, score_distance_factor, early_out_factor, max_icp_iterations);
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
