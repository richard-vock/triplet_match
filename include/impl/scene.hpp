#include <random>
#include <chrono>
#include "timer.hpp"
#include <boost/compute/interop/eigen.hpp>
#include <pcl/registration/icp.h>

namespace triplet_match {

namespace detail {

//constexpr uint32_t random_kernel_size = 100;
constexpr bool early_out = true;
constexpr bool deterministic = true;
constexpr double match_probability = 0.99;
constexpr uint64_t min_sample_count = 10ull;
constexpr float corr_dist_factor = 1.0;
constexpr bool allow_scale = false;
constexpr int threads_per_block = 512;
constexpr int query_limit = 10;

}  // namespace detail

template <typename Proj, typename Point>
struct scene<Proj, Point>::impl {
    impl(typename cloud_t::Ptr cloud) : cloud_(cloud) {}

    ~impl() {
    }

    std::vector<match_t>
    find_all(model<Proj, Point>& m, float octree_diameter_factor, float model_match_factor, float early_out_factor, const sample_parameters& sample_params, uint32_t max_icp_iterations, bool only_first) {
        subset_t tangent_indices;
        for (auto && [idx, pnt] : vw::zip(vw::ints(0ul), cloud_->points)) {
            if (tangent(pnt).norm() > 0.7f) {
                tangent_indices.push_back(idx);
            }
        }
        cloud_->set_indices(tangent_indices);
        mask_.resize(cloud_->size(), false);
        uint32_t considered = 0;

        std::vector<match_t> results;
        float min_voxel = octree_diameter_factor * m.cloud()->diameter();
        pdebug("min voxel size: {}", min_voxel);
        typename octree_t::sptr_t octree = octree_t::from_pointcloud(cloud_, 10, min_voxel_size{min_voxel});

        bool found = false;

        for (int32_t level = octree->depth(); level >= 0; --level) {
            if (only_first && found) break;

            std::cout << "############## LEVEL " << level << " #############" << "\n";
            typename octree_t::level_traverse level_rng(octree->root(), static_cast<uint8_t>(level));
            uint32_t leaf_count = ranges::distance(level_rng);
            uint32_t leaf_idx = 0;
            ranges::for_each(level_rng, [&] (node const& n) {
                if (only_first && found) return;
                std::cout << "####### LEAF " << (++leaf_idx) << "/" << leaf_count << " ######" << "\n";

                // gather subtree indices
                subset_t sample_candidates;
                // initialize sample candidates by voxel indices
                if (leaf_node const* leaf = std::get_if<leaf_node>(&n)) {
                    sample_candidates = leaf->points;
                } else {
                    uint32_t subtree_size = 0;
                    ranges::for_each(typename octree_t::leaf_traverse(n), [&] (node const& subtree_leaf) {
                        leaf_node const& leaf = std::get<leaf_node>(subtree_leaf);
                        sample_candidates.insert(sample_candidates.end(), leaf.points.begin(), leaf.points.end());
                        ++subtree_size;
                    });
                    std::sort(sample_candidates.begin(), sample_candidates.end());
                }

                // restrict sampling to tangent indices (score/correspondences are done on all points, neighborhood queries on all tangent indices)
                subset_t residual;
                std::set_intersection(sample_candidates.begin(), sample_candidates.end(), tangent_indices.begin(), tangent_indices.end(), std::back_inserter(residual));
                sample_candidates = residual;

                uint32_t subtree_count = sample_candidates.size();
                println("Search in {} sample candidates...", subtree_count);
                if (!subtree_count) return;

                uint32_t before = results.size();
                while(true) {
                    if (only_first && found) break;
                    uint32_t min_points = model_match_factor * m.cloud()->size();
                    // check if remaining points could even surpass the validity threshold
                    if ((cloud_->size() - considered) < min_points) {
                        min_points = model_match_factor * cloud_->size();
                    }

                    if (!min_points) {
                        break;
                    }

                    match_t match = find(
                        m,
                        early_out_factor * m.cloud()->size(),
                        sample_params, sample_candidates);

                    if (std::get<1>(match).size() < min_points) {
                        break;
                    }

                    mat4f_t uvw_trans;
                    subset_t scene_corrs, model_corrs;
                    typename Proj::handle_t proj;
                    std::tie(uvw_trans, scene_corrs, model_corrs, proj) = max_icp_iterations ? icp_(m, match, max_icp_iterations) : match;

                    pdebug("  accept at {} of {} points", scene_corrs.size(), min_points);

                    // transform is good enough
                    pdebug("removed {} indices from index list", scene_corrs.size());
                    subset_t new_subset;
                    std::set_difference(sample_candidates.begin(), sample_candidates.end(), scene_corrs.begin(), scene_corrs.end(), std::back_inserter(new_subset));
                    sample_candidates = new_subset;

                    for (auto idx : scene_corrs) {
                        mask_[idx] = true;
                    }
                    considered += scene_corrs.size();

                    results.push_back({uvw_trans.inverse(), scene_corrs, model_corrs, proj});
                    found = true;
                }
                std::cout << (results.size() - before) << " transformations found.\n";
            });
        }

        return results;
    }

    match_t
    find(model<Proj, Point>& m, uint32_t early_out_threshold, const sample_parameters& params, const subset_t& sample_candidates) {
        uint32_t n_model = m.point_count();
        uint32_t n_scene = sample_candidates.size();

        float lower = m.diameter() * params.min_diameter_factor;
        float upper = m.diameter() * params.max_diameter_factor;
        lower *= lower;

        std::mt19937 rng;
        uint32_t seed = 13;
        if (!detail::deterministic) {
            auto now = std::chrono::system_clock::now();
            seed = now.time_since_epoch().count();
        }
        rng.seed(seed);

        uint32_t best_score = 0;
        mat4f_t  best_uvw_trans = mat4f_t::Identity();
        typename Proj::handle_t best_projector = nullptr;
        uint64_t valid_sample_count = 0, sample_count = 0;

        double outer_prob = static_cast<double>(n_model) / n_scene;
        uint64_t outer_bound = static_cast<uint64_t>(std::log(1.0 - detail::match_probability) / std::log(1.0 - outer_prob));
        outer_bound = std::max(outer_bound, detail::min_sample_count);
        outer_bound = std::min(outer_bound, static_cast<uint64_t>(n_scene));

        pdebug("outer bound: {},  p: {} / {} = {}", outer_bound, n_model, n_scene, outer_prob);

        for (auto i : vw::sample(sample_candidates, outer_bound, rng)) {
            if (mask_[i]) continue;
            const Point& p1 = cloud_->points[i];

            auto nn = cloud_->radius_search_inclusive(upper, p1).first;
            if (nn.empty()) continue;

            double prob = static_cast<double>(n_model) / nn.size();
            uint64_t inner_bound = static_cast<uint64_t>(-std::log(1.0 - detail::match_probability) / prob);
            inner_bound = std::max(inner_bound, detail::min_sample_count);
            inner_bound = std::min(inner_bound, nn.size());
            //pdebug("inner bound: {},  p: {} / {} = {}", inner_bound, n_model, nn.size(), prob);

            //uint32_t valid_in_ball = 0;
            //uint32_t max_ball_score = 0;

            for (auto j : vw::sample(nn, inner_bound, rng)) {
                if (mask_[j]) continue;
                if (static_cast<uint32_t>(j) == i) continue;

                bool debug = false; //i==54 && j ==313;
                const Point& p2 = cloud_->points[j];
                ++sample_count;

                if ((p2.getVector3fMap() - p1.getVector3fMap()).squaredNorm() < lower) continue;

                typename Proj::handle_t proj;
                if constexpr (Proj::sample_count == 2) {
                    proj = traits_t::init_from_samples({p1, p2});
                } else {
                    proj = traits_t::init_from_samples({p1});
                }
                auto f = traits_t::feature(proj, p1, p2);

                if (!traits_t::valid(proj, f, m.feature_bounds(), M_PI / 10.f, 2.0*M_PI, 0.4f, 1.f)) {
                    continue;
                }

                auto && [q_first, q_last] = m.query(f, debug);
                if (q_first != q_last) {
                    //pdebug("query count: {}", std::distance(q_first, q_last));
                    ++valid_sample_count;
                    //++valid_in_ball;
                }

                uint32_t query = 0;
                for (auto it = q_first; it != q_last; ++it) {
                    if (detail::query_limit > 0 && (query++) > detail::query_limit) break;
                    auto && [m_i, m_j] = it->second;
                    //debug = i==54 && j ==313 && m_i == 54 && m_j == 313;

                    vec3f_t uvw_ref_model = Proj::project(m.projector(), m.cloud()->points[m_i].getVector3fMap());
                    vec3f_t uvw_ref_scene = Proj::project(proj, cloud_->points[i].getVector3fMap());
                    vec3f_t uvw_tgt_0_model = Proj::tangent(m.projector(), m.cloud()->points[m_i]);
                    vec3f_t uvw_tgt_1_model = Proj::tangent(m.projector(), m.cloud()->points[m_j]);
                    vec3f_t uvw_tgt_0_scene = Proj::tangent(proj, cloud_->points[i]);
                    vec3f_t uvw_tgt_1_scene = Proj::tangent(proj, cloud_->points[j]);
                    mat4f_t uvw_trans = base_to_base_(uvw_ref_scene, uvw_tgt_0_scene, uvw_tgt_1_scene, uvw_ref_model, uvw_tgt_0_model, uvw_tgt_1_model, debug);

                    project_(m, proj, uvw_trans);
                    uint32_t score = scene_corrs_.size();
                    //max_ball_score = std::max(max_ball_score, score);
                    if (score > best_score) {
                        best_uvw_trans = uvw_trans;
                        best_score = score;
                        best_projector = proj;

                        if (detail::early_out && best_score >= early_out_threshold) {
                            fmt::print("early out at: {} of {} points\n", best_score, m.cloud()->size());
                            return {uvw_trans, scene_corrs_, model_corrs_, proj};
                        }
                    }
                }
            }

            //pdebug("valid pairs in ball: {}", valid_in_ball);
            //pdebug("valid pairs in ball: {}, with max score of: {}", valid_in_ball, max_ball_score);
        }


        if (best_projector) {
            project_(m, best_projector, best_uvw_trans);
        } else {
            scene_corrs_.clear();
        }
        return {best_uvw_trans, scene_corrs_, model_corrs_, best_projector};
    }

    typename cloud_t::Ptr
    instantiate(model<Proj, Point>& m, const match_t& match) {
        mat4f_t t;
        subset_t model_corrs;
        typename Proj::handle_t proj;
        std::tie(t, std::ignore, model_corrs, proj) = match;

        typename cloud_t::Ptr inst = cloud_t::empty();
        for (uint32_t idx : model_corrs) {
            vec3f_t uvw_model = m.uvw_cloud()->points[idx].getVector3fMap();
            vec3f_t uvw_scene = (t * uvw_model.homogeneous()).head(3);
            Point inst_pnt;
            inst_pnt.getVector3fMap() = Proj::unproject(proj, uvw_scene);
            inst->push_back(inst_pnt);
        }
        // scene example
        //for (auto idx : corrs) {
            //vec3f_t pos = cloud_->points[idx].getVector3fMap();
            //vec4f_t uvw = t * Proj::project(proj, pos).homogeneous();
            //Point inst_pnt;
            //inst_pnt.getVector3fMap() = Proj::unproject(m.projector(), uvw.head(3));
            //inst->push_back(inst_pnt);
        //}

        return inst;
    }

    void
    project_(model<Proj, Point>& m, typename Proj::const_handle_t scene_proj, const mat4f_t& uvw_trans, float corr_dist_factor = -1.f, bool debug = false) {
        uint32_t concurrency = std::thread::hardware_concurrency() - 1;
        uint32_t batch_size = cloud_->size() / concurrency;
        if (cloud_->size() % concurrency > 0) {
            ++batch_size;
        }

        float thres = corr_dist_factor < 0.f ? detail::corr_dist_factor * m.uvw_resolution() : corr_dist_factor * m.uvw_resolution();

        std::vector<std::future<std::pair<subset_t, subset_t>>> futures;
        for (auto chunk : vw::chunk(vw::ints(0ul, cloud_->size()), batch_size)) {
            futures.push_back(std::async(std::launch::async, [&, chunk] () {
                subset_t sub_scene_corrs, sub_model_corrs;
                for (uint32_t idx : chunk) {
                    if (mask_[idx]) continue;
                    vec3f_t uvw = Proj::project(scene_proj, cloud_->points[idx].getVector3fMap());
                    uvw = (uvw_trans * uvw.homogeneous()).head(3);
                    std::optional<uint32_t> uvw_n = m.voxel_query(uvw);
                    if (!uvw_n) {
                        continue;
                    }
                    float dist = Proj::intrinsic_distance(m.projector(), uvw, m.uvw_cloud()->points[uvw_n.value()].getVector3fMap());
                    if (dist <= thres) {
                        sub_scene_corrs.push_back(idx);
                        sub_model_corrs.push_back(uvw_n.value());
                    }
                }
                //if (debug) pdebug("chunk score: {} / {}", sub_scene_corrs.size(), chunk.size());
                return std::pair<subset_t, subset_t>(sub_scene_corrs, sub_model_corrs);
            }));
        }

        scene_corrs_.clear();
        model_corrs_.clear();
        for (auto& fut : futures) {
            auto && [subs, subm] = fut.get();
            scene_corrs_.insert(scene_corrs_.end(), subs.begin(), subs.end());
            model_corrs_.insert(model_corrs_.end(), subm.begin(), subm.end());
        }

        //if (debug) pdebug("score: {} / {}", scene_corrs_.size(), cloud_->size());
    }

    mat3f_t gram_schmidt_(const mat3f_t& base, bool col_space = true) {
        mat3f_t m = base;

        if (col_space) {
            m.col(1) -= m.col(1).dot(m.col(0)) * m.col(0);
            m.col(1).normalize();
            m.col(2) -= m.col(2).dot(m.col(0)) * m.col(0);
            m.col(2) -= m.col(2).dot(m.col(1)) * m.col(1);
            m.col(2).normalize();
        } else {
            m.row(1) -= m.row(1).dot(m.row(0)) * m.row(0);
            m.row(1).normalize();
            m.row(2) -= m.row(2).dot(m.row(0)) * m.row(0);
            m.row(2) -= m.row(2).dot(m.row(1)) * m.row(1);
            m.row(2).normalize();
        }

        return m;
    }

    mat4f_t base_to_base_(const vec3f_t& origin_0, const mat3f_t& base_0, const vec3f_t& origin_1, const mat3f_t& base_1) {
        mat4f_t t = mat4f_t::Identity();

        // rotational part
        quatf_t q0 = quatf_t::FromTwoVectors(base_0.col(0), base_1.col(0));
        vec3f_t d1 = q0._transformVector(base_0.col(1));
        quatf_t q1 = quatf_t::FromTwoVectors(d1, base_1.col(1));

        // translational part
        t.topLeftCorner<3,3>() = (q1*q0).toRotationMatrix();
        t.block<3,1>(0,3) = origin_1 - t.topLeftCorner<3,3>() * origin_0;

        return t;
    }

    mat4f_t base_to_base_(const vec3f_t& origin_0, const vec3f_t& dir_a_0, const vec3f_t& dir_b_0, const vec3f_t& origin_1, const vec3f_t& dir_a_1, const vec3f_t& dir_b_1, bool debug = false) {
        mat4f_t t = mat4f_t::Identity();

        // rotational part
        quatf_t q0 = quatf_t::FromTwoVectors(dir_a_0, dir_a_1);

        // localized version of second vectors
        vec3f_t l0 = q0._transformVector(dir_b_0);
        vec3f_t l1 = dir_b_1;
        // "local" due to gram-schmidt orthogonalization with respect to first vectors:
        l0 -= l0.dot(dir_a_1) * dir_a_1;
        l1 -= l1.dot(dir_a_1) * dir_a_1;
        quatf_t q1 = quatf_t::FromTwoVectors(l0, l1);

        // translational part
        t.topLeftCorner<3,3>() = (q1*q0).toRotationMatrix();
        t.block<3,1>(0,3) = origin_1 - t.topLeftCorner<3,3>() * origin_0;

        return t;
    }

    match_t
    icp_(model<Proj, Point>& m, const match_t& match, uint32_t max_iterations) {
        if (max_iterations == 0) return match;

        mat4f_t best_trans;
        subset_t best_corrs;
        typename Proj::handle_t proj;
        std::tie(best_trans, best_corrs, std::ignore, proj) = match;


        // ICP is performed in uvw space
        typename cloud_t::Ptr uvw_scene = cloud_t::empty();
        for (auto idx : vw::ints(0ul, cloud_->size())) {
            if (mask_[idx]) continue;
            const Point& pnt = cloud_->points[idx];
            vec3f_t uvw = Proj::project(proj, pnt.getVector3fMap());
            Point uvw_point;
            uvw_point.getVector4fMap() = best_trans * uvw.homogeneous();
            uvw_scene->push_back(uvw_point);
        }
        typename cloud_t::Ptr uvw_model = m.uvw_cloud();

        float max_corr_dist = 3.f * detail::corr_dist_factor * m.uvw_resolution();
        pcl::IterativeClosestPoint<Point, Point, double> icp;
        icp.setInputSource(uvw_scene->native());
        icp.setInputTarget(uvw_model->native());
        icp.setSearchMethodSource(uvw_scene->kdtree());
        icp.setSearchMethodTarget(uvw_model->kdtree());
        icp.setMaximumIterations(max_iterations);
        icp.setMaxCorrespondenceDistance(max_corr_dist);
        //double test_before = icp.getFitnessScore(max_corr_dist);
        icp.align(*uvw_scene->native());
        //double test_after = icp.getFitnessScore(max_corr_dist);

        best_trans = icp.getFinalTransformation().template cast<float>() * best_trans;
        project_(m, proj, best_trans);

        return {best_trans, scene_corrs_, model_corrs_, proj};
    }

    typename cloud_t::ConstPtr cloud() const {
        return cloud_;
    }

    typename cloud_t::Ptr      cloud_;

    subset_t scene_corrs_;
    subset_t model_corrs_;
    std::vector<bool> mask_;
};


template <typename Proj, typename Point>
inline
scene<Proj, Point>::scene(typename cloud_t::Ptr cloud) : impl_(new impl(cloud)) {
}

template <typename Proj, typename Point>
inline
scene<Proj, Point>::~scene() {
}

template <typename Proj, typename Point>
inline std::vector<typename scene<Proj, Point>::match_t>
scene<Proj, Point>::find_all(model<Proj, Point>& m, float octree_diameter_factor, float model_match_factor, float early_out_factor, const sample_parameters& sample_params, uint32_t max_icp_iterations, bool only_first) {
    return impl_->find_all(m, octree_diameter_factor, model_match_factor, early_out_factor, sample_params, max_icp_iterations, only_first);
}

template <typename Proj, typename Point>
inline typename scene<Proj, Point>::match_t
scene<Proj, Point>::find(model<Proj, Point>& m, uint32_t early_out_threshold, const sample_parameters& sample_params, const subset_t& sample_candidates) {
    return impl_->find(m, early_out_threshold, sample_params, sample_candidates);
}

template <typename Proj, typename Point>
inline typename scene<Proj, Point>::cloud_t::Ptr
scene<Proj, Point>::instantiate(model<Proj, Point>& m, const match_t& match) {
    return impl_->instantiate(m, match);
}

template <typename Proj, typename Point>
inline typename scene<Proj, Point>::cloud_t::ConstPtr
scene<Proj, Point>::cloud() const {
    return impl_->cloud();
}

}  // namespace triplet_match
