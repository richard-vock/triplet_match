#include <random>
#include <chrono>
#include "timer.hpp"
#include <boost/compute/interop/eigen.hpp>
#include <pcl/registration/icp.h>
#include <progress_bar>

namespace triplet_match {

namespace detail {

//constexpr uint32_t random_kernel_size = 100;
constexpr bool early_out = true;
constexpr bool deterministic = false;
constexpr double match_probability = 0.99;
constexpr uint64_t min_sample_count = 10ull;
constexpr float corr_dist_factor = 3.0;
constexpr bool allow_scale = false;
constexpr int threads_per_block = 512;
constexpr int query_limit = 200; //disabled: -1;
constexpr bool force_exhaustive = true;
constexpr bool samples_on_all = true;
constexpr bool final_corrs_on_all = true;

}  // namespace detail

template <typename Proj, typename Point>
struct scene<Proj, Point>::impl {
    impl(typename cloud_t::Ptr cloud) : cloud_(cloud) {}

    ~impl() {
    }

    std::vector<match_t>
    find_all(model<Proj, Point>& m, float dist_thres, float octree_diameter_factor, float model_match_factor, float early_out_factor, const sample_parameters& sample_params, uint32_t max_icp_iterations, bool only_first, subset_t* firsts) {
        subset_t tangent_indices;
        uint32_t possible_scene = 0;
        for (auto && [idx, pnt] : vw::zip(vw::ints(0ul), cloud_->points)) {
            if (tangent(pnt).norm() > 0.7f) {
                tangent_indices.push_back(idx);
            }
            if (considered_correspondence_(pnt)) {
                ++possible_scene;
            }
        }
        cloud_->set_indices(tangent_indices);
        mask_ = std::vector<bool>(cloud_->size(), false);
        uint32_t considered = 0;

        std::vector<match_t> results;
        float min_voxel = octree_diameter_factor * m.cloud()->diameter();
        //pdebug("min voxel size: {}", min_voxel);
        typename octree_t::sptr_t octree = octree_t::from_pointcloud(cloud_, 10, min_voxel_size{min_voxel});

        bool found = false;

        uint32_t possible_model = 0;
        for (const auto& pnt : *m.cloud()) {
            if (considered_correspondence_(pnt)) {
                ++possible_model;
            }
        }

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
                    uint32_t min_points = model_match_factor * possible_model;
                    // check if remaining points could even surpass the validity threshold
                    int32_t possible_left = static_cast<int64_t>(possible_scene) - static_cast<int64_t>(considered);
                    if (possible_left < static_cast<int64_t>(min_points)) {
                        min_points = model_match_factor * possible_left;
                        if (min_points < 0.4f * possible_model) {
                            break;
                        }
                    }

                    match_t match = find(
                        m,
                        dist_thres,
                        early_out_factor * possible_model,
                        sample_params, sample_candidates, firsts);

                    //pdebug("rel score: {}", static_cast<double>(std::get<1>(match).size()) / possible_model);
                    if (std::get<1>(match).size() < min_points) {
                        break;
                    }

                    mat4f_t uvw_trans;
                    subset_t scene_corrs, model_corrs;
                    typename Proj::handle_t proj;
                    std::tie(uvw_trans, scene_corrs, model_corrs, proj) = max_icp_iterations ? icp_(m, match, max_icp_iterations, dist_thres) : match;

                    if constexpr (!detail::samples_on_all && detail::final_corrs_on_all) {
                        //pdebug("before: {}", scene_corrs_.size());
                        project_(m, proj, uvw_trans, dist_thres, true);
                        //pdebug("after: {}", scene_corrs_.size());
                    }

                    println("accept at {} of {} points", scene_corrs.size(), min_points);

                    // transform is good enough
                    //pdebug("removed {} indices from index list", scene_corrs.size());
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
    find(model<Proj, Point>& m, float dist_thres, uint32_t early_out_threshold, const sample_parameters& params, const subset_t& sample_candidates, subset_t* firsts) {
        uint32_t n_model = m.point_count();
        uint32_t n_scene = sample_candidates.size();

        float lower = m.diameter() * params.min_diameter_factor;
        float upper = m.diameter() * params.max_diameter_factor;
        lower *= lower;

        std::mt19937 rng;
        uint64_t seed = 13;
        if (!detail::deterministic) {
            auto now = std::chrono::system_clock::now();
            seed = now.time_since_epoch().count();
        }
        rng.seed(seed);

        uint32_t best_score = 0;
        mat4f_t  best_uvw_trans = mat4f_t::Identity();
        typename Proj::handle_t best_projector = nullptr;
        uint64_t valid_query_count = 0; //, sample_count = 0;

        double outer_prob = static_cast<double>(n_model) / n_scene;
        uint64_t outer_bound = static_cast<uint64_t>(std::log(1.0 - detail::match_probability) / std::log(1.0 - outer_prob));
        outer_bound = std::max(outer_bound, detail::min_sample_count);
        //outer_bound = std::max(outer_bound, static_cast<uint64_t>(n_model));
        outer_bound = std::min(outer_bound, static_cast<uint64_t>(n_scene));

        if (detail::force_exhaustive) {
            outer_bound = n_scene;
        }

        auto outer_subset = vw::sample(sample_candidates, outer_bound, rng) | ranges::to_vector;
        outer_subset |= act::shuffle(rng);

        //pdebug("outer bound: {},  p: {} / {} = {}", outer_bound, n_model, n_scene, outer_prob);
        //uint32_t progress = 0;
        //
        //pdebug("outer_bound: {} in {}", outer_bound, n_scene);
        //progress_bar pbar("Performing RanSaC:  ");

        uint32_t sample_count = 0;
        //uint32_t progress = 0;
        for (auto i : outer_subset) {
            if (mask_[i]) continue;
            const Point& p1 = cloud_->points[i];

            if (++sample_count < 5) pdebug("i: {}", i);

            auto nn = cloud_->radius_search_inclusive(upper, p1).first;
            if (nn.empty()) continue;

            double prob = static_cast<double>(n_model) / nn.size();
            uint64_t inner_bound = static_cast<uint64_t>(-std::log(1.0 - detail::match_probability) / prob);
            inner_bound = std::max(inner_bound, detail::min_sample_count);
            inner_bound = std::min(inner_bound, nn.size());
            if (detail::force_exhaustive) {
                inner_bound = nn.size();
            }
            inner_bound *= inner_bound;
            //pdebug("progress: {} / {} (inner_bound^2: {})", ++progress, outer_bound, inner_bound);
            //pdebug("inner bound: {},  p: {} / {} = {}", inner_bound, n_model, nn.size(), prob);

            typename Proj::handle_t proj;
            std::optional<vec3f_t> uvw_s_i, uvw_s_j, uvw_s_k;
            if constexpr (Proj::sample_count == 1) {
                proj = traits_t::init_from_samples(m.projector(), std::make_tuple(p1));
                if (!proj) continue;
            }

            std::vector<int> inner0(nn.begin(), nn.end());
            std::vector<int> inner1(nn.begin(), nn.end());
            inner0 |= act::shuffle(rng);
            inner1 |= act::shuffle(rng);

            if (firsts) firsts->push_back(i);

            uint32_t valid_samples = 0;
            for (auto && [j, k] : vw::cartesian_product(inner0, inner1)) {
                bool debug = false; //i==54 && j ==313;

                if (mask_[j] || mask_[k] || i == static_cast<uint32_t>(j) || i == static_cast<uint32_t>(k) || j == k) continue;
                const Point& p2 = cloud_->points[j];
                const Point& p3 = cloud_->points[k];

                vec3f_t d0 = p2.getVector3fMap() - p1.getVector3fMap();
                if (d0.squaredNorm() < lower) continue;
                d0.normalize();
                vec3f_t d1 = p3.getVector3fMap() - p1.getVector3fMap();
                if (d1.squaredNorm() < lower) continue;
                d1.normalize();
                if (fabs(d0.dot(d1)) > 0.8f) continue;

                if constexpr (Proj::sample_count == 2) {
                    proj = traits_t::init_from_samples(m.projector(), std::make_tuple(p1, p2));
                }
                if constexpr (Proj::sample_count == 3) {
                    proj = traits_t::init_from_samples(m.projector(), std::make_tuple(p1, p2, p3));
                }
                if (!proj) {
                    continue;
                }

                uvw_s_i = Proj::project(proj, p1.getVector3fMap());
                uvw_s_j = Proj::project(proj, p2.getVector3fMap());
                uvw_s_k = Proj::project(proj, p3.getVector3fMap());

                if (!uvw_s_i) {
                    continue;
                }
                if (!uvw_s_j) {
                    continue;
                }
                if (!uvw_s_k) {
                    continue;
                }

                auto f = feature<Proj, Point>(proj, p1, p2, p3);
                if (!f) continue;

                if (!valid<Proj, Point>(proj, *f, m.feature_bounds())) {
                    continue;
                }

                auto && [q_first, q_last] = m.query(*f, debug);
                if (q_first != q_last) {
                    //pdebug("query: {}", std::distance(q_first, q_last));
                    ++valid_query_count;
                    ++valid_samples;
                }

                uint32_t query = 0;
                for (auto it = q_first; it != q_last; ++it) {
                    if (detail::query_limit > 0 && (++query) > detail::query_limit) break;
                    auto && [m_i, m_j, m_k] = it->second;
                    std::optional<vec3f_t> uvw_m_i = Proj::project(m.projector(), m.cloud()->points[m_i].getVector3fMap());
                    std::optional<vec3f_t> uvw_m_j = Proj::project(m.projector(), m.cloud()->points[m_j].getVector3fMap());
                    std::optional<vec3f_t> uvw_m_k = Proj::project(m.projector(), m.cloud()->points[m_k].getVector3fMap());
                    if (!uvw_m_i || !uvw_m_j || !uvw_m_k) {
                        continue;
                    }

                    std::optional<mat4f_t> uvw_trans = uvw_map_(*uvw_s_i, *uvw_s_j, *uvw_s_k, *uvw_m_i, *uvw_m_j, *uvw_m_k, debug);

                    if (!uvw_trans) {
                        continue;
                    }

                    if constexpr (scale_invariant) {
                        float det = uvw_trans->topLeftCorner<3,3>(0,0).determinant();
                        if (det < 0.97f || det > 1.03f) continue;
                    }

                    project_(m, proj, *uvw_trans, dist_thres);
                    uint32_t score = scene_corrs_.size();
                    //if (score) {
                        //scene_corrs_[0] = i;
                    //}
                    if (score > best_score) {
                        best_uvw_trans = *uvw_trans;
                        best_score = score;
                        best_projector = proj;

                        if (detail::early_out && best_score >= early_out_threshold) {
                            //pbar.finish();
                            fmt::print("early out at: {} points\n", best_score);
                            return {*uvw_trans, scene_corrs_, model_corrs_, proj};
                        }
                    }
                }

                if (valid_samples > inner_bound) {
                    break;
                }
            }

            //pbar.poll(++progress, outer_bound);
        }

        //pbar.finish();

        if (best_projector) {
            project_(m, best_projector, best_uvw_trans, dist_thres);
        } else {
            scene_corrs_.clear();
        }
        return {best_uvw_trans, scene_corrs_, model_corrs_, best_projector};
    }

    typename cloud_t::Ptr
    instantiate(model<Proj, Point>& m, const match_t& match, bool instantiate_all = false) {
        mat4f_t t;
        subset_t model_corrs;
        typename Proj::handle_t proj;
        std::tie(t, std::ignore, model_corrs, proj) = match;

        if (instantiate_all) {
            model_corrs = vw::ints(0u, static_cast<uint32_t>(m.cloud()->size())) | ranges::to_vector;
        }

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

    bool
    considered_correspondence_(const Point& pnt, bool allow_all = false) {
        return allow_all || detail::samples_on_all || tangent(pnt).norm() > 0.7f;
    }

    void
    project_(model<Proj, Point>& m, typename Proj::const_handle_t scene_proj, const mat4f_t& uvw_trans, float dist_thres, bool allow_all = false, bool debug = false) {
        uint32_t concurrency = std::thread::hardware_concurrency() - 1;
        uint32_t batch_size = cloud_->size() / concurrency;
        if (cloud_->size() % concurrency > 0) {
            ++batch_size;
        }

        float thres = dist_thres * m.uvw_resolution();

        mat3f_t uvw_trans_tgt = uvw_trans.topLeftCorner<3,3>();
        mat3f_t uvw_trans_nrm = uvw_trans_tgt.inverse().transpose();

        std::vector<std::future<std::tuple<subset_t, subset_t/*, double*/>>> futures;
        for (auto chunk : vw::chunk(vw::ints(0ul, cloud_->size()), batch_size)) {
            futures.push_back(std::async(std::launch::async, [&, chunk] () {
                subset_t sub_scene_corrs, sub_model_corrs;
                //uint32_t count_cons = 0, count_cons_poss = 0;
                for (uint32_t idx : chunk) {
                    if (mask_[idx]) continue;
                    const Point& pnt = cloud_->points[idx];
                    //++count_cons_poss;
                    if (!considered_correspondence_(pnt, allow_all)) {
                        continue;
                    }
                    //++count_cons;
                    std::optional<vec3f_t> uvw = Proj::project(scene_proj, pnt.getVector3fMap());
                    if (!uvw) {
                        continue;
                    }

                    bool use_tangent = tangent(pnt).norm() > 0.7f;
                    vec3f_t uvw_ref = use_tangent ? Proj::tangent(scene_proj, pnt) : Proj::normal(scene_proj, pnt);

                    *uvw = (uvw_trans * uvw->homogeneous()).head(3);
                    std::optional<uint32_t> uvw_n = m.voxel_query(*uvw);
                    if (!uvw_n) {
                        continue;
                    }
                    uvw_ref = use_tangent ? uvw_trans_tgt * uvw_ref : uvw_trans_nrm * uvw_ref;
                    uvw_ref.normalize();
                    float dist = Proj::intrinsic_distance(m.projector(), *uvw, m.uvw_cloud()->points[uvw_n.value()].getVector3fMap());
                    vec3f_t uvw_ref_n = use_tangent ? vec3f_t(tangent(m.uvw_cloud()->points[uvw_n.value()])) : vec3f_t(m.uvw_cloud()->points[uvw_n.value()].getNormalVector3fMap());
                    bool coll = fabs(uvw_ref.dot(uvw_ref_n)) > 0.7f;
                    if (dist <= thres && (allow_all || coll)) {
                        sub_scene_corrs.push_back(idx);
                        sub_model_corrs.push_back(uvw_n.value());
                    }
                }
                return std::tuple<subset_t, subset_t/*, double*/>(sub_scene_corrs, sub_model_corrs/*, static_cast<double>(count_cons) / count_cons_poss*/);
            }));
        }

        scene_corrs_.clear();
        model_corrs_.clear();
        for (auto& fut : futures) {
            auto && [subs, subm/*, ratio*/] = fut.get();
            scene_corrs_.insert(scene_corrs_.end(), subs.begin(), subs.end());
            model_corrs_.insert(model_corrs_.end(), subm.begin(), subm.end());
        }
    }

    //mat3f_t gram_schmidt_(const mat3f_t& base, bool col_space = true) {
        //mat3f_t m = base;

        //if (col_space) {
            //m.col(1) -= m.col(1).dot(m.col(0)) * m.col(0);
            //m.col(1).normalize();
            //m.col(2) -= m.col(2).dot(m.col(0)) * m.col(0);
            //m.col(2) -= m.col(2).dot(m.col(1)) * m.col(1);
            //m.col(2).normalize();
        //} else {
            //m.row(1) -= m.row(1).dot(m.row(0)) * m.row(0);
            //m.row(1).normalize();
            //m.row(2) -= m.row(2).dot(m.row(0)) * m.row(0);
            //m.row(2) -= m.row(2).dot(m.row(1)) * m.row(1);
            //m.row(2).normalize();
        //}

        //return m;
    //}

    std::optional<mat4f_t> uvw_map_(const vec3f_t& src_i, const vec3f_t& src_j, const vec3f_t& src_k, const vec3f_t& tgt_i, const vec3f_t& tgt_j, const vec3f_t& tgt_k, bool debug = false) {
        vec3f_t o_a = src_i;
        vec3f_t o_b = tgt_i;

        vec3f_t u_a = src_j - o_a;
        vec3f_t v_a = src_k - o_a;
        vec3f_t u_b = tgt_j - o_b;
        vec3f_t v_b = tgt_k - o_b;

        if constexpr (!scale_invariant) {
            u_a.normalize();
            u_b.normalize();
            v_a -= v_a.dot(u_a) * u_a;
            v_b -= v_b.dot(u_b) * u_b;
            v_a.normalize();
            v_b.normalize();
        }

        mat3f_t base_a, base_b;
        base_a <<
            u_a, v_a, u_a.cross(v_a).normalized();
        base_b <<
            u_b, v_b, u_b.cross(v_b).normalized();

        mat4f_t t = mat4f_t::Identity();
        t.topLeftCorner<3,3>(0,0) = base_b * base_a.inverse();
        t.block<3,1>(0,3) = o_b - t.topLeftCorner<3,3>(0,0) * o_a;

        return t;
    }

    /*
    mat4f_t uvw_map_(const vec3f_t& src_i, const vec3f_t& src_j, const vec3f_t& src_k, const vec3f_t& tgt_i, const vec3f_t& tgt_j, const vec3f_t& tgt_k, bool debug = false) {
        // build scaled (x,y,dz) and orthonormal (dx,dy,dz) bases
        vec3f_t src_x = src_j - src_i;
        vec3f_t src_y = src_k - src_i;
        vec3f_t src_dx = src_x.normalized();
        src_y -= src_y.dot(src_dx) * src_dx;
        float src_ny = src_y.norm();
        vec3f_t src_dy = src_y / src_ny;
        vec3f_t src_dz = src_dx.cross(src_dy).normalized();
        vec3f_t tgt_x = tgt_j - tgt_i;
        vec3f_t tgt_y = tgt_k - tgt_i;
        vec3f_t tgt_dx = tgt_x.normalized();
        tgt_y -= tgt_y.dot(tgt_dx) * tgt_dx;
        float tgt_ny = tgt_y.norm();
        vec3f_t tgt_dy = tgt_y / tgt_ny;
        vec3f_t tgt_dz = tgt_dx.cross(tgt_dy).normalized();

        float s_x = tgt_x.norm() / src_x.norm();
        float s_y = tgt_ny / src_ny;

        mat3f_t src_base, tgt_base;
        src_base <<
            src_dx, src_dy, src_dz;
        tgt_base <<
            tgt_dx, tgt_dy, tgt_dz;
        mat3f_t scale = vec3f_t(s_x, s_y, 1.f).asDiagonal();

        mat4f_t t = mat4f_t::Identity();
        t.topLeftCorner<3,3>() = tgt_base * scale * src_base.transpose();
        t.block<3,1>(0,3) = tgt_i - t.topLeftCorner<3,3>() * src_i;
        return t;
    }
    */

    match_t
    icp_(model<Proj, Point>& m, const match_t& match, uint32_t max_iterations, float dist_thres) {
        if (max_iterations == 0) return match;

        mat4f_t best_trans;
        typename Proj::handle_t proj;
        std::tie(best_trans, std::ignore, std::ignore, proj) = match;

        // ICP is performed in uvw space
        typename cloud_t::Ptr uvw_scene = cloud_t::empty();
        for (auto idx : vw::ints(0ul, cloud_->size())) {
            if (mask_[idx]) continue;
            const Point& pnt = cloud_->points[idx];
            std::optional<vec3f_t> uvw = Proj::project(proj, pnt.getVector3fMap());
            Point uvw_point;
            uvw_point.getVector4fMap() = best_trans * uvw->homogeneous();
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
        project_(m, proj, best_trans, dist_thres);

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
scene<Proj, Point>::find_all(model<Proj, Point>& m, float dist_thres, float octree_diameter_factor, float model_match_factor, float early_out_factor, const sample_parameters& sample_params, uint32_t max_icp_iterations, bool only_first, subset_t* firsts) {
    return impl_->find_all(m, dist_thres, octree_diameter_factor, model_match_factor, early_out_factor, sample_params, max_icp_iterations, only_first, firsts);
}

template <typename Proj, typename Point>
inline typename scene<Proj, Point>::match_t
scene<Proj, Point>::find(model<Proj, Point>& m, float dist_thres, uint32_t early_out_threshold, const sample_parameters& sample_params, const subset_t& sample_candidates) {
    return impl_->find(m, dist_thres, early_out_threshold, sample_params, sample_candidates, nullptr);
}

template <typename Proj, typename Point>
inline typename scene<Proj, Point>::cloud_t::Ptr
scene<Proj, Point>::instantiate(model<Proj, Point>& m, const match_t& match, bool instantiate_all) {
    return impl_->instantiate(m, match, instantiate_all);
}

template <typename Proj, typename Point>
inline typename scene<Proj, Point>::cloud_t::ConstPtr
scene<Proj, Point>::cloud() const {
    return impl_->cloud();
}

}  // namespace triplet_match
