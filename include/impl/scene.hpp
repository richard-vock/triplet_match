#include <random>
#include <chrono>
#include "timer.hpp"
#include <pcl/registration/icp.h>
#include <progress_bar>

namespace triplet_match {

namespace detail {

//constexpr uint32_t random_kernel_size = 100;
constexpr bool early_out = true;
constexpr bool deterministic = false;
constexpr double match_probability = 0.999;
constexpr uint64_t min_sample_count = 10ull;
constexpr float corr_dist_factor = 3.0;
constexpr bool allow_scale = false;
constexpr int threads_per_block = 512;
constexpr int query_limit = 200; //disabled: -1;
constexpr bool force_exhaustive = false;
constexpr bool samples_on_all = true;
constexpr bool final_corrs_on_all = true;
constexpr uint32_t curvature_k = 30;
constexpr float test_ratio = 0.1;
constexpr bool early_drop = true;
constexpr bool naive_corrs = false;
constexpr bool weighted_distance = false;
constexpr bool negative_scores = false;

}  // namespace detail

template <typename Point>
struct scene<Point>::impl {
    impl(typename cloud_t::Ptr cloud) : cloud_(cloud) {}

    ~impl() {
    }

    std::vector<match_t>
    find_all_parallel(model<Point>& m, float dist_thres, float model_match_factor, float early_out_factor, const sample_parameters& sample_params, uint32_t max_icp_iterations) {
        mask_ = std::vector<int>(cloud_->size(), 0);
        tangent_mask_ = std::vector<int>(cloud_->size(), 0);
        fp_mask_ = std::vector<int>(cloud_->size(), 0);

        uint32_t possible_scene = 0, n_scene = 0;
        cinfo_.resize(cloud_->size());
        subset_t tangent_indices;
        for (auto && [idx, pnt] : vw::zip(vw::ints(0ul), cloud_->points)) {
            cinfo_[idx] = cloud_->curvature(detail::curvature_k, idx);
            if (tangent(pnt).norm() > 0.7f && (cinfo_[idx].pc_min / cinfo_[idx].pc_max) < 0.2f) {
                tangent_indices.push_back(idx);
                tangent_mask_[idx] = 1;
                ++n_scene;
            }
            if (considered_correspondence_(pnt)) {
                ++possible_scene;
            }
        }
        //uint32_t considered = 0;

        std::vector<match_t> results;

        uint32_t possible_model = 0;
        for (const auto& pnt : *m.cloud()) {
            if (considered_correspondence_(pnt)) {
                ++possible_model;
            }
        }

        while(true) {
            std::vector<match_t> matches = find_parallel(
                m,
                tangent_indices,
                n_scene,
                dist_thres,
                max_icp_iterations,
                model_match_factor,
                early_out_factor * possible_model,
                sample_params);

            if (matches.empty()) {
                break;
            }

            println("accepted {} transformations", matches.size());

            for (auto && [t, sc, mc, score] : matches) {
                for (auto idx : sc) {
                    mask_[idx] = 1;
                }
                //considered += sc.size();
                results.push_back(match_t{t.inverse(), sc, mc, score});
            }

        }

        return results;
    }

    match_t
    finish_find(model<Point>& m, const mat4f_t& trans, float accept_prob, float dist_thres) {
        std::vector<int> all = vw::ints(0, static_cast<int>(cloud_->size())) | ranges::to_vector;
        uint32_t saved = 0;
        auto sc = project_(saved, m, all, trans, accept_prob, dist_thres);
        return { trans, std::get<0>(sc), std::get<1>(sc), std::get<2>(sc) };
    }

    std::vector<match_t>
    find_parallel(model<Point>& m, const subset_t& tangent_indices, uint32_t n_scene, float dist_thres, uint32_t max_icp_iterations, float accept_prob, uint32_t early_out_threshold, const sample_parameters& params) {
        constexpr float edge_factor = 1.f;
        uint32_t n_model = m.point_count();
        uint32_t n_model_all = m.cloud()->size();

        uint32_t min_points = (edge_factor * accept_prob) * n_model;
        uint32_t min_points_all = accept_prob * n_model_all;

        float lower = m.diameter() * params.min_diameter_factor;
        float upper = m.diameter() * params.max_diameter_factor;
        lower *= lower;
        upper *= upper;

        std::mt19937 rng;
        uint64_t seed = 13;
        if (!detail::deterministic) {
            auto now = std::chrono::system_clock::now();
            seed = now.time_since_epoch().count();
        }
        rng.seed(seed);


        subset_t left = vw::filter(tangent_indices, [&] (uint32_t idx) { return !fp_mask_[idx] && !mask_[idx]; }) | ranges::to_vector;

        double outer_prob = static_cast<double>(n_model) / left.size();
        uint64_t outer_bound = static_cast<uint64_t>(std::log(1.0 - detail::match_probability) / std::log(1.0 - outer_prob));
        outer_bound = std::max(outer_bound, detail::min_sample_count);
        //outer_bound = std::max(outer_bound, static_cast<uint64_t>(n_model));
        outer_bound = std::min(outer_bound, static_cast<uint64_t>(n_scene));

        if (detail::force_exhaustive) {
            outer_bound = n_scene;
        }

        auto outer_subset = vw::sample(left, outer_bound, rng) | ranges::to_vector;
        outer_subset |= act::shuffle(rng);

        uint32_t concurrency = std::thread::hardware_concurrency() - 1;
        uint32_t batch_size = outer_bound / concurrency;
        if (outer_bound % concurrency > 0) {
            ++batch_size;
        }

        pdebug("outer bound: {}", outer_bound);

        std::vector<std::future<std::vector<match_t>>> futs;
        std::mutex mut;
        for (auto chunk : vw::chunk(outer_subset, batch_size)) {
            futs.push_back(std::async(std::launch::async, [&, chunk] () {
                auto && [matches, cons] = find_in_subset(m, chunk, lower, upper, dist_thres, accept_prob, early_out_threshold, params.force_up);
                mut.lock();
                for (auto idx : vw::take_exactly(chunk, cons)) {
                    fp_mask_[idx] = 1;
                }
                mut.unlock();
                return matches;
            }));
        }

        std::vector<match_t> cands;
        for (auto& fut : futs) {
            for (auto&& match : fut.get()) {
                auto icp_match = icp_(m, match, max_icp_iterations, dist_thres, accept_prob);
                //mat4f_t transform = icp_match.transform.inverse();
                //std::vector<int8_t> visible_scene(m.cloud()->size(), 0), visible;
                //for (const auto& origin : origins_) {
                    //rayc->cast(origin, transform, visible);
                    //for (auto idx : vw::ints(0ul, m.cloud()->size())) {
                        //visible_scene[idx] |= visible[idx];
                    //}
                //}
                //uint64_t possible = std::accumulate(visible_scene.begin(), visible_scene.end(), 0ul);

                //float cast_score = 0.f;
                //if (possible) {
                    //cast_score = static_cast<float>(icp_match.scene_corrs.size()) / possible;
                    ////float cmp_score = static_cast<float>(icp_match.signed_score) / possible;
                    ////pdebug("cast score: {} (signed: {})", cast_score, cmp_score);
                //}
                cands.push_back(icp_match);
                //std::cout << "Cand score: " << std::get<1>(match).size() << ", trans: " << std::get<0>(match) << std::endl << std::endl;
            }
            //cands.push_back(icp_(m, fut.get(), max_icp_iterations, dist_thres, accept_prob));
        }
        std::sort(cands.begin(), cands.end(), [&] (const auto& a, const auto& b) {
            return a.scene_corrs.size() > b.scene_corrs.size();
        });
        //bool first = true;
        std::vector<match_t> result;
        std::set<uint32_t> used_points;
        //double correction = 1.0 / (1.0 * m.cloud()->size());
        for (const auto& cand : cands) {
            auto && [t, is, im, sscore] = cand;
            double signed_score = sscore;

            subset_t isres, imres;

            std::set<uint32_t> used_points_copy = used_points;

            for (uint32_t i = 0; i < is.size(); ++i) {
                if (used_points.find(is[i]) == used_points.end()) {
                    isres.push_back(is[i]);
                    imres.push_back(im[i]);
                } else {
                    //signed_score -= correction;
                }
                used_points.insert(is[i]);
            }

            uint32_t score = 0, score_all = isres.size();
            for (uint32_t idx : isres) {
                if (tangent_mask_[idx]) {
                    ++score;
                }
            }

            //uint32_t score_all = signed_score < 0 ? 0u : static_cast<uint32_t>(signed_score);

            if (score < min_points || score_all < min_points_all) { //score_all < min_points_all) {
                used_points = used_points_copy;
                continue;
            }

            //pdebug("{} of {}", signed_score, static_cast<double>(score_all) / m.cloud()->size());

            result.push_back(match_t{t, isres, imres, signed_score});
            //used_points.insert(used_points.end(), residual.begin(), residual.end());
            //std::sort(used_points.begin(), used_points.end());
        }

        return result;
    }

    std::pair<std::vector<match_t>, uint32_t>
    find_in_subset(model<Point>& m, const subset_t& firsts, float lower, float upper, float dist_thres, float accept_prob, uint32_t early_out_threshold, bool force_up) {
        std::mt19937 rng;
        uint64_t seed = 13;
        if (!detail::deterministic) {
            auto now = std::chrono::system_clock::now();
            seed = now.time_since_epoch().count();
        }
        rng.seed(seed);

        mat4f_t  best_t = mat4f_t::Identity();
        double best_score = 0;

        std::deque<mat4f_t> best_ts;
	    bool out = false;
        uint32_t cons = 0;

        double avg_time = 0.0;
        double avg_subset = 0.0;
        double avg_saved = 0.0;
        uint32_t avg_count = 0;

        auto before_first = std::chrono::system_clock::now();

        for (auto i : firsts) {
            if (out) break;
            if (fp_mask_[i] || mask_[i]) continue;
            //fp_mask_[i] = 1;
            const Point& p1 = cloud_->points[i];
            ++cons;

            auto nn = cloud_->radius_search_inclusive(m.diameter(), p1).first;
            if (nn.empty()) continue;

            double prob = static_cast<double>(m.cloud()->size()) / nn.size();
            uint64_t inner_bound = static_cast<uint64_t>(-std::log(1.0 - detail::match_probability) / prob);
            inner_bound = std::max(inner_bound, detail::min_sample_count);
            inner_bound = std::min(inner_bound, nn.size());
            if (detail::force_exhaustive) {
                inner_bound = nn.size();
            }

            std::vector<int> inner0(nn.begin(), nn.end());
            inner0 |= act::shuffle(rng);

            uint32_t valid_samples = 0;
            for (auto j : inner0) {
                if (out) break;
                if (!tangent_mask_[j] || mask_[j] || i == static_cast<uint32_t>(j)) continue;
                const Point& p2 = cloud_->points[j];

                vec3f_t d0 = p2.getVector3fMap() - p1.getVector3fMap();
                float sqn0 = d0.squaredNorm();
                d0.normalize();
                if (sqn0 < lower || sqn0 > upper) continue;
                if (1.f - fabs(d0.dot(tangent(p1))) < 0.01f) continue;

                auto f = feature<Point>(p1, p2, cinfo_[i], cinfo_[j]);
                if (!f || !valid<Point>(*f, m.feature_bounds())) {
                    continue;
                }

                auto && [q_first, q_last] = m.query(*f);
                ++valid_samples;

                uint32_t query = 0;
                for (auto it = q_first; it != q_last; ++it) {
                    if (out) break;
                    if (detail::query_limit > 0 && (++query) > detail::query_limit) break;
                    auto && [m_i, m_j] = it->second;
                    vec3f_t p_m_i = m.cloud()->points[m_i].getVector3fMap();
                    vec3f_t p_m_j = m.cloud()->points[m_j].getVector3fMap();

                    mat4f_t t = base_transform_(p1.getVector3fMap(), p2.getVector3fMap(), tangent(p1), p_m_i, p_m_j, tangent(m.cloud()->points[m_i]));

                    if (force_up && fabs(1.f - t(2, 2)) > 0.01f) {
                        continue;
                    }

                    auto before = std::chrono::system_clock::now();
                    avg_subset += (static_cast<double>(nn.size()) - avg_subset) / ++avg_count;
                    uint32_t saved = 0;
                    subset_t scene_corrs;
                    double score;
                    std::tie(scene_corrs, std::ignore, score) = project_(saved, m, nn, t, accept_prob, dist_thres, true);
                    avg_saved += (static_cast<double>(saved) - avg_saved) / avg_count;
                    auto after = std::chrono::system_clock::now();
                    avg_time += (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(after - before).count()) - avg_time) / avg_count;;
                    if (score > best_score) {
                        if (best_ts.empty() && scene_corrs.size() > accept_prob * m.cloud()->size()) {
                            auto after_first = std::chrono::system_clock::now();
                            auto first_us = std::chrono::duration_cast<std::chrono::microseconds>(after_first - before_first).count();
                            pdebug("First sufficient after {}us", first_us);
                            best_t = t;
                            best_score = score;
                            best_ts.push_front(best_t);
                        }


                        if (detail::early_out && best_score >= early_out_threshold) {
                            auto after_first = std::chrono::system_clock::now();
                            auto first_us = std::chrono::duration_cast<std::chrono::microseconds>(after_first - before_first).count();
                            pdebug("Early out after {}us", first_us);
                            out = true;
                        }
                    }
                }

                if (valid_samples > inner_bound) {
                    break;
                }
            }
        }

        //pdebug("best score: {}", best_score);
        if (avg_subset) {
            pdebug("average project time: {:.1f}us by saving {} of {} ({:.2f}%%)", avg_time, avg_saved, avg_subset, 100.0*avg_saved/avg_subset);
        }

        std::vector<match_t> matches;
        for (auto best_ts_it = best_ts.begin(); best_ts_it != best_ts.end() && matches.size() < 5; ++best_ts_it) {
            matches.push_back(finish_find(m, *best_ts_it, accept_prob, dist_thres));
        }

        return {matches, cons};
    }

    match_t
    icp_(model<Point>& m, const match_t& match, uint32_t max_iterations, float dist_thres, float accept_prob) {
        if (max_iterations == 0) return match;

        match_t best_proj = finish_find(m, match.transform, accept_prob, 2*dist_thres);

        if (best_proj.scene_corrs.size() != best_proj.model_corrs.size()) {
            std::cout << "Scene corrs size != model corrs size!" << std::endl;
        }

        uint32_t iter = 0;
        while (true) {
            if (best_proj.scene_corrs.size() < 3) {
                return best_proj;
            }

            Eigen::Matrix<float, 3, Eigen::Dynamic> mdl(3, best_proj.model_corrs.size());
            Eigen::Matrix<float, 3, Eigen::Dynamic> scn(3, best_proj.scene_corrs.size());

            for (int i = 0; i < best_proj.scene_corrs.size(); ++i) {
                mdl.block<3,1>(0, i) = m.cloud()->native()->points[best_proj.model_corrs[i]].getVector3fMap();
                scn.block<3,1>(0, i) = cloud_->native()->points[best_proj.scene_corrs[i]].getVector3fMap();
            }

            mat4f_t trans = Eigen::umeyama(scn, mdl, false);

            match_t proj = finish_find(m, trans, accept_prob, 2*dist_thres);
            if (proj.scene_corrs.size() < best_proj.scene_corrs.size()) {
                return best_proj;
            }
            best_proj = proj;
            if (++iter == max_iterations) {
                return best_proj;
            }
        }
    }

    double
    normalize_score(model<Point>& m, double score) {
        return score / m.cloud()->size();
    }

    std::tuple<subset_t, subset_t, double>
    project_(uint32_t& saved, model<Point>& m, const std::vector<int>& subset, const mat4f_t& t, float accept_prob, float dist_thres, bool early_out = false, bool allow_all = false) {
        float thres = dist_thres * m.cloud()->resolution();

        mat3f_t t_tgt = t.topLeftCorner<3,3>();
        //mat3f_t t_nrm = t_tgt.inverse().transpose();
        subset_t scene_corrs;
        subset_t model_corrs;

        uint32_t tried = 0;
        int next_test = 0;
        constexpr float step_size = 0.05f;
        std::vector<uint32_t> tests(static_cast<uint32_t>(1.f / step_size) - 2);
        for (uint32_t i = 0; i < tests.size(); ++i) {
            tests[i] = step_size * (i + 1) * subset.size();
        }

        saved = 0;
        double signed_score = 0;
        double exp_factor = -1.0 / (2.0 * thres);

        for (int idx : subset) {
            ++tried;
            if (mask_[idx]) continue;
            const Point& pnt = cloud_->points[idx];
            if (!considered_correspondence_(pnt, allow_all)) {
                continue;
            }
            vec4f_t pos = pnt.getVector3fMap().homogeneous();

            bool use_tangent = tangent_mask_[idx];
            vec3f_t ref = use_tangent ? tangent(pnt) : vec3f_t(pnt.getNormalVector3fMap());

            pos = t * pos;
            std::optional<typename model<Point>::voxel_data_t> pos_n;
            if constexpr (detail::naive_corrs) {
                Point query_pnt;
                query_pnt.getVector3fMap() = pos.head(3);
                auto mis = m.cloud()->knn_inclusive(voxel_multiplicity, query_pnt).first;
                pos_n = typename model<Point>::voxel_data_t();
                for (int i = 0; i < voxel_multiplicity; ++i) {
                    (*pos_n)[i] = mis[i];
                }
            } else {
                pos_n = m.voxel_query(pos);
            }

            if (!pos_n) {
                continue;
            }
            ref = t_tgt * ref;

            for (int qr = 0; qr < voxel_multiplicity; ++qr) {
                float dist = (pos.head(3) - m.cloud()->points[pos_n.value()[qr]].getVector3fMap()).norm();
                if (dist > thres) {
                    break;
                }
                double weight = detail::weighted_distance ? std::exp(exp_factor * dist * dist) : 1.0;
                vec3f_t ref_n = tangent(m.cloud()->points[pos_n.value()[qr]]);
                bool is_tangent = ref_n.norm() > 0.7f;
                //if (!is_tangent) printf("should not happen");
                if (use_tangent != is_tangent) {
                    if (detail::negative_scores && qr == (voxel_multiplicity - 1)) {
                        signed_score -= weight;
                    }
                    //--signed_score;
                    continue;
                }
                if (!is_tangent) {
                    ref_n = m.cloud()->points[pos_n.value()[qr]].getNormalVector3fMap();
                }

                signed_score += weight * static_cast<double>(fabs(ref.dot(ref_n)));
                //if (coll > 0.7f) {
                    //++signed_score;
                    scene_corrs.push_back(idx);
                    model_corrs.push_back(pos_n.value()[qr]);
                    break;
                //}
            }

            if (detail::early_drop && early_out && next_test < tests.size() && tried >= tests[next_test]) {
                double N = -2.0 - tried;
                double x = -2.0 - subset.size();
                double n = -1.0 - model_corrs.size();
                double tmp = std::sqrt((x*n*(N-x)*(N-n)) / (N-1.0));
                //uint32_t lower = -1.0 - static_cast<uint32_t>((x*n - tmp) / N);
                uint32_t upper = -1.0 - static_cast<uint32_t>((x*n + tmp) / N);

                if (upper < accept_prob * m.cloud()->size()) {
                    saved = subset.size() - tried;
                    return {scene_corrs, model_corrs, signed_score};
                    //debug = next_test;
                }
                ++next_test;
            }
        }

        return {scene_corrs, model_corrs, normalize_score(m, signed_score)};
    }

    typename cloud_t::Ptr
    instantiate(model<Point>& m, const match_t& match, bool instantiate_all = false) {
        mat4f_t t = match.transform;
        subset_t model_corrs;

        if (instantiate_all) {
            model_corrs = vw::ints(0u, static_cast<uint32_t>(m.cloud()->size())) | ranges::to_vector;
        }

        typename cloud_t::Ptr inst = cloud_t::empty();
        for (uint32_t idx : model_corrs) {
            vec3f_t uvw_model = m.cloud()->points[idx].getVector3fMap();
            vec3f_t uvw_scene = (t * uvw_model.homogeneous()).head(3);
            Point inst_pnt;
            inst_pnt.getVector3fMap() = uvw_scene;
            inst->push_back(inst_pnt);
        }

        return inst;
    }

    bool
    considered_correspondence_(const Point& pnt, bool allow_all = false) {
        return allow_all || detail::samples_on_all || tangent(pnt).norm() > 0.7f;
    }

    mat4f_t base_transform_(const vec3f_t& src_i, const vec3f_t& src_j, const vec3f_t& src_t, const vec3f_t& tgt_i, const vec3f_t& tgt_j, const vec3f_t& tgt_t, bool debug = false) {
        vec3f_t o_a = src_i;
        vec3f_t o_b = tgt_i;

        vec3f_t u_a = src_j - o_a;
        vec3f_t u_b = tgt_j - o_b;
        vec3f_t v_a = src_t;
        vec3f_t v_b = tgt_t;

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
        t.topLeftCorner<3,3>() = base_b * base_a.inverse();
        t.block<3,1>(0,3) = o_b - t.topLeftCorner<3,3>() * o_a;

        return t;
    }

    typename cloud_t::ConstPtr cloud() const {
        return cloud_;
    }

    typename cloud_t::Ptr      cloud_;

    std::vector<typename pointcloud<Point>::curvature_info_t> cinfo_;
    subset_t scene_corrs_;
    subset_t model_corrs_;
    std::vector<int> mask_;
    std::vector<int> fp_mask_;
    std::vector<int> tangent_mask_;
};


template <typename Point>
inline
scene<Point>::scene(typename cloud_t::Ptr cloud) : impl_(new impl(cloud)) {
}

template <typename Point>
inline
scene<Point>::~scene() {
}

//template <typename Point>
//inline std::vector<typename scene<Point>::match_t>
//scene<Point>::find_all(model<Point>& m, float dist_thres, float octree_diameter_factor, float model_match_factor, float early_out_factor, const sample_parameters& sample_params, uint32_t max_icp_iterations, bool only_first, subset_t* firsts) {
//    return impl_->find_all(m, dist_thres, octree_diameter_factor, model_match_factor, early_out_factor, sample_params, max_icp_iterations, only_first, firsts);
//}

template <typename Point>
inline std::vector<typename scene<Point>::match_t>
scene<Point>::find_all_parallel(model<Point>& m, float dist_thres, float model_match_factor, float early_out_factor, const sample_parameters& sample_params, uint32_t max_icp_iterations) {
    return impl_->find_all_parallel(m, dist_thres, model_match_factor, early_out_factor, sample_params, max_icp_iterations);
}

//template <typename Point>
//inline typename scene<Point>::match_t
//scene<Point>::find(model<Point>& m, float dist_thres, uint32_t early_out_threshold, const sample_parameters& sample_params, const subset_t& sample_candidates) {
    //return impl_->find(m, dist_thres, early_out_threshold, sample_params, sample_candidates, nullptr);
//}

template <typename Point>
inline typename scene<Point>::cloud_t::Ptr
scene<Point>::instantiate(model<Point>& m, const match_t& match, bool instantiate_all) {
    return impl_->instantiate(m, match, instantiate_all);
}

template <typename Point>
inline typename scene<Point>::cloud_t::ConstPtr
scene<Point>::cloud() const {
    return impl_->cloud();
}

}  // namespace triplet_match
