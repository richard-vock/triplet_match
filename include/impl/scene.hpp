#include <random>
#include <chrono>
#include "timer.hpp"
#include <pcl/registration/icp.h>
#include <progress_bar>
#include <boost/math/special_functions/binomial.hpp>

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

}  // namespace detail

template <typename Point>
struct scene<Point>::impl {
    impl(typename cloud_t::Ptr cloud) : cloud_(cloud) {}

    ~impl() {
    }

    std::vector<match_t>
    find_all(model<Point>& m, float dist_thres, float octree_diameter_factor, float model_match_factor, float early_out_factor, const sample_parameters& sample_params, uint32_t max_icp_iterations, bool only_first, subset_t* firsts) {
        mask_ = std::vector<bool>(cloud_->size(), false);
        tangent_mask_ = std::vector<bool>(cloud_->size(), false);
        fp_mask_ = std::vector<bool>(cloud_->size(), false);

        uint32_t possible_scene = 0, n_scene = 0;
        cinfo_.resize(cloud_->size());
        subset_t tangent_indices;
        for (auto && [idx, pnt] : vw::zip(vw::ints(0ul), cloud_->points)) {
            cinfo_[idx] = cloud_->curvature(detail::curvature_k, idx);
            if (tangent(pnt).norm() > 0.7f && (cinfo_[idx].pc_min / cinfo_[idx].pc_max) < 0.2f) {
                tangent_indices.push_back(idx);
                tangent_mask_[idx] = true;
                ++n_scene;
            }
            if (considered_correspondence_(pnt)) {
                ++possible_scene;
            }
        }
        uint32_t considered = 0;

        std::vector<match_t> results;

        uint32_t possible_model = 0;
        for (const auto& pnt : *m.cloud()) {
            if (considered_correspondence_(pnt)) {
                ++possible_model;
            }
        }

        bool found = false;
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
                tangent_indices,
                n_scene,
                dist_thres,
                model_match_factor,
                early_out_factor * possible_model,
                sample_params);

            //pdebug("rel score: {}", static_cast<double>(std::get<1>(match).size()) / possible_model);
            if (std::get<1>(match).size() < min_points) {
                break;
            }

            mat4f_t t;
            subset_t scene_corrs, model_corrs;
            //std::tie(t, scene_corrs, model_corrs) = max_icp_iterations ? icp_(m, match, max_icp_iterations, dist_thres) : match;
            std::tie(t, scene_corrs) = match;

            //if constexpr (!detail::samples_on_all && detail::final_corrs_on_all) {
                //project_(m, proj, t, dist_thres, true);
            //}

            println("accept at {} of {} points", scene_corrs.size(), min_points);

            for (auto idx : scene_corrs) {
                mask_[idx] = true;
            }
            considered += scene_corrs.size();

            results.push_back({t.inverse(), scene_corrs});
            found = true;
        }
        //std::cout << (results.size() - before) << " transformations found.\n";

        return results;
    }

    std::vector<match_t>
    find_all_parallel(model<Point>& m, float dist_thres, float model_match_factor, float early_out_factor, const sample_parameters& sample_params, uint32_t max_icp_iterations) {
        mask_ = std::vector<bool>(cloud_->size(), false);
        tangent_mask_ = std::vector<bool>(cloud_->size(), false);
        fp_mask_ = std::vector<bool>(cloud_->size(), false);

        uint32_t possible_scene = 0, n_scene = 0;
        cinfo_.resize(cloud_->size());
        subset_t tangent_indices;
        for (auto && [idx, pnt] : vw::zip(vw::ints(0ul), cloud_->points)) {
            cinfo_[idx] = cloud_->curvature(detail::curvature_k, idx);
            if (tangent(pnt).norm() > 0.7f && (cinfo_[idx].pc_min / cinfo_[idx].pc_max) < 0.2f) {
                tangent_indices.push_back(idx);
                tangent_mask_[idx] = true;
                ++n_scene;
            }
            if (considered_correspondence_(pnt)) {
                ++possible_scene;
            }
        }
        uint32_t considered = 0;

        std::vector<match_t> results;

        uint32_t possible_model = 0;
        for (const auto& pnt : *m.cloud()) {
            if (considered_correspondence_(pnt)) {
                ++possible_model;
            }
        }

        while(true) {
            uint32_t min_points = model_match_factor * possible_model;
            // check if remaining points could even surpass the validity threshold
            int32_t possible_left = static_cast<int64_t>(possible_scene) - static_cast<int64_t>(considered);
            if (possible_left < static_cast<int64_t>(min_points)) {
                min_points = model_match_factor * possible_left;
                if (min_points < 0.4f * possible_model) {
                    break;
                }
            }

            std::vector<match_t> matches = find_parallel(
                m,
                tangent_indices,
                min_points,
                n_scene,
                dist_thres,
                model_match_factor,
                early_out_factor * possible_model,
                sample_params);

            if (matches.empty()) {
                break;
            }

            //if constexpr (!detail::samples_on_all && detail::final_corrs_on_all) {
                //project_(m, proj, t, dist_thres, true);
            //}

            println("accepted {} transformations", matches.size());

            for (auto && [t, sc] : matches) {
                for (auto idx : sc) {
                    mask_[idx] = true;
                }
                considered += sc.size();
                results.push_back({t.inverse(), sc});
            }

        }

        return results;
    }

    match_t
    finish_find(model<Point>& m, const mat4f_t& trans, float accept_prob, float dist_thres) {
        std::vector<int> all = vw::ints(0, static_cast<int>(cloud_->size())) | ranges::to_vector;
        subset_t sc = project_(m, all, trans, accept_prob, dist_thres);
        return { trans, sc };
    }

    std::pair<match_t, uint32_t>
    find_in_subset(model<Point>& m, const subset_t& firsts, float lower, float upper, float dist_thres, float accept_prob, uint32_t early_out_threshold) {
        std::mt19937 rng;
        uint64_t seed = 13;
        if (!detail::deterministic) {
            auto now = std::chrono::system_clock::now();
            seed = now.time_since_epoch().count();
        }
        rng.seed(seed);

        mat4f_t  best_t = mat4f_t::Identity();
        uint32_t best_score = 0;

        uint32_t cons = 0;
        for (auto i : firsts) {
            if (fp_mask_[i] || mask_[i]) continue;
            fp_mask_[i] = true;
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
                    if (detail::query_limit > 0 && (++query) > detail::query_limit) break;
                    auto && [m_i, m_j] = it->second;
                    vec3f_t p_m_i = m.cloud()->points[m_i].getVector3fMap();
                    vec3f_t p_m_j = m.cloud()->points[m_j].getVector3fMap();

                    mat4f_t t = base_transform_(p1.getVector3fMap(), p2.getVector3fMap(), tangent(p1), p_m_i, p_m_j, tangent(m.cloud()->points[m_i]));

                    subset_t scene_corrs = project_(m, nn, t, accept_prob, dist_thres, true);
                    //++pruned_possible;
                    uint32_t score = scene_corrs.size();
                    if (score > best_score) {
                        best_t = t;
                        best_score = score;

                        if (detail::early_out && best_score >= early_out_threshold) {
                            //fmt::print("early out at: {} points\n", best_score);
                            //pdebug("pruned ratio: {}", static_cast<double>(pruned) / pruned_possible);
                            return {finish_find(m, t, accept_prob, dist_thres), cons};
                        }
                    }
                }

                if (valid_samples > inner_bound) {
                    break;
                }
            }
        }

        return {finish_find(m, best_t, accept_prob, dist_thres), cons};
    }

    std::vector<match_t>
    find_parallel(model<Point>& m, const subset_t& tangent_indices, uint32_t min_points, uint32_t n_scene, float dist_thres, float accept_prob, uint32_t early_out_threshold, const sample_parameters& params) {
        uint32_t n_model = m.point_count();

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

        double outer_prob = static_cast<double>(n_model) / n_scene;
        uint64_t outer_bound = static_cast<uint64_t>(std::log(1.0 - detail::match_probability) / std::log(1.0 - outer_prob));
        outer_bound = std::max(outer_bound, detail::min_sample_count);
        //outer_bound = std::max(outer_bound, static_cast<uint64_t>(n_model));
        outer_bound = std::min(outer_bound, static_cast<uint64_t>(n_scene));

        if (detail::force_exhaustive) {
            outer_bound = n_scene;
        }

        auto outer_subset = vw::sample(tangent_indices, outer_bound, rng) | ranges::to_vector;
        outer_subset |= act::shuffle(rng);

        uint32_t concurrency = std::thread::hardware_concurrency() - 1;
        uint32_t batch_size = outer_bound / concurrency;
        if (outer_bound % concurrency > 0) {
            ++batch_size;
        }

        pdebug("outer bound: {}", outer_bound);

        std::vector<std::future<match_t>> futs;
        std::mutex mut;
        for (auto chunk : vw::chunk(outer_subset, batch_size)) {
            futs.push_back(std::async(std::launch::async, [&, chunk] () {
                auto && [match, cons] = find_in_subset(m, chunk, lower, upper, dist_thres, accept_prob, early_out_threshold);
                mut.lock();
                uint32_t i = 0;
                for (auto idx : chunk) {
                    if (++i > cons) fp_mask_[idx] = true;
                }
                mut.unlock();
                return match;
            }));
        }

        std::vector<match_t> cands;
        for (auto& fut : futs) {
            cands.push_back(fut.get());
        }
        std::sort(cands.begin(), cands.end(), [&] (const auto& a, const auto& b) {
            return std::get<1>(a).size() > std::get<1>(b).size();
        });
        bool first = true;
        std::vector<match_t> result;
        subset_t used_points;
        for (const auto& cand : cands) {
            auto && [t, is] = cand;
            if (std::exchange(first, false) && is.size() >= min_points) {
                result.push_back({t, is});
                used_points = is;
                continue;
            }

            uint32_t before = is.size();
            subset_t residual;
            std::set_difference(is.begin(), is.end(), used_points.begin(), used_points.end(), std::back_inserter(residual));
            uint32_t after = residual.size();
            pdebug("throwing away {} overlap points", before-after);
            if (residual.size() < min_points) continue;
            result.push_back({t, residual});
            used_points.insert(used_points.end(), residual.begin(), residual.end());
            std::sort(used_points.begin(), used_points.end());
        }

        return result;
    }

    match_t
    find(model<Point>& m, const subset_t& tangent_indices, uint32_t n_scene, float dist_thres, float accept_prob, uint32_t early_out_threshold, const sample_parameters& params) {
        uint32_t n_model = m.point_count();

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

        uint32_t best_score = 0;
        mat4f_t  best_t = mat4f_t::Identity();

        double outer_prob = static_cast<double>(n_model) / n_scene;
        uint64_t outer_bound = static_cast<uint64_t>(std::log(1.0 - detail::match_probability) / std::log(1.0 - outer_prob));
        outer_bound = std::max(outer_bound, detail::min_sample_count);
        //outer_bound = std::max(outer_bound, static_cast<uint64_t>(n_model));
        outer_bound = std::min(outer_bound, static_cast<uint64_t>(n_scene));

        if (detail::force_exhaustive) {
            outer_bound = n_scene;
        }

        auto outer_subset = vw::sample(tangent_indices, outer_bound, rng) | ranges::to_vector;
        outer_subset |= act::shuffle(rng);

        progress_bar pbar("Performing RanSaC:  ");

        uint32_t progress = 0;
        for (auto i : outer_subset) {
            ++progress;
            if (fp_mask_[i] || mask_[i]) continue;
            fp_mask_[i] = true;
            const Point& p1 = cloud_->points[i];

            auto nn = cloud_->radius_search_inclusive(m.diameter(), p1).first;
            if (nn.empty()) continue;

            double prob = static_cast<double>(m.cloud()->size()) / nn.size();
            uint64_t inner_bound = static_cast<uint64_t>(-std::log(1.0 - detail::match_probability) / prob);
            inner_bound = std::max(inner_bound, detail::min_sample_count);
            inner_bound = std::min(inner_bound, nn.size());
            if (detail::force_exhaustive) {
                inner_bound = nn.size();
            }
            //pdebug("progress: {} / {} (inner_bound^2: {})", ++progress, outer_bound, inner_bound);
            //pdebug("inner bound: {},  p: {} / {} = {}", inner_bound, n_model, nn.size(), prob);

            std::vector<int> inner0(nn.begin(), nn.end());
            inner0 |= act::shuffle(rng);

            uint32_t valid_samples = 0;
            for (auto j : inner0) {
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
                if (q_first != q_last) {
                    //pdebug("query results: {}", std::distance(q_first, q_last));
                }
                ++valid_samples;

                uint32_t query = 0;
                for (auto it = q_first; it != q_last; ++it) {
                    if (detail::query_limit > 0 && (++query) > detail::query_limit) break;
                    auto && [m_i, m_j] = it->second;
                    vec3f_t p_m_i = m.cloud()->points[m_i].getVector3fMap();
                    vec3f_t p_m_j = m.cloud()->points[m_j].getVector3fMap();

                    mat4f_t t = base_transform_(p1.getVector3fMap(), p2.getVector3fMap(), tangent(p1), p_m_i, p_m_j, tangent(m.cloud()->points[m_i]));

                    project_(m, nn, t, accept_prob, dist_thres, true);
                    uint32_t score = scene_corrs_.size();
                    if (score > best_score) {
                        best_t = t;
                        best_score = score;

                        if (detail::early_out && best_score >= early_out_threshold) {
                            pbar.finish();
                            fmt::print("early out at: {} points\n", best_score);
                            return finish_find(m, t, accept_prob, dist_thres);
                        }
                    }
                }

                if (valid_samples > inner_bound) {
                    break;
                }
            }

            pbar.poll(progress, outer_bound);
        }

        pbar.finish();

        return finish_find(m, best_t, accept_prob, dist_thres);
    }

    typename cloud_t::Ptr
    instantiate(model<Point>& m, const match_t& match, bool instantiate_all = false) {
        mat4f_t t = std::get<0>(match);
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

    subset_t
    project_(model<Point>& m, const std::vector<int>& subset, const mat4f_t& t, float accept_prob, float dist_thres, bool early_out = false, bool allow_all = false, bool debug = false) {
        //uint32_t concurrency = std::thread::hardware_concurrency() - 1;
        //uint32_t batch_size = cloud_->size() / concurrency;
        //if (cloud_->size() % concurrency > 0) {
            //++batch_size;
        //}

        float thres = dist_thres * m.cloud()->resolution();

        //mat3f_t t_tgt = t.topLeftCorner<3,3>();
        //mat3f_t t_nrm = t_tgt.inverse().transpose();
        subset_t scene_corrs;
        //subset_t model_corrs;

        uint32_t tried = 0;
        uint32_t test_count = static_cast<uint32_t>(detail::test_ratio * subset.size());
        bool tested = !early_out;
        for (int idx : subset) {
            ++tried;
            if (mask_[idx]) continue;
            const Point& pnt = cloud_->points[idx];
            if (!considered_correspondence_(pnt, allow_all)) {
                continue;
            }
            vec4f_t pos = pnt.getVector3fMap().homogeneous();

            //bool use_tangent = tangent_mask_[idx];
            //vec3f_t ref = use_tangent ? tangent(pnt) : vec3f_t(pnt.getNormalVector3fMap());

            pos = t * pos;
            std::optional<uint32_t> pos_n = m.voxel_query(pos);
            if (!pos_n) {
                continue;
            }
            //ref = use_tangent ? t_tgt * ref : t_nrm * ref;
            //ref.normalize();
            float dist = (pos.head(3) - m.cloud()->points[pos_n.value()].getVector3fMap()).norm();
            //vec3f_t ref_n = use_tangent ? vec3f_t(tangent(m.cloud()->points[pos_n.value()])) : vec3f_t(m.cloud()->points[pos_n.value()].getNormalVector3fMap());
            //bool coll = fabs(ref.dot(ref_n)) > 0.7f;
            if (dist <= thres /*&& (allow_all || coll)*/) {
                scene_corrs.push_back(idx);
                //model_corrs.push_back(pos_n.value());
            }

            if (tried >= test_count && !std::exchange(tested, true)) {
                float success = static_cast<float>(scene_corrs.size()) / tried;
                if (success < 0.3f * accept_prob) {
                    return scene_corrs;
                }/* else {
                    pdebug("success: {}", success);
                }*/
            }
        }

        return scene_corrs;
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
        t.topLeftCorner<3,3>(0,0) = base_b * base_a.inverse();
        t.block<3,1>(0,3) = o_b - t.topLeftCorner<3,3>(0,0) * o_a;

        return t;
    }

    /*
    mat4f_t base_transform_(const vec3f_t& src_i, const vec3f_t& src_j, const vec3f_t& src_k, const vec3f_t& tgt_i, const vec3f_t& tgt_j, const vec3f_t& tgt_k, bool debug = false) {
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

    //match_t
    //icp_(model<Point>& m, const match_t& match, uint32_t max_iterations, float dist_thres) {
        //if (max_iterations == 0) return match;

        //mat4f_t best_trans;
        //std::tie(best_trans, std::ignore, std::ignore) = match;

        //// ICP is performed in uvw space
        //typename cloud_t::Ptr uvw_scene = cloud_t::empty();
        //for (auto idx : vw::ints(0ul, cloud_->size())) {
            //if (mask_[idx]) continue;
            //const Point& pnt = cloud_->points[idx];
            //std::optional<vec3f_t> uvw = Proj::project(proj, pnt.getVector3fMap());
            //Point uvw_point;
            //uvw_point.getVector4fMap() = best_trans * uvw->homogeneous();
            //uvw_scene->push_back(uvw_point);
        //}
        //typename cloud_t::Ptr uvw_model = m.uvw_cloud();

        //float max_corr_dist = 3.f * detail::corr_dist_factor * m.uvw_resolution();
        //pcl::IterativeClosestPoint<Point, Point, double> icp;
        //icp.setInputSource(uvw_scene->native());
        //icp.setInputTarget(uvw_model->native());
        //icp.setSearchMethodSource(uvw_scene->kdtree());
        //icp.setSearchMethodTarget(uvw_model->kdtree());
        //icp.setMaximumIterations(max_iterations);
        //icp.setMaxCorrespondenceDistance(max_corr_dist);
        ////double test_before = icp.getFitnessScore(max_corr_dist);
        //icp.align(*uvw_scene->native());
        ////double test_after = icp.getFitnessScore(max_corr_dist);

        //best_trans = icp.getFinalTransformation().template cast<float>() * best_trans;
        //project_(m, proj, best_trans, dist_thres);

        //return {best_trans, scene_corrs_, model_corrs_, proj};
    //}

    typename cloud_t::ConstPtr cloud() const {
        return cloud_;
    }

    typename cloud_t::Ptr      cloud_;

    std::vector<typename pointcloud<Point>::curvature_info_t> cinfo_;
    subset_t scene_corrs_;
    subset_t model_corrs_;
    std::vector<bool> mask_;
    std::vector<bool> fp_mask_;
    std::vector<bool> tangent_mask_;
};


template <typename Point>
inline
scene<Point>::scene(typename cloud_t::Ptr cloud) : impl_(new impl(cloud)) {
}

template <typename Point>
inline
scene<Point>::~scene() {
}

template <typename Point>
inline std::vector<typename scene<Point>::match_t>
scene<Point>::find_all(model<Point>& m, float dist_thres, float octree_diameter_factor, float model_match_factor, float early_out_factor, const sample_parameters& sample_params, uint32_t max_icp_iterations, bool only_first, subset_t* firsts) {
    return impl_->find_all(m, dist_thres, octree_diameter_factor, model_match_factor, early_out_factor, sample_params, max_icp_iterations, only_first, firsts);
}

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
