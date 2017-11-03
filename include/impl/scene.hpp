#include <random>
#include <chrono>
#include "timer.hpp"
#include "gpu_program"
#include <boost/compute/interop/eigen.hpp>

#include "debug.hpp"

namespace triplet_match {

namespace detail {

//constexpr uint32_t random_kernel_size = 100;
constexpr bool early_out = true;
constexpr bool deterministic = true;
constexpr double match_probability = 0.99;
constexpr float max_dist_factor = 1.0;
constexpr bool allow_scale = false;
constexpr int threads_per_block = 512;

template <typename Point>
inline mat3f_t
make_base(typename pcl::PointCloud<Point>::ConstPtr cloud, int i, int j) {
    vec3f_t p0 = cloud->points[i].getVector3fMap();
    vec3f_t p1 = cloud->points[j].getVector3fMap();

    // d
    vec3f_t e0 = (p1 - p0).normalized();
    // tangent
    vec3f_t e1(cloud->points[i].data_c[1], cloud->points[i].data_c[2], cloud->points[i].data_c[3]);

    vec3f_t e2 = e0.cross(e1).normalized();
    e1 = e2.cross(e0).normalized();
    mat3f_t base;
    base << e0, e1, e2;
    return base;
}

inline quatf_t
base_rotation(const mat3f_t& b0, const mat3f_t& b1) {
    quatf_t q1, q2;
    vec3f_t b0x = b0.col(0);
    vec3f_t b0y = b0.col(1);
    vec3f_t b1x = b1.col(0);
    vec3f_t b1y = b1.col(1);
    q1.setFromTwoVectors(b0x, b1x);
    vec3f_t b0y_trans = q1._transformVector(b0y);
    q2.setFromTwoVectors(b0y_trans, b1y);
    return q2 * q1;
}

template <typename Point0, typename Point1>
inline mat4f_t
base_transformation(typename pcl::PointCloud<Point0>::ConstPtr c0, typename pcl::PointCloud<Point1>::ConstPtr c1, uint32_t i0, uint32_t j0, uint32_t i1, uint32_t j1) {
    auto b0 = make_base<Point0>(c0, i0, j0);
    auto b1 = make_base<Point1>(c1, i1, j1);
    mat4f_t r = mat4f_t::Identity();
    mat4f_t t = mat4f_t::Identity();

    // rotation
    r.topLeftCorner<3,3>() = base_rotation(b0, b1).toRotationMatrix();

    // rotation * translate first
    t.block<3,1>(0,3) = -c0->points[i0].getVector3fMap();
    r = r * t;

    // translate second * scale * rotation * translate first
    t.block<3,1>(0,3) = c1->points[i1].getVector3fMap();
    r = t * r;

    return r;
}

constexpr char program_source[] =
    "__kernel void icp_projection(__global const float4* pnts, int n,\n"
    "                             __global const float4* voxel,\n"
    "                             int sx, int sy, int sz,\n"
    "                             __global const float* proj,\n"
    "                             __global const float* guess,\n"
    "                             float max_voxel_dist,\n"
    "                             float max_score_dist,\n"
    "                             __global float4* out_positions,\n"
    "                             __global int* model_indices,\n"
    "                             __global int* scene_indices) {\n"
    "    const uint index = get_global_id(0);\n"
    "    if (index >= n) {\n"
    "        return;\n"
    "    }\n"
    "    float4 pnt = pnts[index];\n"
    "    float4 aligned = (float4)(\n"
    "        guess[0] * pnt.x + guess[4] * pnt.y + guess[ 8] * pnt.z + guess[12] * pnt.w,\n"
    "        guess[1] * pnt.x + guess[5] * pnt.y + guess[ 9] * pnt.z + guess[13] * pnt.w,\n"
    "        guess[2] * pnt.x + guess[6] * pnt.y + guess[10] * pnt.z + guess[14] * pnt.w,\n"
    "        guess[3] * pnt.x + guess[7] * pnt.y + guess[11] * pnt.z + guess[15] * pnt.w\n"
    "    );\n"
    "    int x = (int)(proj[0] * aligned.x + proj[4] * aligned.y + proj[ 8] * aligned.z + proj[12] * aligned.w);\n"
    "    int y = (int)(proj[1] * aligned.x + proj[5] * aligned.y + proj[ 9] * aligned.z + proj[13] * aligned.w);\n"
    "    int z = (int)(proj[2] * aligned.x + proj[6] * aligned.y + proj[10] * aligned.z + proj[14] * aligned.w);\n"
    "    if (x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz) {\n"
    "       int idx = x*sy*sz + y*sz + z;\n"
    "       float dist = length(voxel[idx].xyz - aligned.xyz) / max_voxel_dist;\n"
    "       if (dist < max_score_dist) {\n"
    "           model_indices[index] = idx;\n"
    "           scene_indices[index] = index;\n"
    "           out_positions[index] = (float4)(aligned.x, aligned.y, aligned.z, 1.f);\n"
    "       }\n"
    "    } else {\n"
    "       model_indices[index] = -1;\n"
    "       scene_indices[index] = -1;\n"
    "       out_positions[index] = (float4)(0.f, 0.f, 0.f, 1.f);\n"
    "    }\n"
    "}\n"
    "__kernel void icp_correlation(__global const float4* scene,\n"
    "                              __global const float4* model,\n"
    "                              __global const int* indices_scene,\n"
    "                              __global const int* indices_model,\n"
    "                              int n,\n"
    "                              float4 centroid_scene,\n"
    "                              float4 centroid_model,\n"
    "                              __global float16* output) {\n"
    "    const uint index = get_global_id(0);\n"
    "    if (index >= n) {\n"
    "        return;\n"
    "    }\n"
    "    float4 spt = scene[indices_scene[index]] - centroid_scene;\n"
    "    float4 mpt = model[indices_model[index]] - centroid_model;\n"
    "    float norm = 1.f / (n-1);"
    "    output[index].s0 = spt.x * mpt.x * norm;\n"
    "    output[index].s1 = spt.y * mpt.x * norm;\n"
    "    output[index].s2 = spt.z * mpt.x * norm;\n"
    "    output[index].s3 = spt.x * mpt.y * norm;\n"
    "    output[index].s4 = spt.y * mpt.y * norm;\n"
    "    output[index].s5 = spt.z * mpt.y * norm;\n"
    "    output[index].s6 = spt.x * mpt.z * norm;\n"
    "    output[index].s7 = spt.y * mpt.z * norm;\n"
    "    output[index].s8 = spt.z * mpt.z * norm;\n"
    "    output[index].s9 = 0.f;\n"
    "    output[index].sa = 0.f;\n"
    "    output[index].sb = 0.f;\n"
    "    output[index].sc = 0.f;\n"
    "    output[index].sd = 0.f;\n"
    "    output[index].se = 0.f;\n"
    "    output[index].sf = 0.f;\n"
    "}\n"
    ;

}  // namespace detail

template <typename Point>
struct scene<Point>::impl {
    impl(typename cloud_t::ConstPtr cloud, gpu_state::sptr_t state) : cloud_(cloud), state_(state), program_(state), projection_(16, state->context), normalization_(16, state->context), transform_(16, state->context), uvw_transform_(16, state->context) {}

    ~impl() {
    }

    void init(model<Point>& m, float max_corr_dist) {
        // copy points to device
        std::vector<float> cpu_data;
        cpu_data.resize(cloud_->size() * 4);
        for (uint32_t i = 0; i < cloud_->size(); ++i) {
            cpu_data[i * 4 + 0] = cloud_->points[i].x;
            cpu_data[i * 4 + 1] = cloud_->points[i].y;
            cpu_data[i * 4 + 2] = cloud_->points[i].z;
            cpu_data[i * 4 + 3] = 1.f;
        }

        gpu_data_ = gpu_data_t(cloud_->size(), state_->context);
        gpu::copy(
            reinterpret_cast<gpu::float4_*>(cpu_data.data()),
            reinterpret_cast<gpu::float4_*>(cpu_data.data()) + cloud_->size(),
            gpu_data_.begin(), state_->queue);
        cpu_data.clear();

        // compile program
        std::string proj_shader = projector_t::opencl_path();
        program_.compile("util.cl", proj_shader, "icp.cl");

        // setup projection kernel
        uint32_t n = gpu_data_.size();
        vec2i_t ext = m.extents();
        vec2i_t margin = m.margin();
        projector_t model_proj = m.projector();
        gpu::eigen_copy_matrix_to_buffer(model_proj.projection_matrix(), projection_.begin(), state_->queue);
        gpu::eigen_copy_matrix_to_buffer(m.normalization(), normalization_.begin(), state_->queue);
        model_corrs_ = gpu::vector<int>(n, state_->context);
        scene_corrs_ = gpu::vector<int>(n, state_->context);
        positions_ = gpu::vector<gpu::float4_>(n, state_->context);
        corr_matrices_ = gpu::vector<gpu::float16_>(n, this->state_->context);

        proj_kernel_ = program_.kernel(
            "icp_projection",
            gpu_data_.get_buffer(),
            n,
            m.device_data().get_buffer(),
            gpu::int2_(ext[0], ext[1]),
            gpu::int2_(margin[0], margin[1]),
            no_arg,
            no_arg,
            projection_.get_buffer(),
            normalization_.get_buffer(),
            max_corr_dist,
            positions_.get_buffer(),
            model_corrs_.get_buffer(),
            scene_corrs_.get_buffer()
        );

        corr_kernel_ = program_.kernel(
            "icp_correlation",
            positions_.get_buffer(),
            m.device_data().get_buffer(),
            scene_corrs_.get_buffer(),
            model_corrs_.get_buffer(),
            no_arg,
            no_arg,
            no_arg,
            corr_matrices_.get_buffer()
        );

        global_threads_ = n;
        int res = n % detail::threads_per_block;
        if (res) {
            global_threads_ += detail::threads_per_block - res;
        }
    }

    std::tuple<mat4f_t, mat4f_t, uint32_t>
    find(model<Point>& m, uint32_t accept_threshold, uint32_t early_out_threshold, const sample_parameters& params, uint32_t max_icp_iterations, subset_t subset, statistics* stats) {
        pcl::IndicesPtr indices;
        if (subset.empty()) {
            indices = pcl::IndicesPtr(new std::vector<int>(cloud_->size()));
            std::iota(indices->begin(), indices->end(), 0);
        } else {
            indices = pcl::IndicesPtr(
                new std::vector<int>(subset.begin(), subset.end()));
        }
        kdtree_.setInputCloud(cloud_, indices);
        //float lower = params.min_diameter_factor;
        float upper = m.diameter() * params.max_diameter_factor;

        std::mt19937 rng;
        uint32_t seed = 13;
        if (!detail::deterministic) {
            auto now = std::chrono::system_clock::now();
            seed = now.time_since_epoch().count();
        }
        rng.seed(seed);
        std::shuffle(indices->begin(), indices->end(), rng);
        // DEBUG
        (*indices)[0] = 0;

        uint32_t best_score = 0;
        mat4f_t  best_transform = mat4f_t::Identity();
        uint64_t valid_sample_count = 0, sample_count = 0;

        uint32_t n_model = m.cloud()->size();
        uint32_t n_scene = indices->size();
        uint64_t outer_bound = static_cast<uint64_t>(std::log(1.0 - detail::match_probability) / std::log(1.0 - static_cast<double>(n_model) / n_scene));
        outer_bound = std::max(outer_bound, 10ul);

        fmt::print("outer bound: {}\n", outer_bound);

        for (uint64_t outer = 0; outer < std::min(indices->size(), outer_bound); ++outer) {
            int i = (*indices)[outer];
            const Point& p1 = cloud_->points[i];

            std::vector<int> nn;
            std::vector<float> dist;
            kdtree_.radiusSearch(p1, upper, nn,
                                 dist);
            if (nn.empty()) continue;
            std::shuffle(nn.begin(), nn.end(), rng);

            double prob = static_cast<double>(n_model) / nn.size();
            uint64_t inner_bound = static_cast<uint64_t>(-std::log(1.0 - detail::match_probability) / prob);
            // DEBUG: set to meaningful minimum
            inner_bound = std::max(inner_bound, nn.size());
            inner_bound = std::min(inner_bound, nn.size());

            for (uint64_t inner0 = 0; inner0 < inner_bound; ++inner0) {
                int j = nn[inner0];
                if (j == i) continue;

                bool debug = j == 265;

                projector_t proj(0.f);


                const Point& p2 = cloud_->points[j];
                proj.fit(p1, p2);

                if (debug) {
                    pdebug("   scene normals:");
                    pdebug("   n0: {}", p1.getNormalVector3fMap().transpose());
                    pdebug("   n1: {}", p2.getNormalVector3fMap().transpose());
                    pdebug("   axis: {}", proj.axis().transpose());
                    pdebug("   dot: {}", p2.getNormalVector3fMap().dot(p1.getNormalVector3fMap()));
                }

                //fmt::print("sample pair {} {}\n", i, j);

                if (debug) pdebug("scene descriptor");
                vec2f_t d = proj.intrinsic_difference(p1.getVector3fMap(), p2.getVector3fMap(), debug);

                auto && [q_first, q_last] = m.query(p1, p2, d, debug);
                if (q_first != q_last) {
                    ++valid_sample_count;
                }

                ++sample_count;
                for (auto it = q_first; it != q_last; ++it) {
                    auto m_i = it->second.first;
                    //auto [m_i, m_j] = it->second;

                    //if (i == static_cast<int>(m_i) && j == static_cast<int>(m_j)) {
                    //if (m_i == 0) {
                        //pdebug("    model pair {} {}", m_i, m_j);
                        //pdebug("    desc {}", it->first.transpose());
                    //}

                    mat4f_t transform = proj.transform_to(m.projector(), p1.getVector3fMap(), m.cloud()->points[m_i].getVector3fMap());
                    uint32_t score = correspondence_count_(m, proj, transform);
                    //fmt::print("score: {}\n", score);

                    if (score > best_score) {
                        best_transform = transform;
                        best_score = score;

                        fmt::print("improved score: {}\n", best_score);

                        if (detail::early_out && best_score > early_out_threshold) {
                            fmt::print("early out at: {} of {} points\n", best_score, m.cloud()->size());
                            mat4f_t uvw_transform;
                            std::tie(uvw_transform, std::ignore) = icp_(m, transform, max_icp_iterations);
                            return {transform, uvw_transform, best_score};
                        }
                    }
                }
            }

            // DEBUG
            break;
        }

        if (stats) {
            stats->rejection_rate =
                static_cast<double>(sample_count - valid_sample_count) /
                sample_count;
        }

        mat4f_t uvw_transform;
        std::tie(uvw_transform, std::ignore) = icp_(m, best_transform, max_icp_iterations);
        return {best_transform, uvw_transform, best_score};
    }

    void
    project_(model<Point>& m, const mat4f_t& transform, const mat4f_t& uvw_transform = mat4f_t::Identity()) {
        gpu::eigen_copy_matrix_to_buffer(transform, transform_.begin(), state_->queue);
        proj_kernel_.set_arg(5, transform_.get_buffer());
        gpu::eigen_copy_matrix_to_buffer(uvw_transform, uvw_transform_.begin(), state_->queue);
        proj_kernel_.set_arg(6, uvw_transform_.get_buffer());
        state_->queue.enqueue_1d_range_kernel(proj_kernel_, 0, global_threads_, detail::threads_per_block);

        std::vector<gpu::float4_> positions(positions_.size());
        //std::vector<int> model_corrs(positions_.size());
        gpu::copy(positions_.begin(), positions_.end(), positions.begin(), state_->queue);
        //gpu::copy(model_corrs_.begin(), model_corrs_.end(), model_corrs.begin(), state_->queue);
        for (uint32_t i = 0; i < positions.size(); ++i) {
            //fmt::print("pos: {} {} {}\n", positions[i][0], positions[i][1], positions[i][2]);
            ////if (model_corrs[i] >= 0) {
            if (positions[i][3] > 0.f) {
                //fmt::print("dist: {}\n", positions[i][3]);
            }
        }
    }

    template <typename IndexIterator>
    gpu::float4_ compute_centroid_(gpu::vector<gpu::float4_>& data, IndexIterator begin, IndexIterator end) {
        gpu::float4_ result;
        gpu::reduce(
            gpu::make_permutation_iterator(data.begin(), begin),
            gpu::make_permutation_iterator(gpu::make_buffer_iterator<gpu::float4_>(data.get_buffer(), end.get_index()), end),
            &result,
            this->state_->queue
        );
        float inv_count = 1.f / end.get_index();
        result[0] *= inv_count;
        result[1] *= inv_count;
        result[2] *= inv_count;
        //result[0] /= result[3];
        //result[1] /= result[3];
        //result[2] /= result[3];
        return result;
    }

    uint32_t
    correspondence_count_(model<Point>& m, const projector_t& proj, const mat4f_t& transform) {
        project_(m, transform);

        uint32_t sum = 0;
        BOOST_COMPUTE_FUNCTION(int, indicator, (int x),
        {
            return x >= 0 ? 1 : 0;
        });
        gpu::transform_reduce(scene_corrs_.begin(), scene_corrs_.end(), &sum, indicator, gpu::plus<int>(), this->state_->queue);

        return sum;
    }

    std::pair<mat4f_t, std::vector<int>>
    icp_(model<Point>& m, const mat4f_t& transform, uint32_t max_iterations) {
        project_(m, transform);

        gpu::buffer_iterator<int> valid_model;
        gpu::buffer_iterator<int> valid_scene;

        valid_model = gpu::stable_partition(
            model_corrs_.begin(), model_corrs_.end(),
            boost::compute::lambda::_1 >= 0, this->state_->queue);
        valid_scene = gpu::stable_partition(
            scene_corrs_.begin(), scene_corrs_.end(),
            boost::compute::lambda::_1 >= 0, this->state_->queue);

        mat4f_t uvw_transform = mat4f_t::Identity();
        for (uint32_t i = 0; i < max_iterations; ++i) {
            uint32_t n_model = valid_model.get_index();
            uint32_t n_scene = valid_scene.get_index();
            if (!n_model) {
                // no correspondences
                return {mat4f_t::Identity(), std::vector<int>()};
            }
            int threads = n_scene;
            int res = n_scene % detail::threads_per_block;
            if (res) {
                threads += detail::threads_per_block - res;
            }

            // compute centroids
            gpu::float4_ centroid_scene =
                compute_centroid_(positions_, scene_corrs_.begin(), valid_scene);
            gpu::float4_ centroid_model =
                compute_centroid_(m.device_data(),
                                  model_corrs_.begin(), valid_model);

            // coll correlation kernel
            corr_kernel_.set_arg(4, n_scene);
            corr_kernel_.set_arg(5, centroid_scene);
            corr_kernel_.set_arg(6, centroid_model);
            this->state_->queue.enqueue_1d_range_kernel(corr_kernel_, 0, threads, detail::threads_per_block);
            // sum of per-point correlation matrices
            gpu::float16_ corr_16;
            gpu::reduce(corr_matrices_.begin(), gpu::make_buffer_iterator<gpu::float16_>(corr_matrices_.get_buffer(), n_scene), &corr_16, this->state_->queue);
            mat3f_t corr_matrix;
            for (int i = 0; i < 9; ++i) {
                corr_matrix.data()[i] = corr_16[i];
            }

            // compute SVD for rotation estimation
            Eigen::JacobiSVD<mat3f_t> svd(corr_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
            mat4f_t icp_mat = mat4f_t::Identity();
            icp_mat.topLeftCorner<3,3>() = svd.matrixV() * svd.matrixU().transpose();
            vec3f_t cm(centroid_model[0], centroid_model[1], centroid_model[2]);
            vec3f_t cs(centroid_scene[0], centroid_scene[1], centroid_scene[2]);
            icp_mat.block<3,1>(0,3) = cm - icp_mat.topLeftCorner<3,3>() * cs;

            // update and reproject for correct correspondences
            uvw_transform = icp_mat * uvw_transform;
            project_(m, transform, uvw_transform);
            valid_model = gpu::stable_partition(
                model_corrs_.begin(), model_corrs_.end(),
                boost::compute::lambda::_1 >= 0, this->state_->queue);
            valid_scene = gpu::stable_partition(
                scene_corrs_.begin(), scene_corrs_.end(),
                boost::compute::lambda::_1 >= 0, this->state_->queue);
        }

        uint32_t n_scene = valid_scene.get_index();
        std::vector<int> corr(n_scene);
        gpu::copy(scene_corrs_.begin(), valid_scene, corr.begin(), this->state_->queue);

        return {uvw_transform, corr};
    }

    typename cloud_t::ConstPtr cloud() const {
        return cloud_;
    }

    //gpu_data_t& device_data() {
        //return gpu_data_;
    //}

    //const gpu_data_t& device_data() const {
        //return gpu_data_;
    //}

    typename cloud_t::ConstPtr cloud_;
    gpu_state::sptr_t          state_;
    pcl::search::KdTree<Point> kdtree_;
    gpu_data_t                 gpu_data_;

    gpu_program program_;
    gpu::kernel proj_kernel_;
    gpu::kernel corr_kernel_;
    int global_threads_;
    gpu::vector<float> projection_;
    gpu::vector<float> normalization_;
    gpu::vector<float> transform_;
    gpu::vector<float> uvw_transform_;
    gpu::vector<gpu::float4_> positions_;
    gpu::vector<gpu::float16_> corr_matrices_;
    gpu::vector<int> model_corrs_;
    gpu::vector<int> scene_corrs_;
};


template <typename Point>
inline
scene<Point>::scene(typename cloud_t::ConstPtr cloud, gpu_state::sptr_t state) : impl_(new impl(cloud, state)) {
}

template <typename Point>
inline
scene<Point>::~scene() {
}

template <typename Point>
inline void
scene<Point>::init(model<Point>& m, float max_corr_dist) {
    impl_->init(m, max_corr_dist);
}

//template <typename Point>
//inline mat4f_t
//scene<Point>::find(model<Point>& m, uint32_t early_out_threshold, const sample_parameters& sample_params, statistics* stats) {
    //return impl_->find(m, early_out_threshold, sample_params, subset_t(), stats);
//}

template <typename Point>
inline std::tuple<mat4f_t, mat4f_t, uint32_t>
scene<Point>::find(model<Point>& m, uint32_t accept_threshold, uint32_t early_out_threshold, const sample_parameters& sample_params, uint32_t max_icp_iterations, const subset_t& subset, statistics* stats) {
    return impl_->find(m, accept_threshold, early_out_threshold, sample_params, max_icp_iterations, subset, stats);
}

template <typename Point>
inline typename scene<Point>::cloud_t::ConstPtr
scene<Point>::cloud() const {
    return impl_->cloud();
}

//template <typename Point>
//inline typename scene<Point>::gpu_data_t&
//scene<Point>::device_data() {
    //return impl_->device_data();
//}

//template <typename Point>
//inline const typename scene<Point>::gpu_data_t&
//scene<Point>::device_data() const {
    //return impl_->device_data();
//}

}  // namespace triplet_match
