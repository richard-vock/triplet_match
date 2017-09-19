namespace triplet_match {

template <typename Point>
struct scene<Point>::impl {
    impl(typename cloud_t::ConstPtr cloud);

    ~impl();

    template <typename PointModel>
    std::pair<mat4f_t, uint32_t>
    find(model<PointModel>& m, std::function<uint32_t (const mat4f_t&)> score_func, std::function<bool (uint32_t)> early_out_func, const sample_parameters& params, subset_t subset, statistics* stats);

    typename cloud_t::ConstPtr cloud() const {
        return cloud_;
    }

    typename cloud_t::ConstPtr cloud_;
    pcl::search::KdTree<Point> kdtree_;
};

template <typename Point>
template <typename PointModel, typename ScoreFunctor, typename EarlyOutFunctor>
inline mat4f_t
scene<Point>::find(model<PointModel>& m, ScoreFunctor&& score_func, EarlyOutFunctor&& early_out_func, const sample_parameters& sample_params, statistics* stats) {
    return impl_->find(m, [&] (const mat4f_t& t) { return score_func(t); }, sample_params, subset_t(), stats);
}

template <typename Point>
template <typename PointModel, typename ScoreFunctor, typename EarlyOutFunctor>
inline std::pair<mat4f_t, uint32_t>
scene<Point>::find(model<PointModel>& m, ScoreFunctor&& score_func, EarlyOutFunctor&& early_out_func, const sample_parameters& sample_params, const subset_t& subset, statistics* stats) {
    return impl_->find(m, [&] (const mat4f_t& t) { return score_func(t); }, [&] (uint32_t score) { return early_out_func(score); }, sample_params, subset, stats);
}

} // triplet_match
