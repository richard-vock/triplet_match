namespace triplet_match {

namespace detail {

inline std::string
load_shader_code() {
    return "";
}

template <typename Arg, typename... Args>
inline std::string
load_shader_code(Arg&& file, Args&&... files) {
    std::ifstream t(std::string(OPENCL_PREFIX) + file);
    std::string str((std::istreambuf_iterator<char>(t)),
                     std::istreambuf_iterator<char>());
    return str + "\n" + detail::load_shader_code(std::forward<Args>(files)...);
}

template <typename Arg>
struct set_kernel_arg {
    void operator()(gpu::kernel& kernel, size_t index, Arg arg) {
        kernel.set_arg(index, arg);
    }
};

template <>
struct set_kernel_arg<no_arg_t> {
    void operator()(gpu::kernel&, size_t index, no_arg_t) {
    }
};

template <typename Arg, typename... Args>
inline void
set_kernel_args(gpu::kernel& kernel, const std::tuple<Arg, Args...>& args, std::index_sequence<> indices) {
}

template <typename Arg, typename... Args, std::size_t I, std::size_t... Is>
inline void
set_kernel_args(gpu::kernel& kernel, const std::tuple<Arg, Args...>& args, std::index_sequence<I, Is...> indices) {
    set_kernel_arg<std::tuple_element_t<I, std::tuple<Arg, Args...>>>()(kernel, I, std::get<I>(args));
    set_kernel_args(kernel, args, std::index_sequence<Is...>{});
}

} // detail

template <typename... Args>
inline void
gpu_program::compile(Args&&... files) {
    std::string code = detail::load_shader_code(std::forward<Args>(files)...);
    program_ = gpu::program::create_with_source(code, state_->context);
    program_.build();
}

template <typename... Args>
inline gpu::kernel
gpu_program::kernel(std::string function_name, Args&&... args) {
    gpu::kernel k = gpu::kernel(program_, function_name);
    detail::set_kernel_args(k, std::tuple(std::forward<Args>(args)...), std::index_sequence_for<Args...>{});
    return k;
}

} // triplet_match
