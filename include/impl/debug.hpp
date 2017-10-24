#ifndef _TRIPLET_MATCH_DEBUG_HPP_
#define _TRIPLET_MATCH_DEBUG_HPP_

#include <common>

namespace triplet_match {

inline void
to_grayscale_image(const std::string& file, const Eigen::MatrixXf& m) {
    float factor = m.maxCoeff();
    if (factor > Eigen::NumTraits<float>::dummy_precision()) {
        factor = 255.f / factor;
    } else {
        factor = 1.f;
    }
    Eigen::MatrixXf scaled = factor * m;
    std::ofstream out(file);
    out << "P2\n";
    out << m.cols() << " " << m.rows() << "\n";
    out << "255\n";
    //out << img.template cast<int>();

    for (uint32_t j = 0; j < static_cast<uint32_t>(scaled.rows()); ++j) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(scaled.cols()); ++i) {
            if (i) out << " ";
            float value = scaled(j,i);
            out << (value < 0.f ? 0 : 255 - static_cast<int>(scaled(j,i)));
        }
        out << "\n";
    }

    out.close();
}

} // triplet_match

#endif /* _TRIPLET_MATCH_DEBUG_HPP_ */
