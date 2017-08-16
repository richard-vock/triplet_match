#include <iostream>

#include <pcl/io/pcd_io.h>
#include <voxel_score/score_functor>
namespace vs = voxel_score;

#include <scene>
namespace tr = triplet_match;

typedef pcl::PointNormal point_t;
typedef pcl::PointCloud<point_t> cloud_t;

int main (int argc, char const* argv[]) {
    cloud_t::Ptr model_cloud(new cloud_t());
    pcl::io::loadPCDFile(argv[1], *model_cloud);

    tr::discretization_params dparams {
        0.05f,
        10.f / 180.f * static_cast<float>(M_PI)
    };

    // model
    std::cout << "init model" << "\n";
    std::unique_ptr<tr::model<point_t>> m(new tr::model<point_t>(model_cloud, dparams));
    auto sync_model = m->init({0.2, 1.0, 0.9});

    std::cout << "init voxel" << "\n";
    vs::gpu_state::sptr_t gstate(new vs::gpu_state());
    vs::score_functor<point_t, point_t> score(gstate);
    score.set_model(model_cloud, 100);

    // scene
    std::cout << "init scene" << "\n";
    cloud_t::Ptr scene_cloud(new cloud_t());
    pcl::io::loadPCDFile(argv[2], *scene_cloud);
    std::unique_ptr<tr::scene<point_t>> s(new tr::scene<point_t>(scene_cloud));
    score.set_scene(scene_cloud);

    sync_model.wait();
    std::cout << "start find" << "\n";
    float max_score = 0.f;
    vs::mat4f_t t = s->find(*m, [&] (const tr::mat4f_t& transform) {
        float s = score(transform);
        if (s > max_score) {
            max_score = s;
            std::cout << max_score << "\n";
        }
        return true;
    }, {0.2f, 1.0f, 0.9f, 1.f});
}
