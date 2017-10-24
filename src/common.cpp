#include <common>

namespace triplet_match {

gpu_state::gpu_state()
    : device(gpu::system::default_device()),
      context(device),
      queue(context, device) {}

} // triplet_match
