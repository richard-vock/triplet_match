#include <gpu_program>

namespace triplet_match {

gpu_program::gpu_program(gpu_state::sptr_t state) : state_(state) {
}

gpu_program::~gpu_program() {
}

} // triplet_match
