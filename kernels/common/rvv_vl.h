#ifndef PFNG_KERNEL_COMMON_RVV_VL_H
#define PFNG_KERNEL_COMMON_RVV_VL_H

#include <cstddef>

namespace pfng::rvv {

inline constexpr size_t vl(size_t element_bits, size_t lmul) {
    return lmul * 256 / element_bits;
}

} // namespace pfng::rvv

#endif