#ifndef IME_KERNEL_COMMON_RVV_VL_H
#define IME_KERNEL_COMMON_RVV_VL_H

#include <cstddef>

namespace ime::rvv {

inline constexpr size_t vl(size_t element_bits, size_t lmul) {
    return lmul * 256 / element_bits;
}

} // namespace ime::rvv

#endif