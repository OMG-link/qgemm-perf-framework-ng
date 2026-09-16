# The cache-blocked variant reuses the M4 immediate-reduction microkernel that
# `q4_0-q8_0-IME-m4i` owns, so this directory only provides the blocked adapter
# and needs no extra microkernel source or assembly target.
ime_add_kernel_sources(SOURCES adapter.cpp
    COMPILE_OPTIONS -fopenmp)
