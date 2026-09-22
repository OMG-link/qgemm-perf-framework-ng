set(PFNG_PLATFORM_ID "spacemit-k1-a60")
set(PFNG_PLATFORM_DISPLAY_NAME "SpacemiT K1 / A60")
set(PFNG_PLATFORM_ARCH "rv64gcv_zvfh_zicbop_zba_xsmtvdot")
# This is the compiler-recognized tuning name. It intentionally differs from
# the project's corrected platform ID.
set(PFNG_PLATFORM_TUNE "spacemit-x60")
set(PFNG_PLATFORM_COMPILE_DEFINITIONS PFNG_PLATFORM_SPACEMIT_K1_A60)
set(PFNG_PLATFORM_DEVICE_TREE_COMPATIBLES "spacemit,k1-x")
set(PFNG_PLATFORM_FIRST_CPU 0)
set(PFNG_PLATFORM_LAST_CPU 3)
set(PFNG_PLATFORM_MAX_THREADS 4)

set(PFNG_PLATFORM_KERNEL_COMPILE_OPTIONS
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
    "SHELL:-mllvm --misched-cluster=false")
set(PFNG_PLATFORM_FIXED_VLEN_COMPILE_OPTIONS
    -mllvm
    -riscv-v-vector-bits-min=256
    -mllvm
    -riscv-v-vector-bits-max=256)