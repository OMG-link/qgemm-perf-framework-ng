set(PFNG_PLATFORM_ID "spacemit-k3-a100")
set(PFNG_PLATFORM_DISPLAY_NAME "SpacemiT K3 / A100")
# VLEN is 1024 bit and the matrix extension is XSMTVDotII, neither of which the
# K1 profiles have. zvl1024b is listed explicitly so the arch string alone
# describes the register width.
set(PFNG_PLATFORM_ARCH "rv64gcv_zvfh_zicbop_zba_zvl1024b_xsmtvdotii")
set(PFNG_PLATFORM_TUNE "spacemit-a100")
set(PFNG_PLATFORM_COMPILE_DEFINITIONS PFNG_PLATFORM_SPACEMIT_K3_A100)
set(PFNG_PLATFORM_DEVICE_TREE_COMPATIBLES "spacemit,k3-pico-itx")
# The A100 cores are CPUs 8-15; 0-7 are X100 cores and run a different ISA
# configuration. Threads must enter the A100 domain before they can be pinned,
# which PFNG_PLATFORM_REMOTE_PROLOGUE does.
set(PFNG_PLATFORM_FIRST_CPU 8)
set(PFNG_PLATFORM_LAST_CPU 15)
set(PFNG_PLATFORM_MAX_THREADS 8)
set(PFNG_PLATFORM_REMOTE_PROLOGUE "echo 0 > /proc/set_ai_thread")

set(PFNG_PLATFORM_KERNEL_COMPILE_OPTIONS
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
    "SHELL:-mllvm --misched-cluster=false")
set(PFNG_PLATFORM_FIXED_VLEN_COMPILE_OPTIONS
    -mllvm
    -riscv-v-vector-bits-min=1024
    -mllvm
    -riscv-v-vector-bits-max=1024)
