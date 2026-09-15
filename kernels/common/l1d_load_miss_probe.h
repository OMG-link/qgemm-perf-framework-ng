#pragma once

// Each probe emits a globally visible [*_start, *_end) ELF symbol range.  The
// start label is exactly on the candidate load; the end follows 64 32-bit NOPs.
#define IME_L1D_PROBE_STRINGIFY_IMPL(value) #value
#define IME_L1D_PROBE_STRINGIFY(value) IME_L1D_PROBE_STRINGIFY_IMPL(value)
#define IME_L1D_PROBE_CAT_IMPL(lhs, rhs) lhs##rhs
#define IME_L1D_PROBE_CAT(lhs, rhs) IME_L1D_PROBE_CAT_IMPL(lhs, rhs)

#define IME_L1D_PROBE_ASM(symbol, load_instruction)                                                                                     \
    ".balign 4\n"                                                                                                                       \
    ".globl " IME_L1D_PROBE_STRINGIFY(symbol) "_start\n"                                                                              \
    IME_L1D_PROBE_STRINGIFY(symbol) "_start:\n" load_instruction "\n"                                                                \
    ".option push\n"                                                                                                                   \
    ".option norvc\n"                                                                                                                  \
    ".rept 64\n"                                                                                                                       \
    "nop\n"                                                                                                                            \
    ".endr\n"                                                                                                                         \
    ".option pop\n"                                                                                                                    \
    ".globl " IME_L1D_PROBE_STRINGIFY(symbol) "_end\n"                                                                                \
    IME_L1D_PROBE_STRINGIFY(symbol) "_end:\n"

#define IME_L1D_PROBE_VLE8_M1(type, value, address, symbol)                                                                             \
    type value;                                                                                                                         \
    asm volatile("vsetvli t0, zero, e8, m1, ta, ma\n" IME_L1D_PROBE_ASM(symbol, "vle8.v %0, (%1)") : "=vr"(value) : "r"(address) : "memory", "t0", "v0", "v1")

#define IME_L1D_PROBE_VL8RE8_M8(type, value, address, symbol)                                                                           \
    type value;                                                                                                                         \
    asm volatile(IME_L1D_PROBE_ASM(symbol, "vl8re8.v %0, (%1)") : "=vr"(value) : "r"(address) : "memory", "v0", "v1")

#define IME_L1D_PROBE_VLE8_M8(type, value, address, avl, symbol)                                                                        \
    type value;                                                                                                                         \
    asm volatile("vsetvli t0, %2, e8, m8, ta, ma\n" IME_L1D_PROBE_ASM(symbol, "vle8.v %0, (%1)") : "=vr"(value) : "r"(address), "r"(avl) : "memory", "t0", "v0", "v1")

#define IME_L1D_PROBE_VLE16_M1(type, value, address, avl, symbol)                                                                       \
    type value;                                                                                                                         \
    asm volatile("vsetvli t0, %2, e16, m1, ta, ma\n" IME_L1D_PROBE_ASM(symbol, "vle16.v %0, (%1)") : "=vr"(value) : "r"(address), "r"(avl) : "memory", "t0", "v0", "v1")

#define IME_L1D_PROBE_VLE32_M2(type, value, address, avl, symbol)                                                                       \
    type value;                                                                                                                         \
    asm volatile("vsetvli t0, %2, e32, m2, ta, ma\n" IME_L1D_PROBE_ASM(symbol, "vle32.v %0, (%1)") : "=vr"(value) : "r"(address), "r"(avl) : "memory", "t0", "v0", "v1")

#define IME_L1D_PROBE_FLW(value, address, symbol)                                                                                       \
    float value;                                                                                                                        \
    asm volatile(IME_L1D_PROBE_ASM(symbol, "flw %0, 0(%1)") : "=f"(value) : "r"(address) : "memory", "v0", "v1")
