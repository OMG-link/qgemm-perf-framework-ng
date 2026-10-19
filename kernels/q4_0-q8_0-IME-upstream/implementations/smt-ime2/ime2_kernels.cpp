// Local source extract of the q4_0 x q8_0 (i8-i4) path from llama.cpp's
// SpacemiT IME2 kernel file. See ../../provenance.md at the kernel root for
// the upstream revision, the extracted line ranges, the ABI the adapter
// depends on, and the differences from the original.
//
// Extracted from ime2_kernels.cpp lines 217-322, 2430-2881, 3007-3358 and
// 5583-5610: the i8i4 layout reference, the M1 and M4 microkernels, and the
// gemm_kernel_i8i4 dispatcher that chooses between them.

#include "ime2_dispatch.h"

#include <algorithm>
#include <cmath>

namespace spacemit_kernels {
namespace ime2 {

template <size_t MB_ROWS, size_t NB_COLS>
void gemm_kernel_i8i4_mrow_ref(size_t          blk_len,
                               const uint8_t * quant_a_ptr,
                               const uint8_t * quant_b_data,
                               const uint8_t * quant_b_zp,
                               float *         c_ptr,
                               size_t          count_m,
                               size_t          count_n,
                               size_t          k_blks,
                               size_t          ldc) {
    constexpr size_t kblks_per_blk = 16;
    GGML_ASSERT(k_blks % kblks_per_blk == 0);

    int64_t b_blk_stride        = (sizeof(_Float16) + (blk_len / 2) + (quant_b_zp ? sizeof(uint8_t) : 0));
    int64_t b_stride            = k_blks * b_blk_stride;
    int64_t a_blk_stride        = q8_blk_size(blk_len, true);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t b_ncol_block_stride = b_blk_stride * NB_COLS;

    float    output[MB_ROWS * NB_COLS]     = { 0 };
    _Float16 output_f16[MB_ROWS * NB_COLS] = { 0 };

    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
        uint8_t * b_data  = (uint8_t *) quant_b_data + ni * b_stride + NB_COLS * sizeof(_Float16);
        if (quant_b_zp) {
            b_data += NB_COLS * sizeof(uint8_t);
        }

        int8_t * a_data = (int8_t *) quant_a_ptr + sizeof(float) * MB_ROWS + sizeof(int16_t) * MB_ROWS;

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < NB_COLS; ci++) {
                output[ci + mi * NB_COLS]     = 0.0f;
                output_f16[ci + mi * NB_COLS] = static_cast<_Float16>(0.0f);
            }
        }

        size_t kii = 0;
        for (size_t ki = 0; ki < k_blks; ki++, a_data += a_nrow_block_stride, b_data += b_ncol_block_stride) {
            _Float16 * b_scale_fp16 = (_Float16 *) (b_data - NB_COLS * sizeof(_Float16));
            uint8_t *  b_zp         = nullptr;
            if (quant_b_zp) {
                b_scale_fp16 = (_Float16 *) (b_data - NB_COLS * sizeof(_Float16) - NB_COLS * sizeof(uint8_t));
                b_zp         = (uint8_t *) (b_data - NB_COLS * sizeof(uint8_t));
            }

            float *   a_scale_row = (float *) (a_data - sizeof(float) * MB_ROWS - sizeof(int16_t) * MB_ROWS);
            int16_t * a_sum_row   = (int16_t *) (a_data - sizeof(int16_t) * MB_ROWS);

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                _Float16 a_scale = a_scale_row[mi];
                int16_t  a_sum   = a_sum_row[mi];

                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    _Float16 b_scale = b_scale_fp16[ci];
                    int32_t  acc     = 0;
                    if (b_zp) {
                        acc += a_sum * b_zp[ci];
                    } else {
                        acc += a_sum * 8;
                    }
                    for (size_t bi = 0; bi < blk_len / 2; bi++) {
                        int8_t  a0 = a_data[mi * blk_len + 2 * bi];
                        int8_t  a1 = a_data[mi * blk_len + 2 * bi + 1];
                        uint8_t b  = b_data[ci * blk_len / 2 + bi];
                        int8_t  b0 = static_cast<int8_t>(b & 0x0F);
                        int8_t  b1 = static_cast<int8_t>((b & 0xF0) >> 4);
                        acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0) +
                               static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
                    }
                    output_f16[ci + mi * NB_COLS] +=
                        static_cast<float>(acc) * static_cast<float>(a_scale) * static_cast<float>(b_scale);
                }
            }

            if (kii == kblks_per_blk - 1) {
                for (size_t mi = 0; mi < MB_ROWS; mi++) {
                    for (size_t ci = 0; ci < NB_COLS; ci++) {
                        output[ci + mi * NB_COLS] += static_cast<float>(output_f16[ci + mi * NB_COLS]);
                        output_f16[ci + mi * NB_COLS] = 0.0f;
                    }
                }
                kii = 0;
            } else {
                kii++;
            }
        }

        if (kii == kblks_per_blk - 1) {
            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    output[ci + mi * NB_COLS] += static_cast<float>(output_f16[ci + mi * NB_COLS]);
                    output_f16[ci + mi * NB_COLS] = 0.0f;
                }
            }
            kii = 0;
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < nb_real; ci++) {
                c_ptr[mi * ldc + ci] = output[mi * NB_COLS + ci];
            }
        }
    }
}
void gemm_kernel_i8i4_m1(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    if (quant_b_zp == NULL) {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                      n * k_blks * blk_len / 2 +      // b data
                                      n * k_blks * sizeof(_Float16);  // scale
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format Version_1 (FP32 SCALE FOR Normal VMADOTins of IME2)
            // A M1K32 int8    256bit
            // Ascale fp32 * 1  32bit
            // || scl*1(fp32) | Asum(int16) | blk0 || scl*1(fp32) | Asum(int16) | blk0 || ...
            // || Element                          || Element                          || ...
            // B format
            // B N8K32 int4    1024bit
            //   4VRF, N32K32, 4096bit
            // Bscale fp16 * N32 512bit;
            // || scl*32..(fp16) | blk0 blk1 ... blk31 || scl*32..(fp16) | blk0 blk1 ... blk31 || ...
            // || Element                              || Element                              || ...
#if 1
            //bias always be nullptr
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*2         \n\t"  // s5 = pBdata;
                "mv           s6, %[pC]               \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc

                // ordinary vmadot: vle*6 flw*1 vecIns*21 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+64        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v0, (s4)                \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64+128*4        \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, zero, e8, m1        \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "smt.vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "smt.vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "smt.vmadotsu     v16, v10, v4, i4        \n\t"  // M0 N0 - N7 INT32(256bit)
                "smt.vmadotsu     v18, v10, v5, i4        \n\t"  // M0 N8 - N15
                "smt.vmadotsu     v20, v10, v6, i4        \n\t"  // M0 N16 - N23
                "smt.vmadotsu     v22, v10, v7, i4        \n\t"  // M0 N24 - N31

                "vsll.vi      v16, v16, 4             \n\t"
                "vsll.vi      v18, v18, 4             \n\t"
                "vsll.vi      v20, v20, 4             \n\t"
                "vsll.vi      v22, v22, 4             \n\t"

                "smt.vmadotu      v16, v8, v4, i4         \n\t"
                "smt.vmadotu      v18, v8, v5, i4         \n\t"
                "smt.vmadotu      v20, v8, v6, i4         \n\t"
                "smt.vmadotu      v22, v8, v7, i4         \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v28, 8                  \n\t"
                "smt.vpack.vv     v24, v16, v18, 2        \n\t"
                "smt.vpack.vv     v26, v20, v22, 2        \n\t"
                "smt.vpack.vv     v16, v24, v26, 3        \n\t"

                "vwmul.vx     v24, v28, t2            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vadd.vv      v16, v16, v24           \n\t"

                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v24, v0                 \n\t"
                // mac result i32 -> fp32
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v16                \n\t"
                // a_scale * b_scale;
                "vfmul.vf     v1, v24, f0             \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v2, v1, v26             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#else
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // init the scale
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*2         \n\t"  // s5 = pBdata;
                "mv           s6, %[pC]               \n\t"

                "vsll.vi      v1, v0, 4               \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc
                "vfcvt.f.x.v  v0, v0                  \n\t"
                "vfcvt.f.x.v  v1, v1                  \n\t"

                // vmadot hp: vle*7 flw*1 vecIns*14 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+64        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v30, (s4)               \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64+128*4        \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v28, 8                  \n\t"  // Bzp u8 -> u16
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmul.vx      v26, v28, t2            \n\t"  // asum*zp i16*i16
                "smt.vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "smt.vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vfcvt.f.x.v  v16, v26                \n\t"  // zp i16 -> fp16
                "vadd.vi      v18, v16, 0             \n\t"
                "vadd.vi      v20, v16, 0             \n\t"
                "vadd.vi      v22, v16, 0             \n\t"

                "smt.vmadotsu.hp  v16, v10, v4, v1, 0, i4 \n\t"  // high 4
                "smt.vmadotsu.hp  v18, v10, v5, v1, 0, i4 \n\t"
                "smt.vmadotsu.hp  v20, v10, v6, v1, 0, i4 \n\t"
                "smt.vmadotsu.hp  v22, v10, v7, v1, 0, i4 \n\t"
                "smt.vmadotu.hp   v16, v8, v4, v0, 0, i4  \n\t"  // low 4
                "smt.vmadotu.hp   v18, v8, v5, v0, 0, i4  \n\t"
                "smt.vmadotu.hp   v20, v8, v6, v0, 0, i4  \n\t"
                "smt.vmadotu.hp   v22, v8, v7, v0, 0, i4  \n\t"

                "smt.vpack.vv     v24, v16, v18, 1        \n\t"
                "smt.vpack.vv     v26, v20, v22, 1        \n\t"
                "smt.vpack.vv     v16, v24, v26, 2        \n\t"

                "vsetvli      t0, x0, e16, mf2        \n\t"
                // mac result * b_scale; f16*f16->f32
                "vfwmul.vv     v31, v30, v16          \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum * b_scale) * a_scale;
                "vfmacc.vf    v2, f0, v31             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");

#endif
        }
    } else {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                      n * k_blks * blk_len / 2 +      // b data
                                      n * k_blks * sizeof(uint8_t) +  // b zp
                                      n * k_blks * sizeof(_Float16);  // scale
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format Version_1 (FP32 SCALE FOR Normal VMADOTins of IME2)
            // A M1K32 int8    256bit
            // Ascale fp32 * 1  32bit
            // || scl*1(fp32) | Asum(int16) | blk0 || scl*1(fp32) | Asum(int16) | blk0 || ...
            // || Element                          || Element                          || ...
            // B format
            // B N8K32 int4    1024bit
            //   4VRF, N32K32, 4096bit
            // Bscale fp16 * N32 512bit;
            // Bzp uint8_t * N32 256bit;
            // || scl*32..(fp16) | zp*32(uint8) | blk0 blk1 ... blk31 || scl*32..(fp16)  ...
            // || Element                                             || Element         ...

            //bias always be nullptr
#if 1
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*3         \n\t"  // s5 = pBdata, (pB+BScl+Bzp)
                "mv           s6, %[pC]               \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc

                // ordinary vmadot: vle*6 flw*1 vecIns*21 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+96        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v0, (s4)                \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64              \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, zero, e8, m1        \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "smt.vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "smt.vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "smt.vmadotsu     v16, v10, v4, i4        \n\t"  // M0 N0 - N7 INT32(256bit)
                "smt.vmadotsu     v18, v10, v5, i4        \n\t"  // M0 N8 - N15
                "smt.vmadotsu     v20, v10, v6, i4        \n\t"  // M0 N16 - N23
                "smt.vmadotsu     v22, v10, v7, i4        \n\t"  // M0 N24 - N31

                "vsll.vi      v16, v16, 4             \n\t"
                "vsll.vi      v18, v18, 4             \n\t"
                "vsll.vi      v20, v20, 4             \n\t"
                "vsll.vi      v22, v22, 4             \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vle8.v       v1, (s4)                \n\t"  // Bzp
                "addi         s4, s4, 32+128*4        \n\t"

                "smt.vmadotu      v16, v8, v4, i4         \n\t"
                "smt.vmadotu      v18, v8, v5, i4         \n\t"
                "smt.vmadotu      v20, v8, v6, i4         \n\t"
                "smt.vmadotu      v22, v8, v7, i4         \n\t"

                "vwaddu.vx    v28, v1, x0             \n\t"  // uint8 -> uint16
                "smt.vpack.vv     v24, v16, v18, 2        \n\t"
                "smt.vpack.vv     v26, v20, v22, 2        \n\t"
                "smt.vpack.vv     v16, v24, v26, 3        \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vwmul.vx     v24, v28, t2            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vadd.vv      v16, v16, v24           \n\t"

                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v24, v0                 \n\t"
                // mac result i32 -> fp32
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v16                \n\t"
                // a_scale * b_scale;
                "vfmul.vf     v1, v24, f0             \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v2, v1, v26             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#else
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // init the scale
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*3         \n\t"  // s5 = pBdata, (pB+BScl+Bzp)
                "mv           s6, %[pC]               \n\t"

                "vsll.vi      v1, v0, 4               \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc
                "vfcvt.f.x.v  v0, v0                  \n\t"
                "vfcvt.f.x.v  v1, v1                  \n\t"

                // vmadot hp: vle*6 flw*1 vecIns*14 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+96        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v30, (s4)               \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64              \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v31, (s4)               \n\t"  // B zp 32Row*uint8 = 256bit
                "addi         s4, s4, 32+128*4        \n\t"

                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "smt.vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "smt.vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "smt.vmadotsu.hp  v16, v10, v4, v1, 0, i4 \n\t"  // high 4
                "smt.vmadotsu.hp  v18, v10, v5, v1, 0, i4 \n\t"
                "smt.vmadotsu.hp  v20, v10, v6, v1, 0, i4 \n\t"
                "smt.vmadotsu.hp  v22, v10, v7, v1, 0, i4 \n\t"
                "smt.vmadotu.hp   v16, v8, v4, v0, 0, i4  \n\t"  // low 4
                "smt.vmadotu.hp   v18, v8, v5, v0, 0, i4  \n\t"
                "smt.vmadotu.hp   v20, v8, v6, v0, 0, i4  \n\t"
                "smt.vmadotu.hp   v22, v8, v7, v0, 0, i4  \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vwaddu.vx    v28, v31, x0            \n\t"  // Bzp u8 -> u16

                "vsetvli      t0, x0, e8, m1          \n\t"
                "smt.vpack.vv     v24, v16, v18, 1        \n\t"
                "smt.vpack.vv     v26, v20, v22, 1        \n\t"
                "smt.vpack.vv     v16, v24, v26, 2        \n\t"

                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vmul.vx      v26, v28, t2            \n\t"  // asum*zp i16*i16
                "vfwcvt.f.f.v v22, v30                \n\t"  // b_scale fp16 -> fp32
                "vfcvt.f.x.v  v18, v26                \n\t"  // zp i16 -> fp16
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vfwadd.vv    v20, v18, v16           \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // mac result * b_scale; f32*f32->f32
                "vfmul.vv     v31, v22, v20           \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum * b_scale) * a_scale;
                "vfmacc.vf    v2, f0, v31             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#endif
        }
    }
}
void gemm_kernel_i8i4_m4(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    int64_t b_data_stride =
        k_blks * (sizeof(ggml_fp16_t) + 16 * sizeof(int8_t) + (quant_b_zp != NULL ? sizeof(int8_t) : 0));
    if (quant_b_zp == NULL) {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data = (uint8_t *) quant_b_data + ni * b_data_stride;
            int8_t *  a_data = (int8_t *) quant_a_ptr;
            float *   dst_c  = c_ptr + ni;
#if 1
            asm volatile(
                "li             t1,  8              \n\t"
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vxor.vv        v28, v28, v28       \n\t"
                "vxor.vv        v29, v29, v29       \n\t"
                "vxor.vv        v30, v30, v30       \n\t"
                "vxor.vv        v31, v31, v31       \n\t"
                "mv             t4, %[BK]           \n\t"

                ".align 4                           \n\t"
                "BLK_LOOP%=:                        \n\t"
                // load scale A
                "flw            fa0, (%[A])         \n\t"
                "flw            fa1, 4(%[A])        \n\t"
                "flw            fa2, 8(%[A])        \n\t"
                "flw            fa3, 12(%[A])       \n\t"
                "addi           %[A], %[A], 16      \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2    \n\t"
                "vle16.v        v12, (%[B])         \n\t"
                "addi           %[B], %[B], 64      \n\t"
                "vfwcvt.f.f.v   v14, v12            \n\t"

                "vsetivli       t0, 4, e16, mf2     \n\t"
                "vle16.v        v8, (%[A])          \n\t"  // asum
                "addi           %[A], %[A], 8       \n\t"
                "vwmul.vx       v10, v8, t1         \n\t"  // 8*asum

                "vsetvli        t0, x0, e8, m1      \n\t"
                "vl1r.v         v0, (%[A])          \n\t"
                "addi           %[A], %[A], 128     \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])          \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512     \n\t"
                "vsrl.vi        v1, v0, 4           \n\t"
                "smt.vnpack4.vv     v12, v0, v1, 3      \n\t"  // A low  u4
                "smt.vupack.vv      v2, v12, v12, 2     \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vxor.vv        v16, v16, v16       \n\t"
                "vxor.vv        v18, v16, v16       \n\t"
                "vxor.vv        v20, v16, v16       \n\t"
                "vxor.vv        v22, v16, v16       \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e32, m1     \n\t"
                "smt.vmadotsu       v16, v3, v4, i4     \n\t"   // high 4
                "smt.vmadotsu       v18, v3, v5, i4     \n\t"
                "smt.vmadotsu       v20, v3, v6, i4     \n\t"
                "smt.vmadotsu       v22, v3, v7, i4     \n\t"
                "vsll.vi        v16, v16, 4         \n\t"
                "vsll.vi        v18, v18, 4         \n\t"
                "vsll.vi        v20, v20, 4         \n\t"
                "vsll.vi        v22, v22, 4         \n\t"
                "smt.vmadotu        v16, v2, v4, i4     \n\t"   // low 4
                "smt.vmadotu        v18, v2, v5, i4     \n\t"
                "smt.vmadotu        v20, v2, v6, i4     \n\t"
                "smt.vmadotu        v22, v2, v7, i4     \n\t"

                "smt.vpack.vv       v0, v16, v18, 2     \n\t"
                "smt.vpack.vv       v2, v20, v22, 2     \n\t"
                "smt.vpack.vv       v16, v0, v2, 3      \n\t"
                "smt.vpack.vv       v18, v1, v3, 3      \n\t"

                "vrgather.vi    v0, v10, 0          \n\t"
                "vrgather.vi    v1, v10, 1          \n\t"
                "vrgather.vi    v2, v10, 2          \n\t"
                "vrgather.vi    v3, v10, 3          \n\t"

                "vadd.vv        v16, v16, v0        \n\t"
                "vadd.vv        v17, v17, v1        \n\t"
                "vadd.vv        v18, v18, v2        \n\t"
                "vadd.vv        v19, v19, v3        \n\t"

                "vfcvt.f.x.v    v16, v16            \n\t"
                "vfcvt.f.x.v    v17, v17            \n\t"
                "vfcvt.f.x.v    v18, v18            \n\t"
                "vfcvt.f.x.v    v19, v19            \n\t"

                // mul scale
                "vfmul.vv       v16, v16, v14       \n\t"
                "vfmul.vv       v17, v17, v14       \n\t"
                "vfmul.vv       v18, v18, v14       \n\t"
                "vfmul.vv       v19, v19, v14       \n\t"

                "addi           t4, t4, -1          \n\t"
                "vfmacc.vf      v28, fa0, v16       \n\t"
                "vfmacc.vf      v29, fa1, v17       \n\t"
                "vfmacc.vf      v30, fa2, v18       \n\t"
                "vfmacc.vf      v31, fa3, v19       \n\t"

                "bgtz           t4, BLK_LOOP%=      \n\t"

                // save
                "vsetvli        t0, x0, e32, m1     \n\t"
                "add            t2, %[LDC], %[DST]  \n\t"
                "vse32.v        v28, (%[DST])       \n\t"
                "add            t3, %[LDC], t2      \n\t"
                "vse32.v        v29, (t2)           \n\t"
                "add            t2, %[LDC], t3      \n\t"
                "vse32.v        v30, (t3)           \n\t"
                "vse32.v        v31, (t2)           \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc*4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                  "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
#else
            asm volatile(
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v29, v29, v29           \n\t"
                "vxor.vv        v30, v30, v30           \n\t"
                "vxor.vv        v31, v31, v31           \n\t"
                "vmv.v.i        v0, 1                   \n\t"  // init the scale
                "vsll.vi        v1, v0, 4               \n\t"
                "vfcvt.f.x.v    v0, v0                  \n\t"
                "vfcvt.f.x.v    v1, v1                  \n\t"
                "mv             t4, %[BK]               \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // load scale A
                "flw            fa0, (%[A])             \n\t"
                "flw            fa1, 4(%[A])            \n\t"
                "flw            fa2, 8(%[A])            \n\t"
                "flw            fa3, 12(%[A])           \n\t"
                "addi           %[A], %[A], 16          \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vle16.v        v12, (%[B])             \n\t"
                "addi           %[B], %[B], 64          \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "smt.vpack.vv       v14, v12, v12, 3        \n\t"

                "vsetivli       t0, 4, e16, mf2         \n\t"
                "vle16.v        v8, (%[A])              \n\t"  // asum
                "addi           %[A], %[A], 8           \n\t"
                "vsll.vi        v8, v8, 3               \n\t"  // asum * 8
                "vfcvt.f.x.v    v9, v8                  \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vrgather.vi    v10, v9, 0              \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vl1r.v         v16, (%[A])             \n\t"
                "addi           %[A], %[A], 128         \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])              \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512         \n\t"
                "vsrl.vi        v17, v16, 4             \n\t"
                "smt.vnpack4.vv     v12, v16, v17, 3        \n\t"  // A low  u4
                "smt.vupack.vv      v2, v12, v12, 2         \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e16, m1         \n\t"
                "smt.vpack.vv       v16, v10, v10,0         \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "smt.vpack.vv       v20, v16, v16,0         \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "smt.vpack.vv       v18, v20, v20, 0        \n\t"
                "vor.vv         v20, v18, v18           \n\t"
                "vor.vv         v21, v18, v18           \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e16, m1         \n\t"
                "smt.vmadotsu.hp    v18, v3, v4, v1, 0, i4  \n\t"  // high 4
                "smt.vmadotsu.hp    v19, v3, v5, v1, 0, i4  \n\t"
                "smt.vmadotsu.hp    v20, v3, v6, v1, 0, i4  \n\t"
                "smt.vmadotsu.hp    v21, v3, v7, v1, 0, i4  \n\t"
                "smt.vmadotu.hp     v18, v2, v4, v0, 0, i4  \n\t"  // low 4
                "smt.vmadotu.hp     v19, v2, v5, v0, 0, i4  \n\t"
                "smt.vmadotu.hp     v20, v2, v6, v0, 0, i4  \n\t"
                "smt.vmadotu.hp     v21, v2, v7, v0, 0, i4  \n\t"

                "smt.vpack.vv       v8, v18, v19, 1         \n\t"
                "smt.vpack.vv       v12, v20, v21, 1        \n\t"
                "smt.vpack.vv       v20, v8, v12, 2         \n\t"

                "vfwmul.vv      v16, v20, v14           \n\t"
                "vfwmul.vv      v18, v21, v14           \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"

                "addi           t4, t4, -1              \n\t"
                "vfmacc.vf      v28, fa0, v16           \n\t"
                "vfmacc.vf      v29, fa1, v17           \n\t"
                "vfmacc.vf      v30, fa2, v18           \n\t"
                "vfmacc.vf      v31, fa3, v19           \n\t"

                "bgtz           t4, BLK_LOOP%=          \n\t"

                // save
                "vsetvli        t0, x0, e32, m1         \n\t"
                "add            t2, %[LDC], %[DST]      \n\t"
                "vse32.v        v28, (%[DST])           \n\t"
                "add            t3, %[LDC], t2          \n\t"
                "vse32.v        v29, (t2)               \n\t"
                "add            t2, %[LDC], t3          \n\t"
                "vse32.v        v30, (t3)               \n\t"
                "vse32.v        v31, (t2)               \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                  "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
#endif
        }
    } else {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data = (uint8_t *) quant_b_data + ni * b_data_stride;
            int8_t *  a_data = (int8_t *) quant_a_ptr;
            float *   dst_c  = c_ptr + ni;

            asm volatile(
                "li             t1,  8          \n\t"
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vxor.vv        v28, v28, v28   \n\t"
                "vxor.vv        v29, v29, v29   \n\t"
                "vxor.vv        v30, v30, v30   \n\t"
                "vxor.vv        v31, v31, v31   \n\t"
                "mv             t4, %[BK]       \n\t"

                ".align 4                        \n\t"
                "BLK_LOOP%=:                     \n\t"
                // load scale A
                "flw            fa0, (%[A])     \n\t"
                "flw            fa1, 4(%[A])    \n\t"
                "flw            fa2, 8(%[A])    \n\t"
                "flw            fa3, 12(%[A])   \n\t"
                "addi           %[A], %[A], 16  \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2\n\t"
                "vle16.v        v12, (%[B])     \n\t"
                "addi           %[B], %[B], 64  \n\t"
                "vfwcvt.f.f.v   v14, v12        \n\t"

                // load zp
                "vsetvli        t0, x0, e8, mf4 \n\t"
                "vle8.v         v8, (%[B])      \n\t"
                "addi           %[B], %[B], 32  \n\t"
                "vwaddu.vx      v10, v8, x0     \n\t"

                // load a sum
                "lh             s1, (%[A])      \n\t"
                "lh             s2, 2(%[A])     \n\t"
                "lh             s3, 4(%[A])     \n\t"
                "lh             s4, 6(%[A])     \n\t"
                "addi           %[A], %[A], 8   \n\t"

                "vsetvli        t0, x0, e8, m1  \n\t"
                "vl1r.v         v0, (%[A])      \n\t"
                "addi           %[A], %[A], 128 \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])      \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512 \n\t"
                "vsrl.vi        v1, v0, 4       \n\t"
                "smt.vnpack4.vv     v12, v0, v1, 3  \n\t"  // A low  u4
                "smt.vupack.vv      v2, v12, v12, 2 \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vxor.vv        v16, v16, v16   \n\t"
                "vxor.vv        v18, v16, v16   \n\t"
                "vxor.vv        v20, v16, v16   \n\t"
                "vxor.vv        v22, v16, v16   \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e32, m1 \n\t"
                "smt.vmadotsu       v16, v3, v4, i4 \n\t"  // high 4
                "smt.vmadotsu       v18, v3, v5, i4 \n\t"
                "smt.vmadotsu       v20, v3, v6, i4 \n\t"
                "smt.vmadotsu       v22, v3, v7, i4 \n\t"
                "vsll.vi        v16, v16, 4     \n\t"
                "vsll.vi        v18, v18, 4     \n\t"
                "vsll.vi        v20, v20, 4     \n\t"
                "vsll.vi        v22, v22, 4     \n\t"
                "smt.vmadotu        v16, v2, v4, i4 \n\t"  // low 4
                "smt.vmadotu        v18, v2, v5, i4 \n\t"
                "smt.vmadotu        v20, v2, v6, i4 \n\t"
                "smt.vmadotu        v22, v2, v7, i4 \n\t"

                "smt.vpack.vv       v0, v16, v18, 2 \n\t"
                "smt.vpack.vv       v2, v20, v22, 2 \n\t"
                "smt.vpack.vv       v16, v0, v2, 3  \n\t"
                "smt.vpack.vv       v18, v1, v3, 3  \n\t"

                "vsetvli        t0, x0, e16, m1 \n\t"
                "vwmul.vx       v0, v10, s1     \n\t"
                "vwmul.vx       v2, v10, s2     \n\t"
                "vwmul.vx       v4, v10, s3     \n\t"
                "vwmul.vx       v6, v10, s4     \n\t"

                "vsetvli        t0, x0, e32, m1 \n\t"
                "vadd.vv        v16, v16, v0    \n\t"
                "vadd.vv        v17, v17, v2    \n\t"
                "vadd.vv        v18, v18, v4    \n\t"
                "vadd.vv        v19, v19, v6    \n\t"

                "vfcvt.f.x.v    v16, v16        \n\t"
                "vfcvt.f.x.v    v17, v17        \n\t"
                "vfcvt.f.x.v    v18, v18        \n\t"
                "vfcvt.f.x.v    v19, v19        \n\t"

                // mul scale
                "vfmul.vv       v16, v16, v14   \n\t"
                "vfmul.vv       v17, v17, v14   \n\t"
                "vfmul.vv       v18, v18, v14   \n\t"
                "vfmul.vv       v19, v19, v14   \n\t"

                "addi           t4, t4, -1      \n\t"
                "vfmacc.vf      v28, fa0, v16   \n\t"
                "vfmacc.vf      v29, fa1, v17   \n\t"
                "vfmacc.vf      v30, fa2, v18   \n\t"
                "vfmacc.vf      v31, fa3, v19   \n\t"

                "bgtz           t4, BLK_LOOP%=  \n\t"

                // save
                "vsetvli        t0, x0, e32, m1 \n\t"
                "add            t2, %[LDC], %[DST]\n\t"
                "vse32.v        v28, (%[DST])   \n\t"
                "add            t3, %[LDC], t2  \n\t"
                "vse32.v        v29, (t2)       \n\t"
                "add            t2, %[LDC], t3  \n\t"
                "vse32.v        v30, (t3)       \n\t"
                "vse32.v        v31, (t2)       \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                  "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3", "s1", "s2", "s3", "s4");
        }
    }
}
size_t gemm_kernel_i8i4(size_t          blk_len,
                        const uint8_t * quant_a_ptr,
                        const uint8_t * quant_b_data,
                        const uint8_t * quant_b_zp,
                        float *         c_ptr,
                        size_t          count_m,
                        size_t          count_n,
                        size_t          k_blks,
                        size_t          ldc) {
    if (count_m >= 4) {
#if 0
        gemm_kernel_i8i4_mrow_ref<4, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                         k_blks, ldc);
#else
        gemm_kernel_i8i4_m4(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 4;
    } else {
#if 0
        gemm_kernel_i8i4_mrow_ref<1, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                         k_blks, ldc);
#else
        gemm_kernel_i8i4_m1(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 1;
    }
}


}  // namespace ime2
}  // namespace spacemit_kernels
