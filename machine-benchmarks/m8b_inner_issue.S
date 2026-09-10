.section .text

# a0=inner iterations, a1=A, a2=packed W, a3=512-byte output.
# One benchmark iteration is one .LBB0_8 inner: 128 B A + 128 B W.
.macro INIT
    vsetvli t6, zero, e8, m1, ta, ma
    vmv.v.i v4, 4
    vmv.v.i v5, 5
    vmv.v.i v6, 6
    vmv.v.i v7, 7
    vmv.v.i v24, 8
    vmv.v.i v25, 9
    vmv.v.i v26, 10
    vmv.v.i v27, 11
    vmv.v.i v28, 12
    vmv.v.i v29, 13
    vmv.v.i v30, 14
    vmv.v.i v31, 15
    vsetivli zero, 16, e32, m2, ta, ma
    vmv.v.i v8, 0
    vmv.v.i v22, 0
    vmv.v.i v20, 0
    vmv.v.i v18, 0
    vmv.v.i v16, 0
    vmv.v.i v14, 0
    vmv.v.i v12, 0
    vmv.v.i v10, 0
    li t0, 0
.endm
.macro ADDRS a_stream,w_stream
.if \w_stream
    slli t1, t0, 7
.else
    andi t1, t0, 63
    slli t1, t1, 7
.endif
    add t2, a2, t1
.if \a_stream == \w_stream
    add t3, a1, t1
.else
.if \a_stream
    slli t4, t0, 7
.else
    andi t4, t0, 63
    slli t4, t4, 7
.endif
    add t3, a1, t4
.endif
.endm


.macro STORE
    vs2r.v v22, (a3)
    addi t1, a3, 64
    vs2r.v v20, (t1)
    addi t1, a3, 128
    vs2r.v v18, (t1)
    addi t1, a3, 192
    vs2r.v v16, (t1)
    addi t1, a3, 256
    vs2r.v v14, (t1)
    addi t1, a3, 320
    vs2r.v v12, (t1)
    addi t1, a3, 384
    vs2r.v v10, (t1)
    addi t1, a3, 448
    vs2r.v v8, (t1)
    ret
.endm

# This is the compiler .LBB0_8 order with scalar registers renamed around the ABI:
# t2=W base for this inner, t3=A base, t4=temporary address, t5=W cursor.
.macro EXACT_PREFIX no_load=0 no_unpack=0
.if \no_load
    nop
.else
    mv t5, t2
.endif
    vsetvli t6, zero, e8, m1, ta, ma
.if \no_load
    nop
.else
    vle8.v v24, (t5)
.endif
.if \no_load
    nop
.else
    addi t6, t5, 32
.endif
.if \no_load
    nop
.else
    vle8.v v25, (t6)
.endif
.if \no_load
    nop
.else
    addi t6, t5, 64
.endif
.if \no_unpack
    nop
.else
    vand.vi v26, v24, 15
.endif
.if \no_load
    nop
.else
    vle8.v v27, (t6)
.endif
.if \no_load
    nop
.else
    addi t5, t5, 96
.endif
.if \no_unpack
    nop
.else
    vand.vi v28, v25, 15
.endif
.if \no_load
    nop
.else
    vle8.v v29, (t5)
.endif
.if \no_unpack
    nop; nop; nop; nop; nop; nop; nop
.else
    vand.vi v30, v27, 15
    vand.vi v31, v29, 15
    vsrl.vi v24, v24, 4
    vsrl.vi v25, v25, 4
    vsrl.vi v27, v27, 4
    vsrl.vi v29, v29, 4
    vadd.vi v26, v26, -8
.endif
.if \no_unpack
    nop; nop
.else
    vadd.vi v28, v28, -8
    vadd.vi v30, v30, -8
.endif
.if \no_load
    nop
.else
    mv t4, t3
.endif
.if \no_load
    nop
.else
    vle8.v v7, (t4)
.endif
.if \no_unpack
    nop
.else
    vadd.vi v31, v31, -8
.endif
.if \no_load
    nop
.else
    addi t5, t4, 32
.endif
.if \no_unpack
    nop
.else
    vadd.vi v24, v24, -8
.endif
.if \no_load
    nop
.else
    vle8.v v6, (t5)
.endif
.if \no_unpack
    nop
.else
    vadd.vi v25, v25, -8
.endif
.if \no_load
    nop
.else
    addi t5, t4, 64
.endif
.if \no_load
    nop
.else
    vle8.v v5, (t5)
.endif
.if \no_unpack
    nop
.else
    vadd.vi v27, v27, -8
.endif
.if \no_load
    nop
.else
    addi t5, t4, 96
.endif
.if \no_unpack
    nop
.else
    vadd.vi v29, v29, -8
.endif
.if \no_load
    nop
.else
    vle8.v v4, (t5)
.endif
    vsetivli zero, 0, e8, m1, tu, ma
.endm
.macro DOTS_EXACT
    smt.vmadot v22, v7, v26
    smt.vmadot v20, v7, v28
    smt.vmadot v18, v7, v30
    smt.vmadot v16, v7, v31
    smt.vmadot v22, v6, v24
    smt.vmadot v20, v6, v25
    smt.vmadot v18, v6, v27
    smt.vmadot v16, v6, v29
    smt.vmadot v14, v5, v26
    smt.vmadot v12, v5, v28
    smt.vmadot v10, v5, v30
    smt.vmadot v8, v5, v31
    smt.vmadot v14, v4, v24
    smt.vmadot v12, v4, v25
    smt.vmadot v10, v4, v27
    smt.vmadot v8, v4, v29
.endm
.macro DOTS_REORDERED
    smt.vmadot v22, v7, v26
    smt.vmadot v20, v7, v28
    smt.vmadot v18, v7, v30
    smt.vmadot v16, v7, v31
    smt.vmadot v14, v5, v26
    smt.vmadot v12, v5, v28
    smt.vmadot v10, v5, v30
    smt.vmadot v8, v5, v31
    smt.vmadot v22, v6, v24
    smt.vmadot v20, v6, v25
    smt.vmadot v18, v6, v27
    smt.vmadot v16, v6, v29
    smt.vmadot v14, v4, v24
    smt.vmadot v12, v4, v25
    smt.vmadot v10, v4, v27
    smt.vmadot v8, v4, v29
.endm
.macro FINISH_ITER label
    addi t0, t0, 1
    bltu t0, a0, \label
    STORE
.endm

.macro DEFINE_CASE name,a_stream=0,w_stream=0,reordered=0,no_load=0,no_unpack=0,no_dots=0
.p2align 6
.global \name
.type \name,@function
\name:
    INIT
.Lloop_\name:
    ADDRS \a_stream,\w_stream
    EXACT_PREFIX \no_load,\no_unpack
.if \no_dots
    .rept 16
      nop
    .endr
.elseif \reordered
    DOTS_REORDERED
.else
    DOTS_EXACT
.endif
    FINISH_ITER .Lloop_\name
.size \name,.-\name
.endm

DEFINE_CASE m8b_exact_hot,0,0,0,0,0,0
DEFINE_CASE m8b_exact_reordered,0,0,1,0,0,0
DEFINE_CASE m8b_no_load,0,0,0,1,0,0
# Exact register topology makes retained W loads overwrite v24/v25/v27/v29.
# Therefore this auditable no-unpack variant preinitializes low sources
# v26/v28/v30/v31, while high-source dots consume the raw loaded packed bytes.
DEFINE_CASE m8b_no_unpack,0,0,0,0,1,0
DEFINE_CASE m8b_no_dots,0,0,0,0,0,1
DEFINE_CASE m8b_stream_exact,1,1,0,0,0,0
DEFINE_CASE m8b_stream_a_fixed_w,1,0,0,0,0,0
DEFINE_CASE m8b_fixed_a_stream_w,0,1,0,0,0,0
