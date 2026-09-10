.section .text

# a0=sweeps, a1=A packed qs (60*48*256), a2=W unpacked (96*48*512), a3=output.
# Every sweep executes 3 * (28+28+4) * 96 * 16 * 2 = 552,960 inner bodies.
.macro PROLOGUE
    addi sp, sp, -80
    sd s0, 0(sp)
    sd s1, 8(sp)
    sd s2, 16(sp)
    sd s3, 24(sp)
    sd s4, 32(sp)
    sd s5, 40(sp)
    sd s6, 48(sp)
    sd s7, 56(sp)
    sd s8, 64(sp)
    sd s9, 72(sp)
    mv s0, a0
    mv s1, a1
    mv s2, a2
    mv s3, a3
    vsetivli zero, 16, e32, m2, ta, ma
    vmv.v.i v6, 0
    vmv.v.i v30, 0
    vmv.v.i v28, 0
    vmv.v.i v26, 0
    vmv.v.i v24, 0
    vmv.v.i v22, 0
    vmv.v.i v20, 0
    vmv.v.i v18, 0
.endm

.macro EPILOGUE
    vs2r.v v6, (s3)
    addi t0, s3, 64
    vs2r.v v30, (t0)
    addi t0, s3, 128
    vs2r.v v28, (t0)
    addi t0, s3, 192
    vs2r.v v26, (t0)
    addi t0, s3, 256
    vs2r.v v24, (t0)
    addi t0, s3, 320
    vs2r.v v22, (t0)
    addi t0, s3, 384
    vs2r.v v20, (t0)
    addi t0, s3, 448
    vs2r.v v18, (t0)
    ld s0, 0(sp)
    ld s1, 8(sp)
    ld s2, 16(sp)
    ld s3, 24(sp)
    ld s4, 32(sp)
    ld s5, 40(sp)
    ld s6, 48(sp)
    ld s7, 56(sp)
    ld s8, 64(sp)
    ld s9, 72(sp)
    addi sp, sp, 80
    ret
.endm

# Exact arithmetic body of production .LBB1_5. t0=W inner address, t1=A inner address.
.macro INNER reordered=0
    vl8r.v v8, (t0)
    vsetvli t2, zero, e8, m1, ta, ma
    vle8.v v5, (t1)
    addi t2, t1, 32
    vle8.v v4, (t2)
    addi t2, t1, 64
    vle8.v v3, (t2)
    addi t2, t1, 96
    vle8.v v2, (t2)
    vsetivli zero, 0, e8, m1, tu, ma
.if \reordered
    smt.vmadot v6, v5, v8
    smt.vmadot v30, v5, v9
    smt.vmadot v28, v5, v10
    smt.vmadot v26, v5, v11
    smt.vmadot v24, v3, v8
    smt.vmadot v22, v3, v9
    smt.vmadot v20, v3, v10
    smt.vmadot v18, v3, v11
    smt.vmadot v6, v4, v12
    smt.vmadot v30, v4, v13
    smt.vmadot v28, v4, v14
    smt.vmadot v26, v4, v15
    smt.vmadot v24, v2, v12
    smt.vmadot v22, v2, v13
    smt.vmadot v20, v2, v14
    smt.vmadot v18, v2, v15
.else
    smt.vmadot v6, v5, v8
    smt.vmadot v30, v5, v9
    smt.vmadot v28, v5, v10
    smt.vmadot v26, v5, v11
    smt.vmadot v6, v4, v12
    smt.vmadot v30, v4, v13
    smt.vmadot v28, v4, v14
    smt.vmadot v26, v4, v15
    smt.vmadot v24, v3, v8
    smt.vmadot v22, v3, v9
    smt.vmadot v20, v3, v10
    smt.vmadot v18, v3, v11
    smt.vmadot v24, v2, v12
    smt.vmadot v22, v2, v13
    smt.vmadot v20, v2, v14
    smt.vmadot v18, v2, v15
.endif
.endm

# Production loop order: panel -> chunk(28/28/4) -> N96 -> M in chunk -> bk16 -> inner2.
.macro DEFINE name,fixed_a=0,fixed_w=0,reordered=0,prefetch_lines=0
.p2align 6
.global \name
.type \name,@function
\name:
    PROLOGUE
.Lsweep_\name:
    li s4, 0
.Lpanel_\name:
    li s5, 0
.Lchunk_\name:
    li t3, 28
    li t4, 56
    beq s5, t4, .Llast_chunk_\name
    j .Lchunk_ready_\name
.Llast_chunk_\name:
    li t3, 4
.Lchunk_ready_\name:
    li s6, 0
.Ln_\name:
    li s7, 0
.Lm_\name:
    add t5, s5, s7
.if \prefetch_lines
    # Once per N panel, four M-tile kernel calls before the switch, request
    # the selected prefix of the next N tile's W panel.
    addi t4, t3, -4
    bne s7, t4, .Lprefetch_done_\name
    li t4, 95
    bgeu s6, t4, .Lprefetch_done_\name
    addi t4, s6, 1
    li t6, 48
    mul t4, t4, t6
    slli t6, s4, 4
    add t4, t4, t6
    slli t4, t4, 9
    add t4, s2, t4
    li t6, \prefetch_lines
.Lprefetch_loop_\name:
    prefetch.r 0(t4)
    addi t4, t4, 64
    addi t6, t6, -1
    bnez t6, .Lprefetch_loop_\name
.Lprefetch_done_\name:
.endif
    li s8, 0
.Lbk_\name:
    li s9, 0
.Linner_\name:
.if \fixed_a
    mv t1, s1
.else
    # A block = panel*960 + tile_m*16 + bk; each block is 256 B.
    li t6, 960
    mul t6, s4, t6
    slli t0, t5, 4
    add t6, t6, t0
    add t6, t6, s8
    slli t6, t6, 8
    add t1, s1, t6
.endif
    slli t6, s9, 7
    add t1, t1, t6
.if \fixed_w
    mv t0, s2
.else
    # W block = n*48 + panel*16 + bk; each block is 512 B.
    li t6, 48
    mul t0, s6, t6
    slli t6, s4, 4
    add t0, t0, t6
    add t0, t0, s8
    slli t0, t0, 9
    add t0, s2, t0
.endif
    slli t6, s9, 8
    add t0, t0, t6
    INNER \reordered
    addi s9, s9, 1
    li t6, 2
    bltu s9, t6, .Linner_\name
    addi s8, s8, 1
    li t6, 16
    bltu s8, t6, .Lbk_\name
    addi s7, s7, 1
    bltu s7, t3, .Lm_\name
    addi s6, s6, 1
    li t6, 96
    bltu s6, t6, .Ln_\name
    addi s5, s5, 28
    li t6, 60
    bltu s5, t6, .Lchunk_\name
    addi s4, s4, 1
    li t6, 3
    bltu s4, t6, .Lpanel_\name
    addi s0, s0, -1
    bnez s0, .Lsweep_\name
    EPILOGUE
.size \name,.-\name
.endm

DEFINE dpu_cb_issue_hot,1,1,0
DEFINE dpu_cb_issue_production,0,0,0
DEFINE dpu_cb_issue_production_fixed_w,0,1,0
DEFINE dpu_cb_issue_production_fixed_a,1,0,0
DEFINE dpu_cb_issue_production_reordered,0,0,1
DEFINE dpu_cb_issue_production_prefetch_512,0,0,0,8
DEFINE dpu_cb_issue_production_prefetch_2048,0,0,0,32
DEFINE dpu_cb_issue_production_prefetch_8192,0,0,0,128
