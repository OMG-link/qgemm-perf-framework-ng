.section .text

# All kernels: a0=iterations, a1=A (>=8192 B), a2=B (>=16384 B), a3=out (>=512 B).
# Addressing wraps every 64 iterations, so no timed iteration can leave the buffers.
.macro BEGIN
    vsetivli zero, 16, e32, m2, ta, ma
    vmv.v.i v16, 0
    vmv.v.i v18, 0
    vmv.v.i v20, 0
    vmv.v.i v22, 0
    vmv.v.i v24, 0
    vmv.v.i v26, 0
    vmv.v.i v28, 0
    vmv.v.i v30, 0
    li t0, 0
1:
    andi t1, t0, 63
    slli t2, t1, 7
    add t2, a1, t2
    slli t3, t1, 8
    add t3, a2, t3
.endm
.macro END
    addi t0, t0, 1
    bltu t0, a0, 1b
    vs2r.v v16, (a3)
    addi t4, a3, 64
    vs2r.v v18, (t4)
    addi t4, a3, 128
    vs2r.v v20, (t4)
    addi t4, a3, 192
    vs2r.v v22, (t4)
    addi t4, a3, 256
    vs2r.v v24, (t4)
    addi t4, a3, 320
    vs2r.v v26, (t4)
    addi t4, a3, 384
    vs2r.v v28, (t4)
    addi t4, a3, 448
    vs2r.v v30, (t4)
    ret
.endm
.macro LOADS
    vl8r.v v8, (t3)
    vsetvli t5, zero, e8, m1, ta, ma
    vle8.v v2, (t2)
    addi t4, t2, 32
    vle8.v v3, (t4)
    addi t4, t2, 64
    vle8.v v4, (t4)
    addi t4, t2, 96
    vle8.v v5, (t4)
    vsetivli zero, 0, e8, m1, tu, ma
.endm
.macro DOTS
    smt.vmadot v16, v2, v8
    smt.vmadot v18, v2, v9
    smt.vmadot v20, v2, v10
    smt.vmadot v22, v2, v11
    smt.vmadot v16, v3, v12
    smt.vmadot v18, v3, v13
    smt.vmadot v20, v3, v14
    smt.vmadot v22, v3, v15
    smt.vmadot v24, v4, v8
    smt.vmadot v26, v4, v9
    smt.vmadot v28, v4, v10
    smt.vmadot v30, v4, v11
    smt.vmadot v24, v5, v12
    smt.vmadot v26, v5, v13
    smt.vmadot v28, v5, v14
    smt.vmadot v30, v5, v15
.endm
.macro NOPS n
    .rept \n
    nop
    .endr
.endm

# Deterministic per-sample operand state.  The runner calls this outside the timed
# region, so removed-load cases never inherit v2-v15 from an earlier case.
.global dyn_init_operands
.type dyn_init_operands,@function
dyn_init_operands:
    vsetvli t5, zero, e8, m1, ta, ma
    vmv.v.i v2, 2
    vmv.v.i v3, 3
    vmv.v.i v4, 4
    vmv.v.i v5, 5
    vmv.v.i v8, 8
    vmv.v.i v9, 9
    vmv.v.i v10, 10
    vmv.v.i v11, 11
    vmv.v.i v12, 12
    vmv.v.i v13, 13
    vmv.v.i v14, 14
    vmv.v.i v15, 15
    ret
.size dyn_init_operands,.-dyn_init_operands

.global dyn_body_full
.type dyn_body_full,@function
dyn_body_full: BEGIN; LOADS; DOTS; END
.size dyn_body_full,.-dyn_body_full

# Same loop/address/checksum; the removed group is replaced one-for-one by scalar nops.
.global dyn_body_no_bload
.type dyn_body_no_bload,@function
dyn_body_no_bload: BEGIN; NOPS 1; vsetvli t5,zero,e8,m1,ta,ma; vle8.v v2,(t2); addi t4,t2,32; vle8.v v3,(t4); addi t4,t2,64; vle8.v v4,(t4); addi t4,t2,96; vle8.v v5,(t4); vsetivli zero,0,e8,m1,tu,ma; DOTS; END
.size dyn_body_no_bload,.-dyn_body_no_bload

.global dyn_body_no_aloads
.type dyn_body_no_aloads,@function
dyn_body_no_aloads: BEGIN; vl8r.v v8,(t3); vsetvli t5,zero,e8,m1,ta,ma; NOPS 7; vsetivli zero,0,e8,m1,tu,ma; DOTS; END
.size dyn_body_no_aloads,.-dyn_body_no_aloads

.global dyn_body_no_dots
.type dyn_body_no_dots,@function
dyn_body_no_dots: BEGIN; LOADS; NOPS 16; END
.size dyn_body_no_dots,.-dyn_body_no_dots

# Alternative for zero-instruction B extraction: retain all 16 dots but use v8 for every B operand.
.global dyn_body_b_fanout1
.type dyn_body_b_fanout1,@function
dyn_body_b_fanout1:
    BEGIN
    LOADS
    smt.vmadot v16,v2,v8; smt.vmadot v18,v2,v8; smt.vmadot v20,v2,v8; smt.vmadot v22,v2,v8
    smt.vmadot v16,v3,v8; smt.vmadot v18,v3,v8; smt.vmadot v20,v3,v8; smt.vmadot v22,v3,v8
    smt.vmadot v24,v4,v8; smt.vmadot v26,v4,v8; smt.vmadot v28,v4,v8; smt.vmadot v30,v4,v8
    smt.vmadot v24,v5,v8; smt.vmadot v26,v5,v8; smt.vmadot v28,v5,v8; smt.vmadot v30,v5,v8
    END
.size dyn_body_b_fanout1,.-dyn_body_b_fanout1

# Experiment A: one load-to-use dot per iteration and matched loop baseline.
.global dyn_dep_load_dot
.type dyn_dep_load_dot,@function
dyn_dep_load_dot: BEGIN; vsetvli t5,zero,e8,m1,ta,ma; vle8.v v2,(t2); vl8r.v v8,(t3); vsetivli zero,0,e8,m1,tu,ma; smt.vmadot v16,v2,v8; END
.size dyn_dep_load_dot,.-dyn_dep_load_dot
.global dyn_dep_load_base
.type dyn_dep_load_base,@function
dyn_dep_load_base: BEGIN; vsetvli t5,zero,e8,m1,ta,ma; nop; nop; vsetivli zero,0,e8,m1,tu,ma; nop; END
.size dyn_dep_load_base,.-dyn_dep_load_base

# One accumulator gives a loop-carried vmadot dependency; baseline preserves one issue slot.
.global dyn_dep_dot_chain
.type dyn_dep_dot_chain,@function
dyn_dep_dot_chain: BEGIN; vsetvli t5,zero,e8,m1,ta,ma; vle8.v v2,(a1); vle8.v v8,(a2); vsetivli zero,0,e8,m1,tu,ma; .rept 16; smt.vmadot v16,v2,v8; .endr; END
.size dyn_dep_dot_chain,.-dyn_dep_dot_chain
.global dyn_dep_dot_base
.type dyn_dep_dot_base,@function
dyn_dep_dot_base: BEGIN; vsetvli t5,zero,e8,m1,ta,ma; vle8.v v2,(a1); vle8.v v8,(a2); vsetivli zero,0,e8,m1,tu,ma; NOPS 16; END
.size dyn_dep_dot_base,.-dyn_dep_dot_base

# Fixed-address control for the compiler-generated intrinsic instruction shape.
.global dyn_body_intrinsic_reordered_fixed_addr
.type dyn_body_intrinsic_reordered_fixed_addr,@function
dyn_body_intrinsic_reordered_fixed_addr:
    vsetivli zero,16,e32,m2,ta,ma
    vmv.v.i v16,0; vmv.v.i v18,0; vmv.v.i v20,0; vmv.v.i v22,0
    vmv.v.i v24,0; vmv.v.i v26,0; vmv.v.i v28,0; vmv.v.i v30,0
    li t0,0
1:
    vsetvli t5,zero,e8,m8,ta,ma
    vle8.v v8,(a2)
    vsetvli t5,zero,e8,m1,ta,ma
    vle8.v v2,(a1); addi t4,a1,32; vle8.v v3,(t4)
    addi t4,a1,64; vle8.v v4,(t4); addi t4,a1,96; vle8.v v5,(t4)
    vsetivli zero,0,e8,m1,tu,ma
    smt.vmadot v16,v2,v8; smt.vmadot v18,v2,v9; smt.vmadot v20,v2,v10; smt.vmadot v22,v2,v11
    smt.vmadot v24,v4,v8; smt.vmadot v26,v4,v9; smt.vmadot v28,v4,v10; smt.vmadot v30,v4,v11
    smt.vmadot v16,v3,v12; smt.vmadot v18,v3,v13; smt.vmadot v20,v3,v14; smt.vmadot v22,v3,v15
    smt.vmadot v24,v5,v12; smt.vmadot v26,v5,v13; smt.vmadot v28,v5,v14; smt.vmadot v30,v5,v15
    addi t0,t0,1
    bltu t0,a0,1b
    vs2r.v v16,(a3); addi t4,a3,64; vs2r.v v18,(t4); addi t4,a3,128; vs2r.v v20,(t4); addi t4,a3,192; vs2r.v v22,(t4)
    addi t4,a3,256; vs2r.v v24,(t4); addi t4,a3,320; vs2r.v v26,(t4); addi t4,a3,384; vs2r.v v28,(t4); addi t4,a3,448; vs2r.v v30,(t4)
    ret
.size dyn_body_intrinsic_reordered_fixed_addr,.-dyn_body_intrinsic_reordered_fixed_addr

# Hardware address-path controls. mode: 0=fixed, 1=dummy calculations,
# 2=dependent zero offset, 3=dependent rotating offset.
.macro DEFINE_ADDR_PATH name, mode
.global \name
.type \name,@function
\name:
    vsetivli zero,16,e32,m2,ta,ma
    vmv.v.i v16,0; vmv.v.i v18,0; vmv.v.i v20,0; vmv.v.i v22,0
    vmv.v.i v24,0; vmv.v.i v26,0; vmv.v.i v28,0; vmv.v.i v30,0
    li t0,0
1:
.if \mode == 0
    mv t2,a1
    mv t3,a2
.else
.if \mode == 2
    andi t1,t0,0
.else
    andi t1,t0,63
.endif
    slli t2,t1,7
    add t2,t2,a1
    slli t3,t1,8
    add t3,t3,a2
.if \mode == 1
    mv t2,a1
    mv t3,a2
.endif
.endif
    vsetvli t5,zero,e8,m8,ta,ma
    vle8.v v8,(t3)
    vsetvli t5,zero,e8,m1,ta,ma
    vle8.v v2,(t2); addi t4,t2,32; vle8.v v3,(t4)
    addi t4,t2,64; vle8.v v4,(t4); addi t4,t2,96; vle8.v v5,(t4)
    vsetivli zero,0,e8,m1,tu,ma
    smt.vmadot v16,v2,v8; smt.vmadot v18,v2,v9; smt.vmadot v20,v2,v10; smt.vmadot v22,v2,v11
    smt.vmadot v24,v4,v8; smt.vmadot v26,v4,v9; smt.vmadot v28,v4,v10; smt.vmadot v30,v4,v11
    smt.vmadot v16,v3,v12; smt.vmadot v18,v3,v13; smt.vmadot v20,v3,v14; smt.vmadot v22,v3,v15
    smt.vmadot v24,v5,v12; smt.vmadot v26,v5,v13; smt.vmadot v28,v5,v14; smt.vmadot v30,v5,v15
    addi t0,t0,1
    bltu t0,a0,1b
    vs2r.v v16,(a3); addi t4,a3,64; vs2r.v v18,(t4); addi t4,a3,128; vs2r.v v20,(t4); addi t4,a3,192; vs2r.v v22,(t4)
    addi t4,a3,256; vs2r.v v24,(t4); addi t4,a3,320; vs2r.v v26,(t4); addi t4,a3,384; vs2r.v v28,(t4); addi t4,a3,448; vs2r.v v30,(t4)
    ret
.size \name,.-\name
.endm

DEFINE_ADDR_PATH dyn_addr_fixed,0
DEFINE_ADDR_PATH dyn_addr_dummy,1
DEFINE_ADDR_PATH dyn_addr_dep_zero,2
DEFINE_ADDR_PATH dyn_addr_dynamic,3

# Current production auto-unrolled pair: two inner iterations per outer loop.
# The 8 KiB A and 16 KiB B working sets wrap every 32 pairs and remain hot in L1.
.global dyn_body_prod_two_inner_hot_l1
.type dyn_body_prod_two_inner_hot_l1,@function
dyn_body_prod_two_inner_hot_l1:
    vsetivli zero,16,e32,m2,ta,ma
    vmv.v.i v2,0
    li t0,0
    li t6,0
1:
    vmv2r.v v6,v2
    vmv2r.v v30,v2
    vmv2r.v v28,v2
    vmv2r.v v26,v2
    vmv2r.v v24,v2
    vmv2r.v v22,v2
    vmv2r.v v20,v2
    vmv2r.v v18,v2
.global dyn_prod_two_inner_begin
.type dyn_prod_two_inner_begin,@notype
dyn_prod_two_inner_begin:
    slli t2,t0,9
    add t2,t2,a2
    vsetvli t5,zero,e8,m8,ta,ma
    vle8.v v8,(t2)
    slli t3,t0,8
    add t3,t3,a1
    vsetvli t5,zero,e8,m1,ta,ma
    vle8.v v16,(t3)
    addi t4,t3,32
    vle8.v v17,(t4)
    addi t4,t3,64
    vle8.v v5,(t4)
    addi t4,t3,96
    vle8.v v4,(t4)
    vsetivli zero,0,e8,m1,tu,ma
    smt.vmadot v6,v16,v8
    smt.vmadot v30,v16,v9
    smt.vmadot v28,v16,v10
    smt.vmadot v26,v16,v11
    smt.vmadot v24,v5,v8
    smt.vmadot v22,v5,v9
    smt.vmadot v20,v5,v10
    smt.vmadot v18,v5,v11
    smt.vmadot v6,v17,v12
    smt.vmadot v30,v17,v13
    smt.vmadot v28,v17,v14
    smt.vmadot v26,v17,v15
    smt.vmadot v24,v4,v12
    smt.vmadot v22,v4,v13
    smt.vmadot v20,v4,v14
    smt.vmadot v18,v4,v15
    addi t4,t2,256
    vsetvli t5,zero,e8,m8,ta,ma
    vle8.v v8,(t4)
    addi t4,t3,128
    vsetvli t5,zero,e8,m1,ta,ma
    vle8.v v16,(t4)
    addi t4,t3,160
    vle8.v v17,(t4)
    addi t4,t3,192
    vle8.v v5,(t4)
    addi t4,t3,224
    vle8.v v4,(t4)
    vsetivli zero,0,e8,m1,tu,ma
    smt.vmadot v6,v16,v8
    smt.vmadot v30,v16,v9
    smt.vmadot v28,v16,v10
    smt.vmadot v26,v16,v11
    smt.vmadot v24,v5,v8
    smt.vmadot v22,v5,v9
    smt.vmadot v20,v5,v10
    smt.vmadot v18,v5,v11
    smt.vmadot v6,v17,v12
    smt.vmadot v30,v17,v13
    smt.vmadot v28,v17,v14
    smt.vmadot v26,v17,v15
    smt.vmadot v24,v4,v12
    smt.vmadot v22,v4,v13
    smt.vmadot v20,v4,v14
    smt.vmadot v18,v4,v15
.global dyn_prod_two_inner_end
.type dyn_prod_two_inner_end,@notype
dyn_prod_two_inner_end:
    addi t0,t0,1
    andi t0,t0,31
    addi t6,t6,1
    bltu t6,a0,1b
    vs2r.v v6,(a3)
    addi t4,a3,64
    vs2r.v v30,(t4)
    addi t4,a3,128
    vs2r.v v28,(t4)
    addi t4,a3,192
    vs2r.v v26,(t4)
    addi t4,a3,256
    vs2r.v v24,(t4)
    addi t4,a3,320
    vs2r.v v22,(t4)
    addi t4,a3,384
    vs2r.v v20,(t4)
    addi t4,a3,448
    vs2r.v v18,(t4)
    ret
.size dyn_body_prod_two_inner_hot_l1,.-dyn_body_prod_two_inner_hot_l1