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
