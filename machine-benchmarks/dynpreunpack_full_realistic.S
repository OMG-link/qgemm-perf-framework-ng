# Real DPU compute-body address trace for M=N=1536,K=8960.
# a0=sweeps, a1=A panel-major (280*192*288), a2=unpacked B (96*280*512), a3=checksum (512 B).
# Exactly 5,160,960 K16 inners/sweep: N96 * M192 * (17*16+8) QK32 * 2.
.macro INIT
    vsetivli zero,16,e32,m2,ta,ma
    vmv.v.i v16,0; vmv.v.i v18,0; vmv.v.i v20,0; vmv.v.i v22,0
    vmv.v.i v24,0; vmv.v.i v26,0; vmv.v.i v28,0; vmv.v.i v30,0
.endm
.macro INNER ap,bp
    vl8r.v v8,(\bp)
    vsetvli t5,zero,e8,m1,ta,ma
    vle8.v v2,(\ap); addi t6,\ap,32; vle8.v v3,(t6)
    addi t6,\ap,64; vle8.v v4,(t6); addi t6,\ap,96; vle8.v v5,(t6)
    vsetivli zero,0,e8,m1,tu,ma
    smt.vmadot v16,v2,v8; smt.vmadot v18,v2,v9; smt.vmadot v20,v2,v10; smt.vmadot v22,v2,v11
    smt.vmadot v16,v3,v12; smt.vmadot v18,v3,v13; smt.vmadot v20,v3,v14; smt.vmadot v22,v3,v15
    smt.vmadot v24,v4,v8; smt.vmadot v26,v4,v9; smt.vmadot v28,v4,v10; smt.vmadot v30,v4,v11
    smt.vmadot v24,v5,v12; smt.vmadot v26,v5,v13; smt.vmadot v28,v5,v14; smt.vmadot v30,v5,v15
.endm
.macro STORE
    vs2r.v v16,(a3); addi t0,a3,64; vs2r.v v18,(t0); addi t0,a3,128; vs2r.v v20,(t0); addi t0,a3,192; vs2r.v v22,(t0)
    addi t0,a3,256; vs2r.v v24,(t0); addi t0,a3,320; vs2r.v v26,(t0); addi t0,a3,384; vs2r.v v28,(t0); addi t0,a3,448; vs2r.v v30,(t0)
.endm
# mode=0 real; mode=1 fixed hot A+B; mode=2 real A/fixed B; mode=3 fixed A/real B.
.macro BODY name,mode
.global \name
.type \name,@function
\name:
    addi sp,sp,-96
    sd s0,0(sp); sd s1,8(sp); sd s2,16(sp); sd s3,24(sp); sd s4,32(sp); sd s5,40(sp)
    sd s6,48(sp); sd s7,56(sp); sd s8,64(sp); sd s9,72(sp); sd s10,80(sp)
    INIT
    mv s0,a0; mv s1,a1; mv s2,a2; mv s3,a3
.Lsweep\@:
    li s4,0                         # n tile
.Ln\@:
    li s5,0                         # global block_k
.Lkc\@:
    li s6,16
    li t0,272
    bne s5,t0,.Lblocksok\@
    li s6,8
.Lblocksok\@:
    # A KC base = A + block_k*192*288; B = B +(n*280+block_k)*512
.if \mode == 1
    mv s7,s1
    mv s8,s2
.else
.if \mode == 2
    li t0,55296
    mul t1,s5,t0
    add s7,s1,t1
    mv s8,s2
.else
.if \mode == 3
    mv s7,s1
    li t0,280
    mul t1,s4,t0
    add t1,t1,s5
    slli t1,t1,9
    add s8,s2,t1
.else
    li t0,55296
    mul t1,s5,t0
    add s7,s1,t1
    li t0,280
    mul t1,s4,t0
    add t1,t1,s5
    slli t1,t1,9
    add s8,s2,t1
.endif
.endif
.endif
    li s9,0                         # m tile
.Lm\@:
    li s10,0                        # local QK32 block
.Lbk\@:
    li t0,288
    mul t1,s10,t0
    add t2,s7,t1
    slli t1,s10,9
    add t3,s8,t1
    addi t2,t2,32
    INNER t2,t3
    addi t2,t2,128
    addi t3,t3,256
    INNER t2,t3
    addi s10,s10,1
    bltu s10,s6,.Lbk\@
.if \mode == 1 || \mode == 3
.else
    li t0,288
    mul t1,s6,t0
    add s7,s7,t1
.endif
    addi s9,s9,1
    li t0,192
    bltu s9,t0,.Lm\@
    add s5,s5,s6
    li t0,280
    bltu s5,t0,.Lkc\@
    addi s4,s4,1
    li t0,96
    bltu s4,t0,.Ln\@
    addi s0,s0,-1
    bnez s0,.Lsweep\@
    mv a3,s3
    STORE
    ld s0,0(sp); ld s1,8(sp); ld s2,16(sp); ld s3,24(sp); ld s4,32(sp); ld s5,40(sp)
    ld s6,48(sp); ld s7,56(sp); ld s8,64(sp); ld s9,72(sp); ld s10,80(sp); addi sp,sp,96
    ret
.size \name,.-\name
.endm

BODY dyn_real,0
BODY dyn_hot,1
BODY dyn_fixed_b,2
BODY dyn_fixed_a,3
