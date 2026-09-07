.option norvc
.text

.macro FN name
  .p2align 8
  .global \name
  .type \name,@function
\name:
.endm
.macro END name
  .size \name,.-\name
.endm

# a0=Q8 A blocks (288 B each), a1=hot 8 KiB B panel, a2=block count,
# a3=output. Each block executes two K16 inners. B wraps every 16 blocks.
# mode: 0=A-only, 1=B-only, 2=A+real-B, 3=A+fixed-hot-B.
.macro DPU_CASE name,mode
FN \name
  addi sp,sp,-32
  sd s0,0(sp)
  sd s1,8(sp)
  sd s2,16(sp)
  sd s3,24(sp)
  mv s0,a0
  mv s1,a1
  mv s2,a2
  mv s3,a3
  li t0,0
  vsetvli t5,zero,e8,m1,ta,ma
  .p2align 6
.Lblock_\@:
  andi t1,t0,15
  slli t1,t1,9
  add t2,s1,t1
.if \mode == 3
  mv t2,s1
.endif

  # Inner 0: B[panel+block*512], A[block+32..159].
.if \mode != 0
  vl8r.v v8,(t2)
.else
  nop
.endif
.if \mode != 1
  addi t3,s0,32
  vle8.v v2,(t3)
  addi t3,s0,64
  vle8.v v3,(t3)
  addi t3,s0,96
  vle8.v v4,(t3)
  addi t3,s0,128
  vle8.v v5,(t3)
.else
  nop; nop; nop; nop
.endif
  .rept 16
    nop
  .endr

  # Inner 1: B+256, A[block+160..287].
.if \mode != 0
  addi t2,t2,256
  vl8r.v v8,(t2)
.else
  nop
.endif
.if \mode != 1
  addi t3,s0,160
  vle8.v v2,(t3)
  addi t3,s0,192
  vle8.v v3,(t3)
  addi t3,s0,224
  vle8.v v4,(t3)
  addi t3,s0,256
  vle8.v v5,(t3)
.else
  nop; nop; nop; nop
.endif
  .rept 16
    nop
  .endr

  addi s0,s0,288
  addi t0,t0,1
  bltu t0,s2,.Lblock_\@
  vs1r.v v2,(s3)
  addi t1,s3,32
  vs1r.v v8,(t1)
  ld s0,0(sp)
  ld s1,8(sp)
  ld s2,16(sp)
  ld s3,24(sp)
  addi sp,sp,32
  ret
END \name
.endm

DPU_CASE dpu_cb_a_only,0
DPU_CASE dpu_cb_b_only,1
DPU_CASE dpu_cb_mixed,2
DPU_CASE dpu_cb_fixed_b,3