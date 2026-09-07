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

# a0=A chunk (28 tiles * 16 blocks * 288 B = 129024 B),
# a1=hot 8 KiB B panel, a2=passes, a3=output.
# Every pass streams the complete L2-resident A chunk. For every A tile, the
# same L1-resident B panel is scanned from its beginning (16 blocks, 2 inners).
# mode: 0=A-only, 1=B-only, 2=A+real-B, 3=A+fixed-hot-B.
.macro DPU_CASE name,mode,astride,aoffset
FN \name
  addi sp,sp,-48
  sd s0,0(sp)
  sd s1,8(sp)
  sd s2,16(sp)
  sd s3,24(sp)
  sd s4,32(sp)
  sd s5,40(sp)
  mv s0,a0
  mv s1,a1
  mv s2,a2
  mv s3,a3
  li s5,0
  vsetvli t5,zero,e8,m1,ta,ma
  .p2align 6
.Lpass_\@:
  mv s4,s0
  li t4,0
.Ltile_\@:
  li t0,0
.Lblock_\@:
  slli t1,t0,9
  add t2,s1,t1
.if \mode == 3
  mv t2,s1
.endif

  # Inner 0: B[block*512 + 0], four contiguous 32-byte A loads.
.if \mode != 0
  vl8r.v v8,(t2)
.else
  nop
.endif
.if \mode != 1
  addi t3,s4,\aoffset
  vle8.v v2,(t3)
  addi t3,s4,\aoffset+32
  vle8.v v3,(t3)
  addi t3,s4,\aoffset+64
  vle8.v v4,(t3)
  addi t3,s4,\aoffset+96
  vle8.v v5,(t3)
.else
  nop; nop; nop; nop
.endif
  .rept 16
    nop
  .endr

  # Inner 1: B[block*512 + 256], next four contiguous A loads.
.if \mode != 0
  addi t2,t2,256
  vl8r.v v8,(t2)
.else
  nop
.endif
.if \mode != 1
  addi t3,s4,\aoffset+128
  vle8.v v2,(t3)
  addi t3,s4,\aoffset+160
  vle8.v v3,(t3)
  addi t3,s4,\aoffset+192
  vle8.v v4,(t3)
  addi t3,s4,\aoffset+224
  vle8.v v5,(t3)
.else
  nop; nop; nop; nop
.endif
  .rept 16
    nop
  .endr

  addi s4,s4,\astride
  addi t0,t0,1
  li t1,16
  bltu t0,t1,.Lblock_\@
  addi t4,t4,1
  li t1,28
  bltu t4,t1,.Ltile_\@
  addi s5,s5,1
  bltu s5,s2,.Lpass_\@
  vs1r.v v2,(s3)
  addi t1,s3,32
  vs1r.v v8,(t1)
  ld s0,0(sp)
  ld s1,8(sp)
  ld s2,16(sp)
  ld s3,24(sp)
  ld s4,32(sp)
  ld s5,40(sp)
  addi sp,sp,48
  ret
END \name
.endm

DPU_CASE dpu_cb_a_only,0,288,32
DPU_CASE dpu_cb_b_only,1,288,32
DPU_CASE dpu_cb_mixed,2,288,32
DPU_CASE dpu_cb_fixed_b,3,288,32
DPU_CASE dpu_cb_packed_a_only,0,256,0
DPU_CASE dpu_cb_packed_mixed,2,256,0
