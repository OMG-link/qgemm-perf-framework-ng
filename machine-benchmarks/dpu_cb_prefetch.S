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

# B-outer sweep matching:
#   for each pass
#     for each B panel selected by the order table
#       reset A and scan all configured A tiles
# a4=number of B visits, a5=A tile count, a6=table of byte offsets,
# a7=512-byte blocks per B panel. mode 0=B-only, 1=A+B.
.macro DPU_B_OUTER_CASE name,mode
FN \name
  addi sp,sp,-96
  sd s0,0(sp)
  sd s1,8(sp)
  sd s2,16(sp)
  sd s3,24(sp)
  sd s4,32(sp)
  sd s5,40(sp)
  sd s6,48(sp)
  sd s7,56(sp)
  sd s8,64(sp)
  sd s9,72(sp)
  sd s10,80(sp)
  sd s11,88(sp)
  mv s0,a0
  mv s1,a1
  mv s2,a2
  mv s3,a3
  mv s6,a4
  mv s7,a5
  mv s10,a6
  mv s11,a7
  li s5,0
  vsetvli t5,zero,e8,m1,ta,ma
  .p2align 6
.Louter_pass_\@:
  li s9,0
.Louter_panel_\@:
  slli t0,s9,3
  add t0,s10,t0
  ld t0,0(t0)
  add s8,s1,t0
  mv s4,s0
  li t4,0
.Louter_tile_\@:
  li t0,0
.Louter_block_\@:
  slli t1,t0,9
  add t2,s8,t1

  vl8r.v v8,(t2)
.if \mode == 1
  addi t3,s4,32
  vle8.v v2,(t3)
  addi t3,s4,64
  vle8.v v3,(t3)
  addi t3,s4,96
  vle8.v v4,(t3)
  addi t3,s4,128
  vle8.v v5,(t3)
.else
  nop; nop; nop; nop
.endif
  .rept 16
    nop
  .endr

  addi t2,t2,256
  vl8r.v v8,(t2)
.if \mode == 1
  addi t3,s4,160
  vle8.v v2,(t3)
  addi t3,s4,192
  vle8.v v3,(t3)
  addi t3,s4,224
  vle8.v v4,(t3)
  addi t3,s4,256
  vle8.v v5,(t3)
.else
  nop; nop; nop; nop
.endif
  .rept 16
    nop
  .endr

  addi s4,s4,288
  addi t0,t0,1
  bltu t0,s11,.Louter_block_\@
  addi t4,t4,1
  bltu t4,s7,.Louter_tile_\@
  addi s9,s9,1
  bltu s9,s6,.Louter_panel_\@
  addi s5,s5,1
  bltu s5,s2,.Louter_pass_\@
  vs1r.v v2,(s3)
  addi t1,s3,32
  vs1r.v v8,(t1)
  ld s0,0(sp)
  ld s1,8(sp)
  ld s2,16(sp)
  ld s3,24(sp)
  ld s4,32(sp)
  ld s5,40(sp)
  ld s6,48(sp)
  ld s7,56(sp)
  ld s8,64(sp)
  ld s9,72(sp)
  ld s10,80(sp)
  ld s11,88(sp)
  addi sp,sp,96
  ret
END \name
.endm

DPU_B_OUTER_CASE dpu_cb_b_outer_b_only,0
DPU_B_OUTER_CASE dpu_cb_b_outer_mixed,1
