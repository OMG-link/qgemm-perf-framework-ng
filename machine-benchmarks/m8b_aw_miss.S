.option norvc
.text

# a0=config {first,second,first_n_stride,second_n_stride}, a1=passes, a2=out.
# Exactly 96*48*2 inners/pass. Every inner has four contiguous 32-byte loads
# from each logical stream. Baseline retains the original shared 16 NOPs;
# probe redistributes the same budget into distinct 4-NOP first and 12-NOP
# second skid windows, so attribution does not add instructions per inner.
.macro TRACE_BODY name, probed
.p2align 8
.global \name
.type \name,@function
\name:
  addi sp,sp,-80
  sd s0,0(sp); sd s1,8(sp); sd s2,16(sp); sd s3,24(sp); sd s4,32(sp)
  sd s5,40(sp); sd s6,48(sp); sd s7,56(sp); sd s8,64(sp); sd s9,72(sp)
  ld s0,0(a0)
  ld s1,8(a0)
  ld s2,16(a0)
  ld s3,24(a0)
  mv s4,a1
  mv s5,a2
  li s6,0
  vsetvli t6,zero,e8,m1,ta,ma
.p2align 6
.Lpass\@:
  li s7,0
  mv s8,s0
  mv s9,s1
.Ln\@:
  li t0,0
.Lblock\@:
  slli t1,t0,8
  add t2,s8,t1
  add t3,s9,t1
  li t4,0
.Linner\@:
.if \probed
  .global m8b_aw_probe_first_start
  .type m8b_aw_probe_first_start,@notype
m8b_aw_probe_first_start:
.endif
  vle8.v v0,(t2)
  addi t5,t2,32
  vle8.v v1,(t5)
  addi t5,t2,64
  vle8.v v2,(t5)
  addi t5,t2,96
  vle8.v v3,(t5)
.if \probed
  .rept 4
    nop
  .endr
  .global m8b_aw_probe_first_end
  .type m8b_aw_probe_first_end,@notype
m8b_aw_probe_first_end:
  .global m8b_aw_probe_second_start
  .type m8b_aw_probe_second_start,@notype
m8b_aw_probe_second_start:
.endif
  vle8.v v4,(t3)
  addi t5,t3,32
  vle8.v v5,(t5)
  addi t5,t3,64
  vle8.v v6,(t5)
  addi t5,t3,96
  vle8.v v7,(t5)
.if \probed
  .rept 12
    nop
  .endr
  .global m8b_aw_probe_second_end
  .type m8b_aw_probe_second_end,@notype
m8b_aw_probe_second_end:
.else
  .rept 16
    nop
  .endr
.endif
  addi t2,t2,128
  addi t3,t3,128
  addi t4,t4,1
  li t5,2
  bltu t4,t5,.Linner\@
  addi t0,t0,1
  li t5,48
  bltu t0,t5,.Lblock\@
  add s8,s8,s2
  add s9,s9,s3
  addi s7,s7,1
  li t5,96
  bltu s7,t5,.Ln\@
  addi s6,s6,1
  bltu s6,s4,.Lpass\@
  vs1r.v v0,(s5)
  addi t0,s5,32; vs1r.v v1,(t0)
  addi t0,s5,64; vs1r.v v2,(t0)
  addi t0,s5,96; vs1r.v v3,(t0)
  addi t0,s5,128; vs1r.v v4,(t0)
  addi t0,s5,160; vs1r.v v5,(t0)
  addi t0,s5,192; vs1r.v v6,(t0)
  addi t0,s5,224; vs1r.v v7,(t0)
  ld s0,0(sp); ld s1,8(sp); ld s2,16(sp); ld s3,24(sp); ld s4,32(sp)
  ld s5,40(sp); ld s6,48(sp); ld s7,56(sp); ld s8,64(sp); ld s9,72(sp)
  addi sp,sp,80
  ret
.size \name,.-\name
.endm

TRACE_BODY m8b_aw_trace, 0
TRACE_BODY m8b_aw_trace_probe, 1
