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

# a0=randomized trial-base pointer array, a1=trials, a2=stream count.
# Streams are 4160 bytes apart so they rotate across L1 cache sets instead of
# aliasing every stream onto the same set. Train each stream on lines 0,1,2 in
# round-robin order, then probe line 3 of every stream.
.macro MULTISTREAM_CASE name, gap
FN \name
  addi sp,sp,-32
  sd s0,0(sp)
  sd s1,8(sp)
  sd s2,16(sp)
  sd s3,24(sp)
  mv s0,a0
  mv s1,a1
  mv s2,a2
  li a0,0
  li t0,0
  li t6,4160
  .p2align 6
.Ltrial_\@:
  ld s3,0(s0)
  addi s0,s0,8
  li t1,0
.Lline_\@:
  li t2,3
  bgeu t1,t2,.Lprobe_\@
  slli t3,t1,6
  add t3,s3,t3
  li t4,0
.Lstream_\@:
  ld t5,0(t3)
  xor a0,a0,t5
  .rept \gap
    nop
  .endr
  add t3,t3,t6
  addi t4,t4,1
  bltu t4,s2,.Lstream_\@
  addi t1,t1,1
  j .Lline_\@
.Lprobe_\@:
  addi t3,s3,192
  li t4,0
.Lprobe_stream_\@:
  ld t5,0(t3)
  xor a0,a0,t5
  .rept \gap
    nop
  .endr
  add t3,t3,t6
  addi t4,t4,1
  bltu t4,s2,.Lprobe_stream_\@
  addi t0,t0,1
  bltu t0,s1,.Ltrial_\@
  ld s0,0(sp)
  ld s1,8(sp)
  ld s2,16(sp)
  ld s3,24(sp)
  addi sp,sp,32
  ret
END \name
.endm

MULTISTREAM_CASE prefetch_multistream_gap16, 16
MULTISTREAM_CASE prefetch_multistream_gap64, 64
