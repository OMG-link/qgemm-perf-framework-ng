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

# a0=array of randomized region pointers, a1=trials, a2=train lines.
# Every training load and the probe are separated by a fixed NOP corridor.
.macro PREFETCH_CASE name, gap
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
  .p2align 6
.Ltrial_\@:
  ld t1,0(s0)
  addi s0,s0,8
  li t2,0
.Ltrain_\@:
  bgeu t2,s2,.Lprobe_\@
  ld t3,0(t1)
  xor a0,a0,t3
  .rept \gap
    nop
  .endr
  addi t1,t1,64
  addi t2,t2,1
  j .Ltrain_\@
.Lprobe_\@:
  ld t3,0(t1)
  xor a0,a0,t3
  .rept \gap
    nop
  .endr
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

PREFETCH_CASE prefetch_gap0, 0
PREFETCH_CASE prefetch_gap16, 16
PREFETCH_CASE prefetch_gap64, 64
PREFETCH_CASE prefetch_gap256, 256
