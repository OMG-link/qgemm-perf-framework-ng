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

# a0 = iterations.  The taken branch skips an aligned 192-byte never-executed
# corridor.  Every executed instruction and every corridor word has a unique PC.
FN pc_corridor
  li t0, 0
  .p2align 6
pc_corridor_loop:
  addi t0,t0,1
  j pc_corridor_after
  .p2align 7
  .global pc_corridor_never_begin
pc_corridor_never_begin:
  .rept 48
    nop
  .endr
  .global pc_corridor_never_end
pc_corridor_never_end:
  .p2align 6
pc_corridor_after:
  addi t1,t1,1
  bltu t0,a0,pc_corridor_loop
  ret
END pc_corridor

# a0 = iterations, a1/a2 = initialized vector operands, a3 = output.
# Sixteen true RAW updates of v16; each vmadot is followed by a distinct
# sentinel PC.  Sentinels are harmless scalar immediates with unique encodings.
FN pc_vmadot_raw
  vsetvli t5,zero,e8,m1,ta,ma
  vle8.v v2,(a1)
  vle8.v v8,(a2)
  vsetivli zero,0,e8,m1,tu,ma
  vsetivli t5,16,e32,m2,ta,ma
  vmv.v.i v16,0
  li t0,0
  .p2align 6
pc_vmadot_raw_loop:
  smt.vmadot v16,v2,v8
  addi t1,zero,1
  smt.vmadot v16,v2,v8
  addi t1,zero,2
  smt.vmadot v16,v2,v8
  addi t1,zero,3
  smt.vmadot v16,v2,v8
  addi t1,zero,4
  smt.vmadot v16,v2,v8
  addi t1,zero,5
  smt.vmadot v16,v2,v8
  addi t1,zero,6
  smt.vmadot v16,v2,v8
  addi t1,zero,7
  smt.vmadot v16,v2,v8
  addi t1,zero,8
  smt.vmadot v16,v2,v8
  addi t1,zero,9
  smt.vmadot v16,v2,v8
  addi t1,zero,10
  smt.vmadot v16,v2,v8
  addi t1,zero,11
  smt.vmadot v16,v2,v8
  addi t1,zero,12
  smt.vmadot v16,v2,v8
  addi t1,zero,13
  smt.vmadot v16,v2,v8
  addi t1,zero,14
  smt.vmadot v16,v2,v8
  addi t1,zero,15
  smt.vmadot v16,v2,v8
  addi t1,zero,16
  addi t0,t0,1
  bltu t0,a0,pc_vmadot_raw_loop
  vs2r.v v16,(a3)
  ret
END pc_vmadot_raw

FN pc_vmadot_nop
  li t0,0
  .p2align 6
pc_vmadot_nop_loop:
  .rept 16
    nop
    addi t1,t1,1
  .endr
  addi t0,t0,1
  bltu t0,a0,pc_vmadot_nop_loop
  ret
END pc_vmadot_nop

# a0=iterations, a1=head, a2=0 immediate / nonzero independent gap.
# Both variants execute this same static pointer-chase function and the same
# dependent load PC.  Gap operations do not consume or alter the pointer.
FN pc_pointer_chase
  li t0,0
  li t2,0
  .p2align 6
  .global pc_pointer_load
pc_pointer_load:
  ld a1,0(a1)
  beqz a2,pc_pointer_no_gap
  addi t2,t2,1
  xori t2,t2,0x55
  addi t2,t2,3
  xori t2,t2,0x2a
pc_pointer_no_gap:
  addi t0,t0,1
  bltu t0,a0,pc_pointer_load
  xor a0,a1,t2
  ret
END pc_pointer_chase

# a0=iterations, a1=A ring, a2=B ring, a3=out, a4=mask.
# Exact DynPreUnpack inner instruction topology and fixed register assignment.
FN pc_dyn_inner
  vsetivli zero,16,e32,m2,ta,ma
  vmv.v.i v16,0; vmv.v.i v18,0; vmv.v.i v20,0; vmv.v.i v22,0
  vmv.v.i v24,0; vmv.v.i v26,0; vmv.v.i v28,0; vmv.v.i v30,0
  li t0,0
  .p2align 6
  .global pc_dyn_inner_loop
pc_dyn_inner_loop:
  and t1,t0,a4
  slli t2,t1,7
  add t2,a1,t2
  slli t3,t1,8
  add t3,a2,t3
  vl8re8.v v8,(t3)
  vsetvli t5,zero,e8,m1,ta,ma
  vle8.v v2,(t2)
  addi t4,t2,32; vle8.v v3,(t4)
  addi t4,t2,64; vle8.v v4,(t4)
  addi t4,t2,96; vle8.v v5,(t4)
  vsetivli zero,0,e8,m1,tu,ma
  smt.vmadot v16,v2,v8; smt.vmadot v18,v2,v9; smt.vmadot v20,v2,v10; smt.vmadot v22,v2,v11
  smt.vmadot v16,v3,v12; smt.vmadot v18,v3,v13; smt.vmadot v20,v3,v14; smt.vmadot v22,v3,v15
  smt.vmadot v24,v4,v8; smt.vmadot v26,v4,v9; smt.vmadot v28,v4,v10; smt.vmadot v30,v4,v11
  smt.vmadot v24,v5,v12; smt.vmadot v26,v5,v13; smt.vmadot v28,v5,v14; smt.vmadot v30,v5,v15
  addi t0,t0,1
  bltu t0,a0,pc_dyn_inner_loop
  vs2r.v v16,(a3); addi t4,a3,64; vs2r.v v18,(t4)
  ret
END pc_dyn_inner

# a0=hot 256-byte source, a1=hot 32-byte source, a2=iterations.
# The three target instructions are contiguous.  a0 is restored before each
# iteration because vsetvli intentionally overwrites it with VL.
FN pc_vl8r_vset_vle_hot
  addi sp,sp,-16
  sd s1,0(sp)
  sd s2,8(sp)
  mv s1,a1
  mv s2,a0
  li t0,0
  .p2align 6
  .global pc_vl8r_vset_vle_hot_loop
pc_vl8r_vset_vle_hot_loop:
  mv a0,s2
  .global pc_vl8r_vset_vle_hot_vl8r
pc_vl8r_vset_vle_hot_vl8r:
  vl8r.v v8,(a0)
  .global pc_vl8r_vset_vle_hot_vsetvli
pc_vl8r_vset_vle_hot_vsetvli:
  vsetvli a0,zero,e8,m1,ta,ma
  .global pc_vl8r_vset_vle_hot_vle8
pc_vl8r_vset_vle_hot_vle8:
  vle8.v v5,(s1)
  .global pc_vl8r_vset_vle_hot_post
pc_vl8r_vset_vle_hot_post:
  addi t1,zero,23
  addi t0,t0,1
  bltu t0,a2,pc_vl8r_vset_vle_hot_loop
  ld s1,0(sp)
  ld s2,8(sp)
  addi sp,sp,16
  ret
END pc_vl8r_vset_vle_hot

# Three diagnostic variants insert a 64-instruction NOP corridor at one
# boundary.  Distinct exported begin/end labels make each corridor directly
# countable by absolute PC without assigning samples to a source line.
FN pc_vl8r_nop_vset_vle_hot
  addi sp,sp,-16
  sd s1,0(sp)
  sd s2,8(sp)
  mv s1,a1
  mv s2,a0
  li t0,0
  .p2align 6
  .global pc_vl8r_nop_vset_vle_hot_loop
pc_vl8r_nop_vset_vle_hot_loop:
  mv a0,s2
  vl8r.v v8,(a0)
  .global pc_vl8r_nop_vset_vle_hot_gap_begin
pc_vl8r_nop_vset_vle_hot_gap_begin:
  .rept 64
    nop
  .endr
  .global pc_vl8r_nop_vset_vle_hot_gap_end
pc_vl8r_nop_vset_vle_hot_gap_end:
  vsetvli a0,zero,e8,m1,ta,ma
  vle8.v v5,(s1)
  addi t1,zero,24
  addi t0,t0,1
  bltu t0,a2,pc_vl8r_nop_vset_vle_hot_loop
  ld s1,0(sp)
  ld s2,8(sp)
  addi sp,sp,16
  ret
END pc_vl8r_nop_vset_vle_hot

FN pc_vl8r_vset_nop_vle_hot
  addi sp,sp,-16
  sd s1,0(sp)
  sd s2,8(sp)
  mv s1,a1
  mv s2,a0
  li t0,0
  .p2align 6
  .global pc_vl8r_vset_nop_vle_hot_loop
pc_vl8r_vset_nop_vle_hot_loop:
  mv a0,s2
  vl8r.v v8,(a0)
  vsetvli a0,zero,e8,m1,ta,ma
  .global pc_vl8r_vset_nop_vle_hot_gap_begin
pc_vl8r_vset_nop_vle_hot_gap_begin:
  .rept 64
    nop
  .endr
  .global pc_vl8r_vset_nop_vle_hot_gap_end
pc_vl8r_vset_nop_vle_hot_gap_end:
  vle8.v v5,(s1)
  addi t1,zero,25
  addi t0,t0,1
  bltu t0,a2,pc_vl8r_vset_nop_vle_hot_loop
  ld s1,0(sp)
  ld s2,8(sp)
  addi sp,sp,16
  ret
END pc_vl8r_vset_nop_vle_hot

FN pc_vl8r_vset_vle_nop_hot
  addi sp,sp,-16
  sd s1,0(sp)
  sd s2,8(sp)
  mv s1,a1
  mv s2,a0
  li t0,0
  .p2align 6
  .global pc_vl8r_vset_vle_nop_hot_loop
pc_vl8r_vset_vle_nop_hot_loop:
  mv a0,s2
  vl8r.v v8,(a0)
  vsetvli a0,zero,e8,m1,ta,ma
  vle8.v v5,(s1)
  .global pc_vl8r_vset_vle_nop_hot_gap_begin
pc_vl8r_vset_vle_nop_hot_gap_begin:
  .rept 64
    nop
  .endr
  .global pc_vl8r_vset_vle_nop_hot_gap_end
pc_vl8r_vset_vle_nop_hot_gap_end:
  addi t1,zero,26
  addi t0,t0,1
  bltu t0,a2,pc_vl8r_vset_vle_nop_hot_loop
  ld s1,0(sp)
  ld s2,8(sp)
  addi sp,sp,16
  ret
END pc_vl8r_vset_vle_nop_hot

# Scalar control with the same loop/prologue and 68 NOP slots as each gap
# variant (three replaced vector instructions plus the 64-NOP corridor and one
# post marker replacement).
FN pc_vl8r_vset_vle_nop_control
  addi sp,sp,-16
  sd s1,0(sp)
  sd s2,8(sp)
  mv s1,a1
  mv s2,a0
  li t0,0
  .p2align 6
  .global pc_vl8r_vset_vle_nop_control_loop
pc_vl8r_vset_vle_nop_control_loop:
  mv a0,s2
  nop
  nop
  nop
  .global pc_vl8r_vset_vle_nop_control_gap_begin
pc_vl8r_vset_vle_nop_control_gap_begin:
  .rept 64
    nop
  .endr
  .global pc_vl8r_vset_vle_nop_control_gap_end
pc_vl8r_vset_vle_nop_control_gap_end:
  nop
  addi t0,t0,1
  bltu t0,a2,pc_vl8r_vset_vle_nop_control_loop
  ld s1,0(sp)
  ld s2,8(sp)
  addi sp,sp,16
  ret
END pc_vl8r_vset_vle_nop_control

# A zero-immediate vector add preserves all active bytes while creating a true
# RAW dependency on one loaded m1 register.  The B variant consumes v8 after
# vsetvli establishes VL=32; the A variant consumes v5 immediately after vle8.
FN pc_vl8r_vset_vadd_vle_hot
  addi sp,sp,-16
  sd s1,0(sp)
  sd s2,8(sp)
  mv s1,a1
  mv s2,a0
  li t0,0
  .p2align 6
  .global pc_vl8r_vset_vadd_vle_hot_loop
pc_vl8r_vset_vadd_vle_hot_loop:
  mv a0,s2
  .global pc_vl8r_vset_vadd_vle_hot_vl8r
pc_vl8r_vset_vadd_vle_hot_vl8r:
  vl8r.v v8,(a0)
  .global pc_vl8r_vset_vadd_vle_hot_vset
pc_vl8r_vset_vadd_vle_hot_vset:
  vsetvli a0,zero,e8,m1,ta,ma
  .global pc_vl8r_vset_vadd_vle_hot_consumer
pc_vl8r_vset_vadd_vle_hot_consumer:
  vadd.vi v8,v8,0
  .global pc_vl8r_vset_vadd_vle_hot_vle
pc_vl8r_vset_vadd_vle_hot_vle:
  vle8.v v5,(s1)
  .global pc_vl8r_vset_vadd_vle_hot_post
pc_vl8r_vset_vadd_vle_hot_post:
  addi t1,zero,27
  addi t0,t0,1
  bltu t0,a2,pc_vl8r_vset_vadd_vle_hot_loop
  ld s1,0(sp)
  ld s2,8(sp)
  addi sp,sp,16
  ret
END pc_vl8r_vset_vadd_vle_hot

# Consume the last m1 register of the vl8r destination group.  Unlike the v8
# variant, this probes whether the tail of the 256-byte whole-register load has
# completed before the following A load is issued.
FN pc_vl8r_vset_vadd15_vle_hot
  addi sp,sp,-16
  sd s1,0(sp)
  sd s2,8(sp)
  mv s1,a1
  mv s2,a0
  li t0,0
  .p2align 6
  .global pc_vl8r_vset_vadd15_vle_hot_loop
pc_vl8r_vset_vadd15_vle_hot_loop:
  mv a0,s2
  .global pc_vl8r_vset_vadd15_vle_hot_vl8r
pc_vl8r_vset_vadd15_vle_hot_vl8r:
  vl8r.v v8,(a0)
  .global pc_vl8r_vset_vadd15_vle_hot_vset
pc_vl8r_vset_vadd15_vle_hot_vset:
  vsetvli a0,zero,e8,m1,ta,ma
  .global pc_vl8r_vset_vadd15_vle_hot_consumer
pc_vl8r_vset_vadd15_vle_hot_consumer:
  vadd.vi v15,v15,0
  .global pc_vl8r_vset_vadd15_vle_hot_vle
pc_vl8r_vset_vadd15_vle_hot_vle:
  vle8.v v5,(s1)
  .global pc_vl8r_vset_vadd15_vle_hot_post
pc_vl8r_vset_vadd15_vle_hot_post:
  addi t1,zero,29
  addi t0,t0,1
  bltu t0,a2,pc_vl8r_vset_vadd15_vle_hot_loop
  ld s1,0(sp)
  ld s2,8(sp)
  addi sp,sp,16
  ret
END pc_vl8r_vset_vadd15_vle_hot

# Consume the complete vl8r destination as one LMUL=8 group, then restore m1
# before issuing the independent A load.
FN pc_vl8r_vsetm8_vadd_vsetm1_vle_hot
  addi sp,sp,-16
  sd s1,0(sp)
  sd s2,8(sp)
  mv s1,a1
  mv s2,a0
  li t0,0
  .p2align 6
  .global pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_loop
pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_loop:
  mv a0,s2
  .global pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_vl8r
pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_vl8r:
  vl8r.v v8,(a0)
  .global pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_vsetm8
pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_vsetm8:
  vsetvli t6,zero,e8,m8,ta,ma
  .global pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_consumer
pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_consumer:
  vadd.vi v8,v8,0
  .global pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_vsetm1
pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_vsetm1:
  vsetvli a0,zero,e8,m1,ta,ma
  .global pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_vle
pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_vle:
  vle8.v v5,(s1)
  .global pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_post
pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_post:
  addi t1,zero,30
  addi t0,t0,1
  bltu t0,a2,pc_vl8r_vsetm8_vadd_vsetm1_vle_hot_loop
  ld s1,0(sp)
  ld s2,8(sp)
  addi sp,sp,16
  ret
END pc_vl8r_vsetm8_vadd_vsetm1_vle_hot

FN pc_vl8r_vset_vle_vadd_hot
  addi sp,sp,-16
  sd s1,0(sp)
  sd s2,8(sp)
  mv s1,a1
  mv s2,a0
  li t0,0
  .p2align 6
  .global pc_vl8r_vset_vle_vadd_hot_loop
pc_vl8r_vset_vle_vadd_hot_loop:
  mv a0,s2
  .global pc_vl8r_vset_vle_vadd_hot_vl8r
pc_vl8r_vset_vle_vadd_hot_vl8r:
  vl8r.v v8,(a0)
  .global pc_vl8r_vset_vle_vadd_hot_vset
pc_vl8r_vset_vle_vadd_hot_vset:
  vsetvli a0,zero,e8,m1,ta,ma
  .global pc_vl8r_vset_vle_vadd_hot_vle
pc_vl8r_vset_vle_vadd_hot_vle:
  vle8.v v5,(s1)
  .global pc_vl8r_vset_vle_vadd_hot_consumer
pc_vl8r_vset_vle_vadd_hot_consumer:
  vadd.vi v5,v5,0
  .global pc_vl8r_vset_vle_vadd_hot_post
pc_vl8r_vset_vle_vadd_hot_post:
  addi t1,zero,28
  addi t0,t0,1
  bltu t0,a2,pc_vl8r_vset_vle_vadd_hot_loop
  ld s1,0(sp)
  ld s2,8(sp)
  addi sp,sp,16
  ret
END pc_vl8r_vset_vle_vadd_hot

.section .note.GNU-stack,"",@progbits
