.option norvc
.text

# a0 = iteration count, a1 = head of a 64-byte-granularity pointer ring.
# Every load depends on the preceding load, preventing memory-level parallelism
# and making hardware look-ahead prefetch ineffective.
.p2align 8
.global k1_memory_latency_chase
.type k1_memory_latency_chase,@function
k1_memory_latency_chase:
  beqz a0, .Ldone
  li t0, 0
  .p2align 6
.global k1_memory_latency_load
k1_memory_latency_load:
  ld a1, 0(a1)
  addi t0, t0, 1
  bltu t0, a0, k1_memory_latency_load
.Ldone:
  mv a0, a1
  ret
.size k1_memory_latency_chase,.-k1_memory_latency_chase
