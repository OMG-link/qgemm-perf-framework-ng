.section .text

# ABI: a0=passes, a1=src (147456 B), a2=dst (294912 B), a3=sink (>=32 B).
# Each inner iteration reproduces one production "inner" group: four 32-byte
# packed loads and stores at low offsets 0..96 and high offsets 128..224.
.macro BEGIN
    vsetvli t5, zero, e8, m1, ta, ma
    li t0, 0
1:
    mv t1, a1
    mv t2, a2
    li t3, 1152
2:
.endm

.macro NEXT_GROUP
    addi t1, t1, 128
    addi t2, t2, 256
    addi t3, t3, -1
    bnez t3, 2b
    addi t0, t0, 1
    bltu t0, a0, 1b
.endm

.macro LOAD reg, offset
    addi t4, t1, \offset
    vle8.v \reg, (t4)
.endm

.macro STORE reg, offset
    addi t4, t2, \offset
    vse8.v \reg, (t4)
.endm

.macro FULL_OPERAND packed, lo, hi, src_offset, dst_offset
    LOAD \packed, \src_offset
    vand.vi \lo, \packed, 15
    vsrl.vi \hi, \packed, 4
    vadd.vi \lo, \lo, -8
    vadd.vi \hi, \hi, -8
    STORE \lo, \dst_offset
    STORE \hi, (\dst_offset + 128)
.endm

.global unpack_conflict_full
.type unpack_conflict_full,@function
unpack_conflict_full:
    BEGIN
    FULL_OPERAND v1, v2, v3, 0, 0
    FULL_OPERAND v1, v2, v3, 32, 32
    FULL_OPERAND v1, v2, v3, 64, 64
    FULL_OPERAND v1, v2, v3, 96, 96
    NEXT_GROUP
    vse8.v v3, (a3)
    ret
.size unpack_conflict_full, .-unpack_conflict_full

.global unpack_conflict_load_2store
.type unpack_conflict_load_2store,@function
unpack_conflict_load_2store:
    BEGIN
    LOAD v1, 0; STORE v1, 0; STORE v1, 128
    LOAD v1, 32; STORE v1, 32; STORE v1, 160
    LOAD v1, 64; STORE v1, 64; STORE v1, 192
    LOAD v1, 96; STORE v1, 96; STORE v1, 224
    NEXT_GROUP
    vse8.v v1, (a3)
    ret
.size unpack_conflict_load_2store, .-unpack_conflict_load_2store

.global unpack_conflict_load_1store
.type unpack_conflict_load_1store,@function
unpack_conflict_load_1store:
    BEGIN
    LOAD v1, 0; STORE v1, 0
    LOAD v1, 32; STORE v1, 32
    LOAD v1, 64; STORE v1, 64
    LOAD v1, 96; STORE v1, 96
    NEXT_GROUP
    vse8.v v1, (a3)
    ret
.size unpack_conflict_load_1store, .-unpack_conflict_load_1store

.global unpack_conflict_load_only
.type unpack_conflict_load_only,@function
unpack_conflict_load_only:
    BEGIN
    LOAD v1, 0
    LOAD v1, 32
    LOAD v1, 64
    LOAD v1, 96
    NEXT_GROUP
    vse8.v v1, (a3)
    ret
.size unpack_conflict_load_only, .-unpack_conflict_load_only

.global unpack_conflict_store_only
.type unpack_conflict_store_only,@function
unpack_conflict_store_only:
    vsetvli t5, zero, e8, m1, ta, ma
    vmv.v.i v1, 1
    li t0, 0
1:
    mv t2, a2
    li t3, 1152
2:
    STORE v1, 0; STORE v1, 128
    STORE v1, 32; STORE v1, 160
    STORE v1, 64; STORE v1, 192
    STORE v1, 96; STORE v1, 224
    addi t2, t2, 256
    addi t3, t3, -1
    bnez t3, 2b
    addi t0, t0, 1
    bltu t0, a0, 1b
    vse8.v v1, (a3)
    ret
.size unpack_conflict_store_only, .-unpack_conflict_store_only

.macro STORE_ONLY_BEGIN
    vsetvli t5, zero, e8, m1, ta, ma
    vmv.v.i v1, 1
    li t0, 0
1:
    mv t2, a2
    li t3, 1152
2:
.endm
.macro STORE_ONLY_END
    addi t2, t2, 256
    addi t3, t3, -1
    bnez t3, 2b
    addi t0, t0, 1
    bltu t0, a0, 1b
    vse8.v v1, (a3)
    ret
.endm

.global unpack_conflict_store_1
.type unpack_conflict_store_1,@function
unpack_conflict_store_1:
    STORE_ONLY_BEGIN
    STORE v1, 0; STORE v1, 32; STORE v1, 64; STORE v1, 96
    STORE_ONLY_END
.size unpack_conflict_store_1, .-unpack_conflict_store_1

.global unpack_conflict_store_2_d32
.type unpack_conflict_store_2_d32,@function
unpack_conflict_store_2_d32:
    STORE_ONLY_BEGIN
    STORE v1, 0; STORE v1, 32
    STORE v1, 32; STORE v1, 64
    STORE v1, 64; STORE v1, 96
    STORE v1, 96; STORE v1, 128
    STORE_ONLY_END
.size unpack_conflict_store_2_d32, .-unpack_conflict_store_2_d32

.global unpack_conflict_store_2_d64
.type unpack_conflict_store_2_d64,@function
unpack_conflict_store_2_d64:
    STORE_ONLY_BEGIN
    STORE v1, 0; STORE v1, 64
    STORE v1, 32; STORE v1, 96
    STORE v1, 64; STORE v1, 128
    STORE v1, 96; STORE v1, 160
    STORE_ONLY_END
.size unpack_conflict_store_2_d64, .-unpack_conflict_store_2_d64
