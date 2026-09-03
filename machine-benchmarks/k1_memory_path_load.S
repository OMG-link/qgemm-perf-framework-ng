.section .text

.global k1_rvv_read_m8
.type k1_rvv_read_m8, @function
k1_rvv_read_m8:
    # a0 = rounds, a1 = buffer, a2 = bytes (a positive multiple of 256)
    vsetvli t0, zero, e8, m8, ta, ma
    li t1, 0
.Lround:
    beq t1, a0, .Ldone
    mv t2, a1
    add t3, a1, a2
.Lscan:
    vle8.v v0, (t2)
    add t2, t2, t0
    bltu t2, t3, .Lscan
    addi t1, t1, 1
    j .Lround
.Ldone:
    ret
.size k1_rvv_read_m8, .-k1_rvv_read_m8