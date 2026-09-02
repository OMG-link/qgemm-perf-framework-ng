# q4_K/q8_K RVV upstream provenance

`kernel.cpp` is a local source copy of `ggml_vec_dot_q4_K_q8_K` from:

- Repository: `/home/linkai/projects/llama.cpp`
- Commit: `314e729347defd9851e857f78084160c5786a7d8`
- Source: `ggml/src/ggml-cpu/arch/riscv/quants.c`, lines 1321-1728

The complete upstream function was copied, including its T-Head Vector and
standard RVV 128/256-bit branches. Changes are limited to local includes and
C-to-C++ pointer/restrict compatibility. This project does not include or link
code from the external checkout during its build.

The copied code is distributed under the MIT license in `LICENSE`.