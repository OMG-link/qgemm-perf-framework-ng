# q4_0/q8_0 RVV upstream provenance

`kernel.cpp` is a local C++ extract of `ggml_vec_dot_q4_0_q8_0` from:

- Repository: `/home/linkai/projects/llama.cpp`
- Commit: `314e729347defd9851e857f78084160c5786a7d8`
- Source: `ggml/src/ggml-cpu/arch/riscv/quants.c`, lines 222-275

The extract uses this project's dependency-free GGML compatibility types and
does not include or link code from the external checkout during its build.

The source is distributed under the MIT license in
`bench/llama_reference/LICENSE`.