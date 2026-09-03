# IQ2_XXS/Q8_K RVV upstream provenance

`kernel.cpp` and `tables.h` are local C++ extracts derived from:

- Repository: `/home/linkai/projects/llama.cpp`
- Commit: `314e729347defd9851e857f78084160c5786a7d8`
- Kernel source: `ggml/src/ggml-cpu/arch/riscv/quants.c`
  - `ggml_vec_dot_iq2_xxs_q8_K_vl256`, lines 3209-3301
  - `ggml_vec_dot_iq2_xxs_q8_K`, lines 3304-3317
- Lookup tables: `ggml/src/ggml-common.h`

Only the 256-bit RVV path required by the SpacemiT X60 benchmark is retained.
The extract uses this project's dependency-free GGML compatibility types and
does not include or link code from the external checkout during its build.

The source is distributed under the MIT license in
`bench/llama_reference/LICENSE`.