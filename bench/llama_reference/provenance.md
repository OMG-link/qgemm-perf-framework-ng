# llama.cpp reference provenance

The reference Q4_0/Q8_0 routines are locally maintained, dependency-free
extracts derived from llama.cpp commit `314e72934`:

- `ggml/src/ggml-quants.c`: Q4_0 and Q8_0 reference quantization.
- `ggml/src/ggml-cpu/quants.c`: scalar Q4_0 × Q8_0 dot product.
- `ggml/src/ggml-cpu/spacemit/repack.cpp`: Q4_0 N16 repack semantics used by adapters.

Unrelated GGML types and runtime dependencies were removed and the routines
were placed in the `ime::bench::llama_reference` C++ namespace. The build does
not include or link files from an external llama.cpp checkout.