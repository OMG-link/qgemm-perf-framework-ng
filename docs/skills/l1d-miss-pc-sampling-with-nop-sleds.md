# Locating L1D-Missing Loads with NOP Sleds

PMU overflow interrupts have skid, so the PC captured by `perf record` may point past the load that triggered the miss. Place a distinct NOP sled after each candidate load and aggregate samples over `[load PC, sled end)` to approximately attribute misses to individual load sites.

```cpp
auto value = load(ptr);
asm volatile(
    ".rept 64\n"
    "nop\n"
    ".endr\n"
    :
    : "vr"(value)
    : "memory");
```

Key details:

- The compiler may schedule independent instructions between the load and its sled, so derive the interval start from the actual load PC in the final binary rather than from the sled start or source position.
- Referencing the loaded value and adding a memory clobber prevents the load from being removed or moved across the sled; still inspect the final assembly to confirm the actual order.
- NOPs give prefetchers more time and can change cache behavior. Compare total miss counts before and after instrumentation, and repeat with different sled lengths to determine whether the attribution still represents the original kernel.
- Samples accumulating near the sled tail or in the next sled indicate that the sled is too short. Once samples consistently remain near the front, a longer sled only adds perturbation.
- A PMU missed-load event is not a cache-line count; loads crossing lines and multiple loads sharing one line are especially easy to misinterpret.
- On RISC-V, explicitly disable RVC for the sled and derive its range from the final ELF, because `nop` may otherwise assemble to a 2-byte compressed instruction and invalidate source-count-based PC attribution.