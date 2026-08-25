#ifndef IME_LLAMA_SELECTED_CYCLES_H
#define IME_LLAMA_SELECTED_CYCLES_H

#include <cstdint>

// Ad-hoc cycle accumulator for marking a selected hot region while tuning.
// The benchmark is single-threaded; these globals intentionally avoid atomic
// overhead in the measured region.
extern int64_t selected_cycles;
extern uint64_t selected_cycles_start;

[[maybe_unused]] static inline uint64_t read_cycle_counter() {
    uint64_t cycle;
    asm volatile("rdcycle %0" : "=r"(cycle) : : "memory");
    return cycle;
}

[[maybe_unused]] static inline void reset_selected_cycles() {
    selected_cycles = 0;
    selected_cycles_start = 0;
}

[[maybe_unused]] static inline void start_select() {
    selected_cycles_start = read_cycle_counter();
}

[[maybe_unused]] static inline void end_select() {
    selected_cycles += static_cast<int64_t>(read_cycle_counter() - selected_cycles_start);
}

#endif