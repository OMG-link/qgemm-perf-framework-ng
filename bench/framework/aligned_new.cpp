#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace {

constexpr std::size_t kMinimumAlignment = 64;

void *allocate_aligned(std::size_t size, std::size_t requested_alignment) {
    const std::size_t alignment = std::max(kMinimumAlignment, requested_alignment);
    const std::size_t allocation_size = std::max<std::size_t>(size, 1);

    for (;;) {
        void *pointer = nullptr;
        if (posix_memalign(&pointer, alignment, allocation_size) == 0)
            return pointer;
        const std::new_handler handler = std::get_new_handler();
        if (!handler)
            throw std::bad_alloc();
        handler();
    }
}

void *allocate_aligned_nothrow(std::size_t size, std::size_t alignment) noexcept {
    try {
        return allocate_aligned(size, alignment);
    } catch (...) {
        return nullptr;
    }
}

} // namespace

void *operator new(std::size_t size) { return allocate_aligned(size, kMinimumAlignment); }
void *operator new[](std::size_t size) { return allocate_aligned(size, kMinimumAlignment); }

void *operator new(std::size_t size, const std::nothrow_t &) noexcept { return allocate_aligned_nothrow(size, kMinimumAlignment); }
void *operator new[](std::size_t size, const std::nothrow_t &) noexcept { return allocate_aligned_nothrow(size, kMinimumAlignment); }

void *operator new(std::size_t size, std::align_val_t alignment) { return allocate_aligned(size, static_cast<std::size_t>(alignment)); }
void *operator new[](std::size_t size, std::align_val_t alignment) { return allocate_aligned(size, static_cast<std::size_t>(alignment)); }
void *operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t &) noexcept { return allocate_aligned_nothrow(size, static_cast<std::size_t>(alignment)); }
void *operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t &) noexcept { return allocate_aligned_nothrow(size, static_cast<std::size_t>(alignment)); }

void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void *pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete(void *pointer, const std::nothrow_t &) noexcept { std::free(pointer); }
void operator delete[](void *pointer, const std::nothrow_t &) noexcept { std::free(pointer); }

void operator delete(void *pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void *pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void *pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::align_val_t, const std::nothrow_t &) noexcept { std::free(pointer); }
void operator delete[](void *pointer, std::align_val_t, const std::nothrow_t &) noexcept { std::free(pointer); }
