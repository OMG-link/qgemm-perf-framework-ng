#include <algorithm>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <riscv_vector.h>

namespace {

constexpr std::size_t kDefaultSingleBytes = 256 * 1024;
constexpr std::size_t kDefaultThreadBytes = 96 * 1024;
constexpr std::size_t kDefaultSamples = 5;
constexpr std::size_t kDefaultDurationMs = 250;
constexpr std::size_t kCacheLine = 64;

struct Options {
    std::string mode = "all";
    std::vector<int> cpus{0, 1, 2, 3};
    std::size_t single_bytes = kDefaultSingleBytes;
    std::size_t thread_bytes = kDefaultThreadBytes;
    std::size_t samples = kDefaultSamples;
    std::size_t duration_ms = kDefaultDurationMs;
};

struct Sample {
    double seconds = 0.0;
    std::uint64_t bytes = 0;
    std::uint64_t checksum = 0;
};

struct AlignedBuffer {
    explicit AlignedBuffer(std::size_t size) : size(size) {
        if (posix_memalign(reinterpret_cast<void **>(&data), kCacheLine, size) != 0) {
            data = nullptr;
        }
        if (data) std::memset(data, 0, size);
    }
    ~AlignedBuffer() { std::free(data); }
    AlignedBuffer(const AlignedBuffer &) = delete;
    AlignedBuffer &operator=(const AlignedBuffer &) = delete;
    AlignedBuffer(AlignedBuffer &&other) noexcept : data(other.data), size(other.size) { other.data = nullptr; }
    AlignedBuffer &operator=(AlignedBuffer &&other) noexcept {
        if (this != &other) {
            std::free(data);
            data = other.data;
            size = other.size;
            other.data = nullptr;
        }
        return *this;
    }
    std::uint8_t *data = nullptr;
    std::size_t size;
};

bool parse_size(std::string_view text, std::size_t &value) {
    if (text.empty()) return false;
    const std::string input(text);
    errno = 0;
    char *end = nullptr;
    const auto parsed = std::strtoull(input.c_str(), &end, 10);
    if (errno || !end || *end || parsed == 0 || parsed > SIZE_MAX) return false;
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool parse_cpus(std::string_view text, std::vector<int> &cpus) {
    cpus.clear();
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        std::size_t value = 0;
        if (item.empty()) return false;
        const std::string input(item);
        errno = 0;
        char *end = nullptr;
        const auto parsed = std::strtoull(input.c_str(), &end, 10);
        if (errno || !end || *end || parsed > static_cast<std::size_t>(CPU_SETSIZE - 1)) return false;
        value = static_cast<std::size_t>(parsed);
        cpus.push_back(static_cast<int>(value));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    return !cpus.empty();
}

void usage(const char *program) {
    std::printf("Usage: %s [--mode all|single|concurrent] [--cpus 0,1,2,3] "
                "[--single-bytes N] [--per-thread-bytes N] [--samples N] [--duration-ms N]\n",
                program);
}

bool set_affinity(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

std::uint64_t read_buffer(const std::uint8_t *data, std::size_t bytes, std::size_t iterations) {
    std::uint64_t checksum = 0;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t offset = 0; offset < bytes; offset += 32) {
            vuint8m1_t loaded = __riscv_vle8_v_u8m1(data + offset, 32);
            asm volatile("" : "+vr"(loaded));
            checksum += __riscv_vmv_x_s_u8m1_u8(loaded);
        }
    }
    return checksum;
}

Sample measure_buffer(const AlignedBuffer &buffer, std::size_t duration_ms) {
    const auto warmup = read_buffer(buffer.data, buffer.size, 2);
    std::size_t iterations = 1;
    auto begin = std::chrono::steady_clock::now();
    auto end = begin;
    do {
        begin = std::chrono::steady_clock::now();
        const auto checksum = read_buffer(buffer.data, buffer.size, iterations);
        end = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() * 2 <
            static_cast<long long>(duration_ms)) {
            iterations *= 2;
        } else {
            const auto seconds = std::chrono::duration<double>(end - begin).count();
            return {seconds, static_cast<std::uint64_t>(buffer.size) * iterations, checksum ^ warmup};
        }
    } while (iterations <= (std::numeric_limits<std::size_t>::max() / 2));
    return {0.0, 0, warmup};
}

void print_sample(const char *label, const Sample &sample) {
    const double gbps = sample.seconds > 0 ? static_cast<double>(sample.bytes) / sample.seconds / 1e9 : 0.0;
    const double gibps = sample.seconds > 0 ? static_cast<double>(sample.bytes) / sample.seconds / (1ull << 30) : 0.0;
    std::printf("%s seconds=%.9f bytes=%llu GB/s=%.4f GiB/s=%.4f checksum=%llu\n", label, sample.seconds,
                static_cast<unsigned long long>(sample.bytes), gbps, gibps,
                static_cast<unsigned long long>(sample.checksum));
}

int run_single(const Options &options) {
    std::vector<AlignedBuffer> buffers;
    for (std::size_t i = 0; i < options.cpus.size(); ++i) buffers.emplace_back(options.single_bytes);
    for (std::size_t i = 0; i < options.cpus.size(); ++i) {
        if (!buffers[i].data) return 2;
        std::vector<Sample> samples;
        for (std::size_t sample = 0; sample < options.samples; ++sample) {
            Sample result;
            bool affinity_ok = false;
            std::thread worker([&, i, result_ptr = &result, affinity_ptr = &affinity_ok] {
                *affinity_ptr = set_affinity(options.cpus[i]);
                if (*affinity_ptr) *result_ptr = measure_buffer(buffers[i], options.duration_ms);
            });
            worker.join();
            if (!affinity_ok || !result.bytes) return 2;
            samples.push_back(result);
        }
        const auto best = *std::min_element(samples.begin(), samples.end(), [](const Sample &a, const Sample &b) {
            return a.seconds < b.seconds;
        });
        char label[64];
        std::snprintf(label, sizeof(label), "single cpu=%d", options.cpus[i]);
        print_sample(label, best);
    }
    return 0;
}

int run_concurrent(const Options &options) {
    std::vector<AlignedBuffer> buffers;
    for (std::size_t i = 0; i < options.cpus.size(); ++i) buffers.emplace_back(options.thread_bytes);
    for (const auto &buffer : buffers) if (!buffer.data || buffer.size < kCacheLine) return 2;

    std::vector<Sample> samples;
    for (std::size_t sample = 0; sample < options.samples; ++sample) {
        std::barrier start(static_cast<std::ptrdiff_t>(options.cpus.size()));
        std::vector<Sample> results(options.cpus.size());
        std::vector<std::thread> workers;
        std::vector<bool> affinity_ok(options.cpus.size(), false);
        for (std::size_t i = 0; i < options.cpus.size(); ++i) {
            workers.emplace_back([&, i] {
                affinity_ok[i] = set_affinity(options.cpus[i]);
                start.arrive_and_wait();
                if (affinity_ok[i]) {
                    read_buffer(buffers[i].data, buffers[i].size, 2);
                    results[i] = measure_buffer(buffers[i], options.duration_ms);
                }
            });
        }
        for (auto &worker : workers) worker.join();
        Sample total;
        for (const auto &result : results) {
            if (!result.bytes) return 2;
            total.bytes += result.bytes;
            total.checksum ^= result.checksum;
            total.seconds = std::max(total.seconds, result.seconds);
        }
        samples.push_back(total);
    }
    const auto best = *std::min_element(samples.begin(), samples.end(), [](const Sample &a, const Sample &b) {
        return a.seconds < b.seconds;
    });
    print_sample("concurrent aggregate", best);
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help") { usage(argv[0]); return 0; }
        if (i + 1 >= argc) { usage(argv[0]); return 2; }
        std::size_t value = 0;
        if (arg == "--mode") {
            options.mode = argv[++i];
        } else if (arg == "--cpus") {
            if (!parse_cpus(argv[++i], options.cpus)) return 2;
        } else if (arg == "--single-bytes") {
            if (!parse_size(argv[++i], options.single_bytes)) return 2;
        } else if (arg == "--per-thread-bytes") {
            if (!parse_size(argv[++i], options.thread_bytes)) return 2;
        } else if (arg == "--samples") {
            if (!parse_size(argv[++i], value)) return 2;
            options.samples = value;
        } else if (arg == "--duration-ms") {
            if (!parse_size(argv[++i], value)) return 2;
            options.duration_ms = value;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (options.mode != "all" && options.mode != "single" && options.mode != "concurrent") return 2;
    if (options.cpus.size() != 4) {
        std::fprintf(stderr, "expected exactly four CPUs, got %zu\n", options.cpus.size());
        return 2;
    }
    std::printf("configuration cpus=%d,%d,%d,%d single_bytes=%zu per_thread_bytes=%zu samples=%zu duration_ms=%zu\n",
                options.cpus[0], options.cpus[1], options.cpus[2], options.cpus[3], options.single_bytes,
                options.thread_bytes, options.samples, options.duration_ms);
    if (options.mode == "all" || options.mode == "single") if (run_single(options)) return 1;
    if (options.mode == "all" || options.mode == "concurrent") if (run_concurrent(options)) return 1;
    return 0;
}