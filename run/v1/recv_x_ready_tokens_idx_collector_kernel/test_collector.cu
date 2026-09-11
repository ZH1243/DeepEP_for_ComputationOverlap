// Standalone GPU protocol test: nvcc ... collector.cu test_collector.cu
#include "collector.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

#define CUDA_CHECK(call) do { \
    const auto error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(1); \
    } \
} while (0)

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        std::exit(1); \
    } \
} while (0)

template <typename T> struct DeviceArray {
    T* ptr = nullptr;
    size_t size;
    explicit DeviceArray(size_t count) : size(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceArray() { cudaFree(ptr); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    void fill_bytes(int value) { CUDA_CHECK(cudaMemset(ptr, value, size * sizeof(T))); }
    void put(const std::vector<T>& values) {
        CHECK(values.size() == size);
        CUDA_CHECK(cudaMemcpy(ptr, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> values(size);
        CUDA_CHECK(cudaMemcpy(values.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

__host__ __device__ int route(int row, int slot) {
    if (row % 11 == 0)
        return -1;
    if (slot == 0)
        return row % 8;
    if (slot == 1 && row % 3 == 0)
        return row % 8;  // Deliberately duplicate a membership.
    return slot == 2 ? (row + 3) % 8 : -1;
}

// Ordinary CUDA stores simulate the dispatch publication contract. This test
// deliberately does not claim to validate a DeepEP TMA integration.
template <typename Index>
__global__ void producer(
    Index* topk, int num_topk, int* payload, const int* boundaries, int num_ranges,
    int* begin, int* end, int* ready, int* observed, int* error,
    unsigned long long timeout) {
    const int lane = threadIdx.x;
    if (lane == 0) {
        for (int r = 0; r < num_ranges; ++r)
            recv_x_ready::initialize_range(begin, end, ready, r, boundaries[r], boundaries[r + 1]);
    }
    __syncwarp();
    for (int stage = 0; stage < 2; ++stage) {
        if (stage == 1) {
            if (lane == 0) {
                const auto started = clock64();
                // Final rows are withheld until a LIVE consumer has seen an
                // expert list entry. A collector that waits for all input will
                // fail this handshake instead of passing a final-output test.
                while (recv_x_ready::load_acquire(observed) == 0) {
                    if (clock64() - started > timeout) {
                        atomicExch(error, 1);
                        break;
                    }
                    __nanosleep(64);
                }
            }
            __syncwarp();
        }
        for (int r = 0; r < num_ranges; ++r) {
            const int first = boundaries[r];
            const int last = boundaries[r + 1];
            const int middle = first + (last - first) / 2;
            for (int row = stage == 0 ? first : middle;
                 row < (stage == 0 ? middle : last); ++row) {
                if (lane < num_topk)
                    topk[static_cast<int64_t>(row) * num_topk + lane] = route(row, lane);
                if (lane == 0)
                    payload[row] = row * 7 + 11;
                __syncwarp();
                if (lane == 0)
                    recv_x_ready::publish_ready(ready, r, row + 1);
                __syncwarp();
            }
        }
    }
}

__global__ void observer(
    const int* indices, const int* counts, const int* status, const int* payload,
    int rows, int num_topk, int capacity, int* observed, int* error,
    unsigned long long timeout) {
    const int lane = threadIdx.x;
    int seen = 0;
    const auto started = clock64();
    while (true) {
        int terminal = lane == 0 ? recv_x_ready::load_acquire(status) : 0;
        terminal = __shfl_sync(0xffffffffu, terminal, 0);
        __syncwarp();
        int count = 0;
        if (lane < 8) {
            count = recv_x_ready::load_acquire(counts + lane);
            if (count < seen || count > capacity) {
                atomicExch(error, 2);
            } else {
                for (int j = seen; j < count; ++j) {
                    const int row = indices[static_cast<int64_t>(lane) * capacity + j];
                    if (row < 0 || row >= rows) {
                        atomicExch(error, 3);
                    } else {
                        bool belongs = false;
                        for (int k = 0; k < num_topk; ++k)
                            belongs |= route(row, k) == lane;
                        if (!belongs || payload[row] != row * 7 + 11)
                            atomicExch(error, 4);
                    }
                }
            }
            seen = count;
        }
        const bool any = __any_sync(0xffffffffu, count > 0);
        if (lane == 0 && any)
            recv_x_ready::store_release(observed, 1);
        if (terminal != recv_x_ready::kRunning)
            return;
        const bool expired = __any_sync(0xffffffffu, clock64() - started > timeout);
        if (expired) {
            if (lane == 0) {
                atomicExch(error, 5);
                recv_x_ready::store_release(observed, 2);
            }
            return;
        }
        __nanosleep(64);
    }
}

template <typename Index>
void check_incremental(int num_topk, unsigned long long timeout) {
    constexpr int ranges = 23;  // Multiple polling groups, including a partial group.
    std::vector<int> bounds(1, 0);
    for (int r = 0; r < ranges; ++r)
        bounds.push_back(bounds.back() + (r % 7) * 19);  // Empty and uneven ranges.
    const int rows = bounds.back();
    DeviceArray<Index> topk(rows * num_topk);
    DeviceArray<int> payload(rows), boundaries(ranges + 1), begin(ranges), end(ranges), ready(ranges);
    DeviceArray<int> indices(8 * rows), counts(8), consumed(ranges), status(1), observed(1), error(1);
    boundaries.put(bounds);
    topk.fill_bytes(0x7f);  // Reading unpublished metadata should fail loudly.
    payload.fill_bytes(0xff);
    begin.fill_bytes(0x7f);
    end.fill_bytes(0x7f);
    ready.fill_bytes(0xff);
    indices.fill_bytes(0xff);
    counts.fill_bytes(0);
    consumed.fill_bytes(0xff);
    status.fill_bytes(0);
    observed.fill_bytes(0);
    error.fill_bytes(0);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t producer_stream, collector_stream, observer_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&collector_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&observer_stream, cudaStreamNonBlocking));
    CUDA_CHECK(recv_x_ready::launch_collector(
        topk.ptr, std::is_same<Index, int64_t>::value, rows, num_topk,
        begin.ptr, end.ptr, ready.ptr, ranges, indices.ptr, rows, counts.ptr,
        consumed.ptr, status.ptr, timeout, collector_stream));
    observer<<<1, 32, 0, observer_stream>>>(
        indices.ptr, counts.ptr, status.ptr, payload.ptr, rows, num_topk, rows,
        observed.ptr, error.ptr, timeout);
    CUDA_CHECK(cudaGetLastError());
    producer<<<1, 32, 0, producer_stream>>>(
        topk.ptr, num_topk, payload.ptr, boundaries.ptr, ranges, begin.ptr, end.ptr,
        ready.ptr, observed.ptr, error.ptr, timeout);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(status.get()[0] == recv_x_ready::kComplete);
    CHECK(observed.get()[0] == 1);
    CHECK(error.get()[0] == 0);
    const auto actual_counts = counts.get();
    const auto actual_indices = indices.get();
    const auto actual_consumed = consumed.get();
    for (int r = 0; r < ranges; ++r)
        CHECK(actual_consumed[r] == bounds[r + 1]);
    for (int expert = 0; expert < 8; ++expert) {
        std::vector<int> expected;
        for (int row = 0; row < rows; ++row) {
            bool belongs = false;
            for (int k = 0; k < num_topk; ++k)
                belongs |= route(row, k) == expert;
            if (belongs)
                expected.push_back(row);
        }
        CHECK(actual_counts[expert] == static_cast<int>(expected.size()));
        std::vector<int> actual(actual_indices.begin() + expert * rows,
                                actual_indices.begin() + expert * rows + actual_counts[expert]);
        std::sort(actual.begin(), actual.end());
        CHECK(actual == expected);  // Includes duplicate/missing-index checks.
    }
    CUDA_CHECK(cudaStreamDestroy(producer_stream));
    CUDA_CHECK(cudaStreamDestroy(collector_stream));
    CUDA_CHECK(cudaStreamDestroy(observer_stream));
    std::printf("PASS incremental int%zu, topk=%d\n", sizeof(Index) * 8, num_topk);
}

// All-ready input forces full tiles and repeated range visits independently of
// producer scheduling. Many empty ranges also exercise the shared-cache tail.
template <typename Index>
void check_backlog(int num_topk, int ranges, unsigned long long timeout) {
    std::vector<int> begins(ranges), ends(ranges);
    int rows = 0;
    for (int r = 0; r < ranges; ++r) {
        begins[r] = rows;
        rows += ranges > 4096 ? (r >= 4094 ? 35 : 0) : 65 + r % 3;
        ends[r] = rows;
    }
    std::vector<Index> routing(rows * num_topk);
    for (int row = 0; row < rows; ++row)
        for (int k = 0; k < num_topk; ++k)
            routing[row * num_topk + k] = route(row, k);
    DeviceArray<Index> topk(routing.size());
    DeviceArray<int> begin(ranges), end(ranges), ready(ranges), consumed(ranges);
    DeviceArray<int> indices(8 * rows), counts(8), status(1);
    topk.put(routing);
    begin.put(begins);
    end.put(ends);
    ready.put(ends);
    consumed.fill_bytes(0xff);
    counts.fill_bytes(0);
    status.fill_bytes(0);
    CUDA_CHECK(recv_x_ready::launch_collector(
        topk.ptr, std::is_same<Index, int64_t>::value, rows, num_topk,
        begin.ptr, end.ptr, ready.ptr, ranges, indices.ptr, rows, counts.ptr,
        consumed.ptr, status.ptr, timeout, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(status.get()[0] == recv_x_ready::kComplete);
    CHECK(consumed.get() == ends);
    const auto actual_counts = counts.get();
    const auto actual_indices = indices.get();
    for (int e = 0; e < 8; ++e) {
        std::vector<int> expected;
        for (int row = 0; row < rows; ++row) {
            bool matches = false;
            for (int k = 0; k < num_topk; ++k)
                matches |= route(row, k) == e;
            if (matches)
                expected.push_back(row);
        }
        CHECK(actual_counts[e] == static_cast<int>(expected.size()));
        std::vector<int> actual(actual_indices.begin() + e * rows,
                                actual_indices.begin() + e * rows + actual_counts[e]);
        std::sort(actual.begin(), actual.end());
        CHECK(actual == expected);
    }
    std::printf("PASS backlog int%zu, topk=%d, ranges=%d\n", sizeof(Index) * 8, num_topk, ranges);
}

// Overflow in a later 32-row segment must reject the entire tile before any
// out-of-capacity write or count publication.
void check_tile_overflow(unsigned long long timeout) {
    constexpr int ranges = 8, rows = 128, capacity = 40;
    DeviceArray<int> topk(rows), begin(ranges), end(ranges), ready(ranges), consumed(ranges);
    DeviceArray<int> indices(8 * capacity), counts(8), status(1);
    std::vector<int> begins, ends;
    for (int r = 0; r < ranges; ++r) {
        begins.push_back(r * 16);
        ends.push_back((r + 1) * 16);
    }
    topk.fill_bytes(0);
    begin.put(begins);
    end.put(ends);
    ready.put(ends);
    consumed.fill_bytes(0xff);
    indices.fill_bytes(0xff);
    counts.fill_bytes(0);
    status.fill_bytes(0);
    CUDA_CHECK(recv_x_ready::launch_collector(
        topk.ptr, false, rows, 1, begin.ptr, end.ptr, ready.ptr, ranges,
        indices.ptr, capacity, counts.ptr, consumed.ptr, status.ptr, timeout, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(status.get()[0] == recv_x_ready::kCapacityExceeded);
    CHECK(consumed.get() == ends);
    for (int count : counts.get()) CHECK(count == 0);
    for (int index : indices.get()) CHECK(index == -1);
    std::puts("PASS multi-segment capacity rejection");
}

void check_terminal(int begin_value, int end_value, int ready_value,
                    int expert, int capacity, int expected, unsigned long long timeout) {
    DeviceArray<int> topk(2), begin(1), end(1), ready(1), indices(8 * capacity);
    DeviceArray<int> counts(8), consumed(1), status(1);
    topk.put({expert, expert});
    begin.put({begin_value});
    end.put({end_value});
    ready.put({ready_value});
    counts.fill_bytes(0);
    consumed.fill_bytes(0xff);
    status.fill_bytes(0);
    CUDA_CHECK(recv_x_ready::launch_collector(
        topk.ptr, false, 2, 1, begin.ptr, end.ptr, ready.ptr, 1,
        indices.ptr, capacity, counts.ptr, consumed.ptr, status.ptr, timeout, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(status.get()[0] == expected);
    for (const auto count : counts.get())
        CHECK(count >= 0 && count <= capacity);
}

int main() {
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    CHECK(properties.major >= 9 && properties.concurrentKernels);
    const auto timeout = static_cast<unsigned long long>(properties.clockRate) * 10000;
    for (int repeat = 0; repeat < 3; ++repeat) {
        for (const int topk : {1, 2, 3, 4, 8, 16, 31, 32}) {
            check_incremental<int>(topk, timeout);
            check_incremental<int64_t>(topk, timeout);
        }
    }
    for (int topk = 1; topk <= 32; ++topk) {
        check_backlog<int>(topk, 23, timeout);
        check_backlog<int64_t>(topk, 23, timeout);
    }
    check_backlog<int>(8, 4101, timeout);
    check_backlog<int64_t>(3, 4101, timeout);
    check_tile_overflow(timeout);
    check_terminal(0, 2, 2, 0, 1, recv_x_ready::kCapacityExceeded, timeout);
    check_terminal(0, 3, 2, 0, 2, recv_x_ready::kInvalidRange, timeout);
    check_terminal(0, 2, 3, 0, 2, recv_x_ready::kInvalidProgress, timeout);
    check_terminal(0, 2, 2, 8, 2, recv_x_ready::kInvalidExpert, timeout);
    check_terminal(0, 0, 0, 0, 0, recv_x_ready::kComplete, timeout);
    check_terminal(0, 2, 2, -1, 0, recv_x_ready::kComplete, timeout);
    check_terminal(0, 2, -1, 0, 2, recv_x_ready::kIdleTimeout,
                   static_cast<unsigned long long>(properties.clockRate) * 20);
    std::puts("PASS all completion/error tests");
}
