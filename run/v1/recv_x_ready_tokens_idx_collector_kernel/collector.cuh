#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace recv_x_ready {

constexpr int kExperts = 8;
constexpr int kThreads = 256;
constexpr int kRowsPerRange = 4;
constexpr int kTileRows = kExperts * kRowsPerRange;

enum Status : int {
    kRunning = 0,
    kComplete = 1,
    kCapacityExceeded = 2,
    kInvalidRange = 3,
    kInvalidProgress = 4,
    kInvalidExpert = 5,
    kIdleTimeout = 6,
};

// All buffers are on the same GPU, in the same memory synchronization domain.
// Concurrent accesses to ready_end / ready_count / status use these helpers.
// The memory clobbers also prevent compiler reordering across publication.
#ifdef __CUDACC__
__device__ __forceinline__ int load_acquire(const int* address) {
    int value;
    asm volatile("ld.acquire.gpu.L1::no_allocate.global.u32 %0, [%1];"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void store_release(int* address, int value) {
    // Keep this polling/publication metadata out of L1 on Hopper. The release
    // still orders all preceding payload and routing stores for the acquire.
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;"
                 :: "l"(address), "r"(value) : "memory");
}

// Called once per range by its producer, including empty ranges. ready_end must
// have been initialized to -1 before either producer or collector can run.
__device__ __forceinline__ void initialize_range(
    int* range_begin, int* range_end, int* ready_end,
    int range_id, int begin, int end) {
    range_begin[range_id] = begin;
    range_end[range_id] = end;
    store_release(ready_end + range_id, begin);
}

// Caller must first complete the payload's TMA stores AND synchronize the
// receiver warp's metadata writes. This function itself does NOT wait for TMA.
__device__ __forceinline__ void publish_ready(
    int* ready_end, int range_id, int exclusive_row_end) {
    store_release(ready_end + range_id, exclusive_row_end);
}
#endif

// Inputs: contiguous recv_topk_idx[num_rows, num_topk], int32 or int64; immutable
// range descriptors once initialized; monotonic absolute ready_end values.
// Outputs: idx_list[8, capacity] (int32), ready_count[8] (int32).
// Workspace: consumed_end[num_ranges] initially -1; status[1] initially 0.
// ready_count must initially be zero. List entries need not be initialized.
// timeout_cycles == 0 disables the watchdog; otherwise it limits cycles without
// any newly gathered rows or newly completed ranges. Launches exactly ONE CTA.
cudaError_t launch_collector(
    const void* recv_topk_idx, bool topk_is_int64,
    int num_rows, int num_topk,
    const int* range_begin, const int* range_end, const int* ready_end,
    int num_ranges, int* idx_list, int capacity, int* ready_count,
    int* consumed_end, int* status, unsigned long long timeout_cycles,
    cudaStream_t stream);

}  // namespace recv_x_ready
