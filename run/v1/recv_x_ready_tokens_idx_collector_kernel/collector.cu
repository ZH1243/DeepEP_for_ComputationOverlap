#include "collector.cuh"

namespace recv_x_ready {

static_assert(kExperts == 8 && kTileRows == 32 && kThreads == 256,
              "the ballot compaction maps one tile row to each warp lane");

template <typename Index>
__global__ __launch_bounds__(kThreads, 1) void collect(
    const Index* __restrict__ topk, int num_rows, int num_topk,
    const int* __restrict__ range_begin, const int* __restrict__ range_end,
    const int* __restrict__ ready_end, int num_ranges,
    int* __restrict__ idx_list, int capacity, int* __restrict__ ready_count,
    int* __restrict__ consumed_end, int* __restrict__ status,
    unsigned long long timeout_cycles) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    constexpr unsigned full = 0xffffffffu;

    __shared__ int tile_rows[kTileRows];
    __shared__ unsigned tile_experts[kTileRows];
    __shared__ int finished_ranges;
    __shared__ int tile_activity;
    __shared__ int failure;

    if (tid == 0) {
        finished_ranges = 0;
        failure = kRunning;
    }
    __syncthreads();

    // The same warp owns an expert count during every compaction phase.
    int expert_count = 0;
    int range_group = 0;
    unsigned long long last_activity = clock64();

    while (true) {
        if (tid == 0)
            tile_activity = 0;
        __syncthreads();

        // Phase 1: all eight warps gather. Each polls one range in this group
        // and takes up to four rows. Rotate groups even when ranges are busy.
        // Lane 0 acquires the producer's publication; __syncwarp below transfers
        // that ordering to the lanes that load routing metadata.
        const int range = range_group + warp;
        int first = 0;
        int take = 0;
        if (lane == 0 && range < num_ranges) {
            const int published = load_acquire(ready_end + range);
            if (published < -1) {
                atomicCAS(&failure, kRunning, kInvalidProgress);
            } else if (published >= 0) {
                const int begin = range_begin[range];
                const int end = range_end[range];
                const int previous = consumed_end[range];
                first = previous == -1 ? begin : previous;
                if (begin < 0 || end < begin || end > num_rows) {
                    atomicCAS(&failure, kRunning, kInvalidRange);
                } else if (published < first || published > end) {
                    atomicCAS(&failure, kRunning, kInvalidProgress);
                } else {
                    take = min(published - first, kRowsPerRange);
                    consumed_end[range] = first + take;
                    if (take > 0)
                        atomicAdd(&tile_activity, take);
                    // Count an empty range on its first observation; count a
                    // nonempty range exactly when its final rows are gathered.
                    if (first + take == end && (previous == -1 || first < end)) {
                        atomicAdd(&finished_ranges, 1);
                        atomicAdd(&tile_activity, 1);
                    }
                }
            }
        }
        first = __shfl_sync(full, first, 0);
        take = __shfl_sync(full, take, 0);
        __syncwarp(full);

        #pragma unroll
        for (int j = 0; j < kRowsPerRange; ++j) {
            unsigned expert_bit = 0;
            bool bad_expert = false;
            if (j < take && lane < num_topk) {
                const auto expert = topk[static_cast<int64_t>(first + j) * num_topk + lane];
                bad_expert = expert < -1 || expert >= kExperts;
                if (expert >= 0 && expert < kExperts)
                    expert_bit = 1u << static_cast<unsigned>(expert);
            }
            // Read each routing entry once, retaining only the local membership
            // mask needed by compaction. Duplicate expert IDs yield one index.
            const unsigned mask = __reduce_or_sync(full, expert_bit);
            const bool invalid = __any_sync(full, bad_expert);
            if (lane == 0) {
                const int slot = warp * kRowsPerRange + j;
                tile_rows[slot] = j < take ? first + j : -1;
                tile_experts[slot] = mask;
                if (invalid)
                    atomicCAS(&failure, kRunning, kInvalidExpert);
            }
        }
        __syncthreads();

        // Phase 2: warp e compacts the full tile for expert e. Slots can have
        // holes, so a ballot and prefix popcount form a dense append.
        const bool matches = (tile_experts[lane] & (1u << warp)) != 0;
        const unsigned selected = __ballot_sync(full, matches);
        const int added = __popc(selected);
        const unsigned lower_lanes = (1u << lane) - 1u;
        const int offset = __popc(selected & lower_lanes);
        if (lane == 0 && added > capacity - expert_count)
            atomicCAS(&failure, kRunning, kCapacityExceeded);
        __syncthreads();

        if (failure != kRunning) {
            if (tid == 0)
                store_release(status, failure);
            return;
        }
        if (matches)
            idx_list[static_cast<int64_t>(warp) * capacity + expert_count + offset] = tile_rows[lane];
        __syncwarp(full);
        expert_count += added;
        if (lane == 0 && added != 0)
            store_release(ready_count + warp, expert_count);

        // No warp may reuse the shared tile until all expert warps finish.
        // Also carries every expert's publications to the status writer.
        __syncthreads();
        if (finished_ranges == num_ranges) {
            if (tid == 0)
                store_release(status, kComplete);
            return;
        }

        // Snapshot before the next barrier: lane 0 may reset tile_activity in
        // the next iteration while other warps are still backing off below.
        const int activity = tile_activity;
        if (tid == 0) {
            const auto now = clock64();
            if (activity != 0)
                last_activity = now;
            else if (timeout_cycles != 0 && now - last_activity >= timeout_cycles)
                failure = kIdleTimeout;
        }
        __syncthreads();
        if (failure != kRunning) {
            if (tid == 0)
                store_release(status, failure);
            return;
        }
        // Brief backoff on empty groups; never wait for a tile to fill.
        if (activity == 0)
            __nanosleep(64);
        range_group = num_ranges - range_group <= kExperts ? 0 : range_group + kExperts;
    }
}

cudaError_t launch_collector(
    const void* recv_topk_idx, bool topk_is_int64,
    int num_rows, int num_topk,
    const int* range_begin, const int* range_end, const int* ready_end,
    int num_ranges, int* idx_list, int capacity, int* ready_count,
    int* consumed_end, int* status, unsigned long long timeout_cycles,
    cudaStream_t stream) {
    if (num_rows < 0 || num_topk < 1 || num_topk > 32 || num_ranges < 1 || capacity < 0)
        return cudaErrorInvalidValue;
    if (topk_is_int64) {
        collect<<<1, kThreads, 0, stream>>>(
            static_cast<const int64_t*>(recv_topk_idx), num_rows, num_topk,
            range_begin, range_end, ready_end, num_ranges, idx_list, capacity,
            ready_count, consumed_end, status, timeout_cycles);
    } else {
        collect<<<1, kThreads, 0, stream>>>(
            static_cast<const int*>(recv_topk_idx), num_rows, num_topk,
            range_begin, range_end, ready_end, num_ranges, idx_list, capacity,
            ready_count, consumed_end, status, timeout_cycles);
    }
    return cudaGetLastError();
}

}  // namespace recv_x_ready
