#include "collector.cuh"

namespace recv_x_ready {

static_assert(kExperts == 8 && kTileRows == 128 && kThreads == 256,
              "eight expert warps compact four 32-row segments");

// Keep shared memory below the default launch limit without restricting the
// public range count. Unusually large invocations retain a global-state tail.
constexpr int kMaxCachedRanges = 4096;

template <typename Index, int TopK>
__global__ __launch_bounds__(kThreads, 1) void collect(
    const Index* __restrict__ topk, int num_rows, int num_topk,
    const int* __restrict__ range_begin, const int* __restrict__ range_end,
    const int* __restrict__ ready_end, int num_ranges,
    int* __restrict__ idx_list, int capacity, int* __restrict__ ready_count,
    int* __restrict__ consumed_end, int* __restrict__ status,
    unsigned long long timeout_cycles, GatherTable gather) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    constexpr unsigned full = 0xffffffffu;
    constexpr int segments = kTileRows / 32;
    // Power-of-two top-k values use one subgroup per row. Other values use
    // the generic full-warp reduction, with the runtime top-k as its stride.
    constexpr int width = TopK == 0 ? 32 : TopK;
    constexpr int rows_per_load = 32 / width;
    const int stride = TopK == 0 ? num_topk : TopK;

    extern __shared__ int range_state[];
    const int cached_ranges = min(num_ranges, kMaxCachedRanges);
    int* cursor = range_state;
    int* final_end = range_state + cached_ranges;
    __shared__ int tile_rows[kTileRows];
    __shared__ unsigned tile_experts[kTileRows];
    __shared__ int warp_rows[kExperts];
    __shared__ int warp_finished[kExperts];
    __shared__ int finished_ranges;
    __shared__ int failure;
    __shared__ int watchdog_status;
    __shared__ int bundle_count[kExperts];
    __shared__ int bundle_start[kExperts];
    __shared__ int table_rows;

    for (int r = tid; r < cached_ranges; r += kThreads) {
        cursor[r] = -1;
        final_end[r] = -1;
    }
    if (tid == 0) {
        table_rows = 0;
        finished_ranges = 0;
        failure = kRunning;
    }
    __syncthreads();

    int expert_count = 0;
    int written_count = 0;
    int range_group = 0;
    bool scan_activity = false;
    unsigned long long last_activity = clock64();
    int terminal = kRunning;
    while (true) {
        const int range = range_group + warp;
        int first = 0;
        int take = 0;
        int completed = 0;
        if (lane == 0 && range < num_ranges) {
            const bool cached = range < cached_ranges;
            const int previous = cached ? cursor[range] : consumed_end[range];
            // A finished cached range never needs another global poll.
            if (!cached || previous == -1 || previous != final_end[range]) {
                const int published = load_acquire(ready_end + range);
                if (published < -1) {
                    atomicCAS(&failure, kRunning, kInvalidProgress);
                } else if (published >= 0) {
                    // Validate/cache descriptors only after their publication.
                    const int begin = previous == -1 || !cached ? range_begin[range] : 0;
                    const int end = previous == -1 || !cached ? range_end[range] : final_end[range];
                    first = previous == -1 ? begin : previous;
                    if ((previous == -1 || !cached) &&
                        (begin < 0 || end < begin || end > num_rows)) {
                        atomicCAS(&failure, kRunning, kInvalidRange);
                    } else if (published < first || published > end) {
                        atomicCAS(&failure, kRunning, kInvalidProgress);
                    } else {
                        take = min(published - first, kRowsPerRange);
                        if (cached) {
                            cursor[range] = first + take;
                            final_end[range] = end;
                        } else {
                            consumed_end[range] = first + take;
                        }
                        completed = first + take == end && (previous == -1 || first < end);
                    }
                }
            }
        }
        first = __shfl_sync(full, first, 0);
        take = __shfl_sync(full, take, 0);
        // Transfer the leader's acquire to every routing-load lane.
        __syncwarp(full);
        if (lane == 0) {
            warp_rows[warp] = take;
            warp_finished[warp] = completed;
        }

        // A warp with no rows can skip all routing instructions, but must clear
        // its slots so a partial tile cannot reuse old membership masks.
        if (take == 0) {
            if (lane < kRowsPerRange)
                tile_experts[warp * kRowsPerRange + lane] = 0;
        } else {
            bool invalid = false;
            #pragma unroll
            for (int base = 0; base < kRowsPerRange; base += rows_per_load) {
                const int row = base + lane / width;
                const int entry = lane % width;
                unsigned bits = 0;
                if (row < take && entry < stride) {
                    const auto expert = topk[static_cast<int64_t>(first + row) * stride + entry];
                    invalid |= expert < -1 || expert >= kExperts;
                    if (expert >= 0 && expert < kExperts)
                        bits = 1u << static_cast<unsigned>(expert);
                }
                if constexpr (width == 32) {
                    bits = __reduce_or_sync(full, bits);
                } else {
                    #pragma unroll
                    for (int delta = width / 2; delta > 0; delta /= 2)
                        bits |= __shfl_down_sync(full, bits, delta, width);
                }
                if (entry == 0 && row < kRowsPerRange) {
                    const int slot = warp * kRowsPerRange + row;
                    tile_rows[slot] = row < take ? first + row : -1;
                    tile_experts[slot] = bits;
                }
            }
            const bool any_invalid = __any_sync(full, invalid);
            if (lane == 0 && any_invalid)
                atomicCAS(&failure, kRunning, kInvalidExpert);
        }
        __syncthreads();

        // Every thread computes the same small reduction. Per-warp slots avoid
        // contended activity/completion atomics and a separate reset barrier.
        int rows = 0;
        int newly_finished = 0;
        #pragma unroll
        for (int w = 0; w < kExperts; ++w) {
            rows += warp_rows[w];
            newly_finished += warp_finished[w];
        }
        if (tid == 0)
            finished_ranges += newly_finished;
        const bool activity = rows != 0 || newly_finished != 0;
        scan_activity |= activity;

        unsigned selected[segments];
        int added = 0;
        if (rows != 0) {
            #pragma unroll
            for (int s = 0; s < segments; ++s) {
                selected[s] = __ballot_sync(full, (tile_experts[s * 32 + lane] & (1u << warp)) != 0);
                added += __popc(selected[s]);
            }
            if (lane == 0 && added > capacity - expert_count)
                atomicCAS(&failure, kRunning, kCapacityExceeded);
        }
        // No output from this tile is published if any warp reports an error.
        __syncthreads();
        if (failure != kRunning) {
            terminal = failure;
            break;
        }
        if (rows != 0) {
            int output = expert_count;
            #pragma unroll
            for (int s = 0; s < segments; ++s) {
                const unsigned mask = selected[s];
                if ((mask & (1u << lane)) != 0) {
                    const int offset = __popc(mask & ((1u << lane) - 1u));
                    idx_list[static_cast<int64_t>(warp) * capacity + output + offset] = tile_rows[s * 32 + lane];
                }
                output += __popc(mask);
            }
            __syncwarp(full);
            expert_count += added;
            if (lane == 0 && added != 0)
                store_release(ready_count + warp, expert_count);
        }
        // Protect both the tile and the per-warp bookkeeping from reuse; also
        // carry all expert publications to the terminal status writer.
        __syncthreads();
        if (gather.ready_rows != nullptr) {
            // Each expert owns a warp. Reserve entire N-group bundles in a
            // deterministic prefix within this iteration; arrival order across
            // iterations is intentionally dynamic.
            const int pending = expert_count - written_count;
            const bool final = finished_ranges == num_ranges;
            const int batches = pending / gather.cluster_rows +
                (final && pending % gather.cluster_rows != 0);
            if (lane == 0)
                bundle_count[warp] = batches;
            __syncthreads();
            if (tid == 0) {
                int64_t next = table_rows;
                for (int e = 0; e < kExperts; ++e) {
                    bundle_start[e] = static_cast<int>(next);
                    next += static_cast<int64_t>(bundle_count[e]) * gather.num_n_groups;
                }
                if (next > gather.capacity_rows)
                    failure = kTableCapacityExceeded;
                else
                    table_rows = static_cast<int>(next);
            }
            __syncthreads();
            if (failure != kRunning) {
                terminal = failure;
                break;
            }
            for (int b = 0; b < batches; ++b) {
                const int take = min(expert_count - written_count, gather.cluster_rows);
                for (int n = 0; n < gather.num_n_groups; ++n) {
                    const int row = bundle_start[warp] + b * gather.num_n_groups + n;
                    int* entry = gather.table + static_cast<int64_t>(row) * (2 + gather.cluster_rows);
                    if (lane == 0) {
                        entry[0] = warp;
                        entry[1] = n * gather.group_size;
                    }
                    for (int i = lane; i < gather.cluster_rows; i += 32)
                        entry[2 + i] = i < take
                            ? idx_list[static_cast<int64_t>(warp) * capacity + written_count + i] : -1;
                }
                written_count += take;
            }
            __syncwarp(full);
            if (lane == 0 && batches != 0)
                store_release(gather.written_count + warp, written_count);
            // Transfer all table/index/payload visibility to the prefix writer.
            __syncthreads();
            if (tid == 0)
                store_release(gather.ready_rows, table_rows);
            __syncthreads();
        }
        if (finished_ranges == num_ranges) {
            terminal = kComplete;
            break;
        }
        if (tid == 0) {
            if (timeout_cycles != 0) {
                const auto now = clock64();
                if (activity)
                    last_activity = now;
                else if (now - last_activity >= timeout_cycles)
                    failure = kIdleTimeout;
            }
            // Snapshot separately: a fast warp may report a gather error in
            // the NEXT iteration before a slow warp reads this decision.
            watchdog_status = failure;
        }
        __syncthreads();
        if (watchdog_status != kRunning) {
            terminal = watchdog_status;
            break;
        }
        const bool end_of_scan = num_ranges - range_group <= kExperts;
        if (end_of_scan) {
            // An empty group alone says nothing about backlog in later groups.
            if (!scan_activity)
                __nanosleep(64);
            scan_activity = false;
            range_group = 0;
        } else {
            range_group += kExperts;
        }
    }
    // consumed_end is private workspace, not live publication metadata. Flush
    // it on every terminal path, including errors, before publishing status.
    for (int r = tid; r < cached_ranges; r += kThreads)
        consumed_end[r] = cursor[r];
    __syncthreads();
    if (tid == 0)
        store_release(status, terminal);
}

template <typename Index>
cudaError_t launch_typed(
    const void* topk, int num_rows, int num_topk,
    const int* range_begin, const int* range_end, const int* ready_end,
    int num_ranges, int* idx_list, int capacity, int* ready_count,
    int* consumed_end, int* status, unsigned long long timeout_cycles,
    cudaStream_t stream, GatherTable gather) {
    const size_t shared_bytes = 2 * sizeof(int) * (num_ranges < kMaxCachedRanges ? num_ranges : kMaxCachedRanges);
#define LAUNCH(TOPK) \
    collect<Index, TOPK><<<1, kThreads, shared_bytes, stream>>>( \
        static_cast<const Index*>(topk), num_rows, num_topk, \
        range_begin, range_end, ready_end, num_ranges, idx_list, capacity, \
        ready_count, consumed_end, status, timeout_cycles, gather)
    switch (num_topk) {
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 4: LAUNCH(4); break;
        case 8: LAUNCH(8); break;
        case 16: LAUNCH(16); break;
        case 32: LAUNCH(32); break;
        default: LAUNCH(0); break;
    }
#undef LAUNCH
    return cudaGetLastError();
}

cudaError_t launch_collector(
    const void* recv_topk_idx, bool topk_is_int64,
    int num_rows, int num_topk,
    const int* range_begin, const int* range_end, const int* ready_end,
    int num_ranges, int* idx_list, int capacity, int* ready_count,
    int* consumed_end, int* status, unsigned long long timeout_cycles,
    cudaStream_t stream, GatherTable gather) {
    if (num_rows < 0 || num_topk < 1 || num_topk > 32 || num_ranges < 1 || capacity < 0)
        return cudaErrorInvalidValue;
    if (gather.ready_rows != nullptr &&
        (gather.written_count == nullptr || gather.capacity_rows < 0 ||
         (gather.capacity_rows > 0 && gather.table == nullptr) ||
         gather.cluster_rows < 1 || gather.cluster_rows > INT32_MAX - 2 ||
         gather.cluster_rows == 2 || gather.num_n_groups < 1 || gather.group_size < 1 ||
         static_cast<int64_t>(gather.num_n_groups) * gather.group_size > INT32_MAX))
        return cudaErrorInvalidValue;
    if (topk_is_int64) {
        return launch_typed<int64_t>(recv_topk_idx, num_rows, num_topk,
            range_begin, range_end, ready_end, num_ranges, idx_list, capacity,
            ready_count, consumed_end, status, timeout_cycles, stream, gather);
    }
    return launch_typed<int>(recv_topk_idx, num_rows, num_topk,
        range_begin, range_end, ready_end, num_ranges, idx_list, capacity,
        ready_count, consumed_end, status, timeout_cycles, stream, gather);
}

}  // namespace recv_x_ready
