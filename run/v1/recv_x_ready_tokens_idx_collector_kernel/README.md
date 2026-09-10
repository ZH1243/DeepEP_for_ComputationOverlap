# DeepEP v1 `recv_x` ready-index collector

Standalone CUDA collector for the proposed internode dispatch readiness
protocol. It launches **one CTA of 256 threads (eight warps)** and builds eight
local expert index lists as receiver ranges advance. It never reads `recv_x`,
scales, or weights: its only token input is `recv_topk_idx`.

The DeepEP v1 internode dispatch kernel now has an opt-in publication path.
`Buffer.dispatch(..., publish_ready_tokens=True)` accepts the three descriptor
tensors through `ready_token_state=(range_begin, range_end, ready_end)`. The
ordinary path selects a separate CUDA template specialization and executes no
publication branch or store. The code targets CUDA 12.9 / Hopper.

## Files and operation

- `collector.cuh`: raw CUDA launch API and producer/consumer publication helpers.
- `collector.cu`: persistent collector kernel; independent of PyTorch/DeepEP.
- `collector.py`, `torch_bindings.cpp`: lazy-built PyTorch extension and launch API.
- `test_collector.cu`: concurrent producer/collector/consumer protocol test.
- `test_torch.py`: PyTorch wrapper and final-output checks.

The **same eight warps** alternate between gathering and expert compaction.
In one gathering phase, warp `w` polls range `range_group + w`, acquire-loads
its frontier, and takes at most four newly ready rows. Its lanes cooperatively
read each row's top-k entries and reduce them to an eight-bit expert membership
mask. The CTA's shared tile holds up to 32 row indices and membership masks;
unused slots carry a zero mask. This avoids rereading routing metadata once
per expert. No token payload enters the shared tile.

After a CTA barrier, warp `e` processes all 32 tile slots for expert `e`.
Ballot/popcount gives matching lanes consecutive append positions. The warp
writes those entries, synchronizes, and its leader release-stores the new
committed count. Another CTA barrier protects shared tile reuse. Range groups
rotate after every tile even when one range has a large backlog. Partial tiles
are processed immediately. Empty polls use a short `__nanosleep` backoff.
Polling acquires and progress publications use Hopper's L1 no-allocate cache
hint, keeping the small synchronization surface in L2 and avoiding L1 cache
pollution in both the dispatch receiver and the collector.

There are no global atomic increments on expert counts. Shared-memory atomics
maintain error/activity/completion bookkeeping. List order depends on observed
arrival progress, is not globally sorted, and is not deterministic across runs.
Repeated expert IDs in one row are deduplicated: a row appears once per expert.

## Memory contract

All tensors are contiguous and on the same GPU. Publication and consumption
must use the same CUDA memory synchronization domain. Buffers must not alias.

| Buffer | Type and shape | Meaning / initial state |
|---|---|---|
| `recv_topk_idx` | int32 or int64 `[num_rows, topk]` | Local expert IDs `0..7`, or `-1`; `1 <= topk <= 32` |
| `range_begin` | int32 `[num_ranges]` | Inclusive absolute row index; producer initializes |
| `range_end` | int32 `[num_ranges]` | Exclusive absolute final row index; producer initializes |
| `ready_end` | int32 `[num_ranges]` | `-1` initially; then monotonic absolute ready frontier |
| `indices` | int32 `[8, capacity]` | Output lists; uninitialized entries are not readable |
| `ready_count` | int32 `[8]` | Zero initially; committed output lengths |
| `consumed_end` | int32 `[num_ranges]` | Private collector workspace, initially `-1` |
| `status` | int32 `[1]` | Initially `0`; terminal status published by collector |

Each range corresponds to `(channel_id, src_nvl_rank, src_rdma_rank)`. Flatten
this tuple identically on both sides, for example:

```cpp
int range_id = (channel_id * 8 + src_nvl_rank) * num_rdma_ranks + src_rdma_rank;
```

For 20 dispatch CTAs, there are ten channels and `80 * num_rdma_ranks` ranges.
Ranges must be disjoint and cover the real received rows that should be
collected; exclude padding. Descriptor overlap is a producer error and is not
checked by the collector. It would cause duplicate indices. Counts/indices are
32-bit, so row indices and capacities must fit in `INT_MAX`.

The invariant for an initialized range is:

```text
0 <= range_begin <= ready_end <= range_end <= num_rows
Every row in [range_begin, ready_end) has complete payload AND routing metadata.
```

Descriptors become immutable when initialized. The collector may skip
intermediate frontier values and still recover every row. Never modify a
published row or reset progress while producer, collector, or consumers still
use this invocation's buffers. Allocate fresh state for each dispatch.

`capacity` is uniform across experts. The default Python allocation uses
`capacity=num_rows`, sufficient even if every row belongs to every expert.
Smaller capacity is allowed when justified by known routing counts. Per-expert
alignment counts are capacities, not actual committed lengths.

## DeepEP receiver integration

The integration is implemented in `csrc/kernels/legacy/internode.cu` and
threaded through the C++ and Python legacy-buffer APIs. Initialization of
`ready_end=-1` must finish before either kernel starts.

In `csrc/kernels/legacy/internode.cu`, after the receiver obtains its offsets
and before its token loop, each lane representing an RDMA rank initializes its
range, including empty ranges:

```cpp
if (lane_id < kNumRDMARanks) {
    int r = (channel_id * 8 + src_nvl_rank) * kNumRDMARanks + lane_id;
    // At this point total_offset is this lane's initial absolute output offset.
    recv_x_ready::initialize_range(
        range_begin, range_end, ready_end, r,
        total_offset, total_offset + end_offset - start_offset);
}
```

At the **end of each token copy**, after the existing full TMA store wait and
warp synchronization:

```cpp
tma_store_wait<0>();  // Existing cp.async.bulk.wait_group 0, NOT .read.
__syncwarp();        // Existing synchronization, including routing/scales writes.
if (lane_id == meta.src_rdma_rank) {
    int r = (channel_id * 8 + src_nvl_rank) * kNumRDMARanks + lane_id;
    recv_x_ready::publish_ready(ready_end, r, total_offset);
}
```

The helper only performs a GPU-scope release store; it does not wait for TMA.
The publishing lane may differ from the TMA issuer, which is why the existing
warp synchronization must stay. Do not publish on the earlier shared-memory
load barrier or directly from the NVLink ring tail. The current API rejects
publication for cached dispatch because that path does not produce new top-k
metadata.

These receiver edits require rebuilding `deep_ep._C`, even with an editable
DeepEP installation. The standalone collector extension builds itself
separately and does not rebuild DeepEP.

## PyTorch API and stream ordering

From `run/v1` (in the existing CUDA 12.9 conda environment):

```python
from recv_x_ready_tokens_idx_collector_kernel import allocate_state, build, launch

build()  # Compile/load before starting any producer.
max_recv_rows = num_local_tokens * ep_size
state = allocate_state(num_ranges=80 * num_rdma_ranks, num_rows=max_recv_rows)
recv_topk_idx = torch.empty(
    (max_recv_rows, num_topk), dtype=topk_dtype, device="cuda")

# Arrange for the producer to wait on state.initialized. For a collector that
# starts before dispatch, preallocate the recv_topk_idx output as well.
run = launch(recv_topk_idx, state, timeout_ms=10000)

# Enqueue the producer on its independent stream without waiting for run.event:
# buffer.dispatch(..., publish_ready_tokens=True,
#                 ready_token_state=(state.range_begin, state.range_end, state.ready_end),
#                 recv_topk_idx_buffer=recv_topk_idx)
# During execution, a custom GPU consumer acquire-loads state.ready_count and
# reads only committed prefixes. Launch that consumer without a completion wait.

# For FINAL results only (host synchronization):
run.wait()
expert_0_rows = state.indices[0, :state.ready_count[0].item()]
```

`launch` defaults to a new CUDA stream, waits only for `state.initialized`, and
returns immediately after enqueueing the collector. It records the collector
stream on all tensors to protect allocator lifetimes. The producer and any
downstream consumer must also retain/record tensors for their own streams.
The caller must retain `recv_x`/scales/weights until computation finishes;
the collector wrapper does not receive or manage those tensors.

For earliest overlap from Python, use `recv_topk_idx_buffer`; without it the v1
API allocates the routing output internally and the pointer is unavailable
until dispatch returns. Do not enqueue the collector after
`dispatch_event.current_stream_wait()` on the same stream. See
`run/v1/deepep_v1_internode_dispatch_with_ready_tokens_collector.py` for the
complete stream ordering.

For already-complete input, ordinary tensor initialization is fine when ordered
before the collector by an event, as in `test_torch.py`. During overlapping
production, ordinary `.fill_()`, plain CUDA stores, or Python polling are not
substitutes for the release/acquire protocol.

A live GPU consumer uses `recv_x_ready::load_acquire(ready_count + expert)`
before reading list entries and payloads. If a leader acquires for other threads,
it must transfer that ordering with an appropriate warp/CTA synchronization.
Acquire the terminal `status` as well, so a consumer does not spin indefinitely
after a collector error. A downstream TMA consumer must also obey its own
asynchronous-copy synchronization requirements.

One CTA occupies at most one SM at a time; it does not reserve an exclusive SM
or guarantee concurrent scheduling. Leave enough resources for the producer
and consumers to run. Keep them in independent nonblocking streams, without
circular completion dependencies. A collector idle timeout is not cancellation
of the producer. Join every participating stream before freeing/reusing buffers.

| Status | Meaning |
|---|---|
| 0 | Running |
| 1 | All ranges consumed and final counts published |
| 2 | Expert list capacity exceeded |
| 3 | Invalid range boundaries |
| 4 | Invalid observed readiness frontier |
| 5 | Routing value outside `[-1, 7]` |
| 6 | No gathering/completion progress before idle timeout |

The watchdog is approximate (`clock64`, converted using GPU clock rate).
`timeout_ms=0` disables it. On an error, previously committed prefixes remain
valid, but output is incomplete and must not be treated as a successful dispatch.
The collector does not detect every producer protocol violation (for example,
unobserved frontier regression or a premature release before payload completion).

## Build and check on Hopper

From the repository root, standalone concurrent protocol test:

```bash
nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo \
  run/v1/recv_x_ready_tokens_idx_collector_kernel/collector.cu \
  run/v1/recv_x_ready_tokens_idx_collector_kernel/test_collector.cu \
  -o /tmp/test_recv_x_ready_collector
/tmp/test_recv_x_ready_collector
```

The synthetic producer withholds final rows until a live consumer observes a
committed list entry. This checks incremental publication, including visibility
of simulated payload stores through both release/acquire handoffs. Final lists
are compared against a CPU membership reference. Coverage includes multiple
polling groups, uneven/empty ranges, partial tiles, int32/int64 routing, top-k
1/8/32, duplicate memberships, repeated invocations with fresh state, capacity
overflow, invalid descriptors/frontiers/experts, and timeout. The simulated
producer uses ordinary CUDA stores; actual DeepEP TMA integration needs its own
correctness and performance checks.

Do not use `CUDA_LAUNCH_BLOCKING=1` or tools that serialize kernels for the
concurrent handshake test: its purpose depends on concurrent execution.

PyTorch wrapper checks:

```bash
python run/v1/recv_x_ready_tokens_idx_collector_kernel/test_torch.py
```

Measure baseline dispatch, dispatch with publications, and dispatch with both
publications and the collector separately. Also measure first useful expert
batch latency, publication delay, and collector backlog. One CTA and a small
metadata footprint do not imply zero dispatch slowdown or sufficient collector
throughput for every routing workload.
