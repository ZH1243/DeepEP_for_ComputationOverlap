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

## Indexed gather-table output

The Python collector now also prepares QuACK's **single-buffer indexed gather**
format, with `recv_x` as X. No GEMM is launched. The CTA remains 256 threads.

| Buffer | Type and shape | Meaning |
|---|---|---|
| `gather_table` | int32 `[Q_capacity, 2 + C]` | Rows `(expert_id, cid_n_base, recv_x_idx_0, ..., recv_x_idx_{C-1})` |
| `gather_ready_rows` | int32 `[1]`, initially zero | Exclusive committed table-row prefix |
| `written_count` | int32 `[8]`, initially zero | Actual tokens copied into table bundles per expert; excludes padding and repeated N-group copies |

Set `C = gather_cluster_rows = tile_m * cluster_m`. With
`clusters_n = ceil(gemm_N / tile_n)`, set `gather_group_size` to
`min(max_swizzle_size, clusters_n)` and `gather_num_n_groups` to
`clusters_n / gather_group_size` (must divide exactly). Python defaults are
C=512, group size=8, two N groups, matching the referenced QuACK runner's
256×256 tile, cluster_m=2, N=4096 defaults. These parameters describe the table;
the future GEMM must also use a supported matching kernel configuration.
Width 4 (C=2) is rejected because QuACK treats it as the non-indexed format.

After compaction, each expert warp emits all available full C-token batches.
Only after **all ranges finish** does it flush a remaining partial batch with
trailing `-1` indices; empty experts emit no rows. Each batch emits consecutive
N-group rows with bases `0, group_size, ...`. Every N-group row repeats the same
indices. The CTA assigns contiguous bundle slots, synchronizes all writes, then
thread 0 release-stores `gather_ready_rows`. A GPU consumer must acquire this
flag before reading the table and payload. `written_count` is per-expert
progress; it is not the table's global commit signal. Its update can precede
that global commit. Previously committed table rows are immutable.

Bundle order follows observed readiness, so experts may be interleaved. For
bundle b, QuACK's output rows are `[b*C, (b+1)*C)`, not fixed expert-contiguous
segments. A future output/reference or down-projection path must honor that
ordering. The table contains direct recv_x row indices, not pointers, and does
not copy recv_x. Keep recv_x alive through any future consumer.

Allocation uses `Q_capacity = 8 * ceil(capacity / C) * num_n_groups` without a
host count read or dispatch completion wait. Only
`gather_table[:gather_ready_rows]` is valid. On successful completion,
`written_count == ready_count` and final Q is
`sum(ceil(ready_count[e] / C)) * num_n_groups`.
**QuACK currently schedules by table.shape[0]**, so do not pass the entire
worst-case allocation to a live GEMM: it would wait on unused rows. Future live
integration needs an exact Q from routing metadata before launching GEMM, or
scheduler end-of-stream support. A post-completion consumer can use a view of
the final Q rows. This step deliberately launches only dispatch and collector.

The existing dispatch command with `--with-collector` now allocates and verifies
the table too. Optional arguments are `--gather-tile-m`, `--gather-cluster-m`,
`--gather-tile-n`, `--gather-output-dim`, and `--gather-max-swizzle-size`.
For a gated projection, pass the full (doubled) GEMM N width as output dim.
The test prints `gather_ready_rows` after completion and validates every row,
N-group replica, padding slot, and per-expert token sequence.

Raw CUDA callers may append a `GatherTable` to `launch_collector`; the default
empty struct preserves index-only operation. Table overflow reports status 7
without publishing any of the failed iteration's bundles. Earlier table
prefixes remain valid; index counts can be ahead of written counts on failure.

## Files and operation

- `collector.cuh`: raw CUDA launch API and producer/consumer publication helpers.
- `collector.cu`: persistent collector kernel; independent of PyTorch/DeepEP.
- `collector.py`, `torch_bindings.cpp`: lazy-built PyTorch extension and launch API.
- `test_collector.cu`: concurrent producer/collector/consumer protocol test.
- `test_torch.py`: PyTorch wrapper and final-output checks.

The **same eight warps** alternate between gathering and expert compaction.
In one gathering phase, warp `w` polls range `range_group + w`, acquire-loads
its frontier, and takes at most **16 newly ready rows**. The shared tile holds
up to **128 row indices and membership masks** (1 KB); unused slots have zero
membership. Partial tiles are processed immediately, without waiting to fill.

Top-k 1/2/4/8/16/32 has compile-time specializations: a power-of-two subgroup
loads one row, so top-k 8 loads four rows per warp instruction. Subgroup OR
reductions build membership masks and deduplicate expert IDs. Other top-k
values retain a generic full-warp reduction. Both int32 and int64 are supported.

After a CTA barrier, warp `e` compacts four 32-row segments for expert `e`.
Ballot/popcount gives matching lanes consecutive append positions. Capacity is
checked for the entire tile before any tile output is written. The warp writes
all segments, synchronizes, and release-stores one updated committed count.
CTA synchronization protects shared tile reuse and terminal publication.

Range cursors and immutable final offsets are cached in dynamic shared memory
for the first 4096 ranges (8 bytes per cached range). Initial descriptors are
validated after acquire; the initial begin is then represented by the advancing
cursor. Completed cached ranges skip further polling. Larger range counts use
the existing global workspace for the uncached tail, avoiding a new API limit.
Cached `consumed_end` entries are flushed on every terminal exit, including
errors; this workspace is **not a live progress signal**. Use `ready_count`.

Per-warp shared slots replace activity/completion atomics. Empty groups skip
routing work and compaction. Range groups rotate after every tile, and a short
`__nanosleep` backoff occurs only after a **complete scan with no activity**.
Polling acquires and progress publications retain the Hopper L1 no-allocate
hint and the existing release/acquire protocol.

There are no global atomic increments on expert counts. Shared-memory atomics
record errors only. List order depends on observed
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

# Buffer.dispatch first runs a cooperative metadata-notification kernel and
# then enqueues the data-dispatch producer. Preallocate recv_topk_idx so the
# collector can be launched immediately after dispatch returns, without first
# waiting for producer completion.
# recv_x, recv_topk_idx_view, ..., dispatch_event = buffer.dispatch(
#     ..., async_finish=True, publish_ready_tokens=True,
#     ready_token_state=(state.range_begin, state.range_end, state.ready_end),
#     recv_topk_idx_buffer=recv_topk_idx)
run = launch(recv_topk_idx, state, timeout_ms=10000)
# Only now wait for dispatch_event if final dispatch output is needed.
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

Device capability and clock-rate discovery happens in `allocate_state`, before
the dispatch producer is enqueued. The hot `launch` path receives precomputed
watchdog cycles and performs no `cudaGetDeviceProperties` call, since that API
can delay collector submission behind an executing cooperative kernel.

For earliest overlap from Python, use `recv_topk_idx_buffer`; without it the v1
API allocates the routing output internally and the pointer is unavailable
until dispatch returns. Do not launch the persistent collector before
`Buffer.dispatch`: DeepEP first launches a cooperative metadata-notification
kernel, and a resident collector can prevent its admission. Launch the collector
after asynchronous `Buffer.dispatch` returns but before calling
`dispatch_event.current_stream_wait()`. See
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
| 7 | Gather table capacity exceeded |

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
committed gather-table bundle. This checks incremental publication, including visibility
of simulated payload stores through both release/acquire handoffs. Final lists
are compared against a CPU membership reference. Coverage includes multiple
polling groups, uneven/empty ranges, partial tiles, int32/int64 routing, top-k
specializations and generic fallback, duplicate memberships, repeated invocations with fresh state, capacity
overflow, invalid descriptors/frontiers/experts, and timeout. All-ready backlog
checks cover every top-k from 1 through 32 in both dtypes, repeated 128-row
tiles, partial tiles, more than 4096 ranges, and whole-tile capacity rejection. The simulated
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

## Rerunning after collector optimization

The dispatch test command is unchanged; Python state now includes gather outputs. In a fresh Python
process, `build()` automatically rebuilds the collector extension from the
changed CUDA source. No DeepEP extension rebuild is needed for this collector-only
change. Run the standalone protocol test above first, then your existing
`deepep_v1_internode_dispatch_with_ready_tokens_collector.py --with-collector`
command and compare dispatch duration, collector duration, and collector finish
lag in Nsight Systems. The script's printed dispatch timing is not a separate
collector-duration measurement. GPU correctness and performance must be checked
on Hopper; the optimization does not assume a particular measured speedup.
