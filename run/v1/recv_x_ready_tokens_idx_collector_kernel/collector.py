"""PyTorch launch wrapper. See README.md for the required producer protocol."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Optional

import torch


_STATUS = {
    0: "running",
    1: "complete",
    2: "expert list capacity exceeded",
    3: "invalid range descriptor",
    4: "invalid readiness frontier",
    5: "routing entry is outside [-1, 7]",
    6: "collector idle timeout: producer stopped publishing or did not run",
    7: "gather table capacity exceeded",
}


@lru_cache(maxsize=1)
def build():
    """Compile/load before launching a producer; requires CUDA, PyTorch, ninja."""
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent
    return load(
        name="deepep_v1_recv_x_ready_collector",
        sources=[str(root / "torch_bindings.cpp"), str(root / "collector.cu")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "-lineinfo",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_90,code=compute_90",
        ],
    )


@lru_cache(maxsize=None)
def _device_clock_rate_khz(device_index: int) -> int:
    """Query once per device, before a dispatch producer is submitted."""
    with torch.cuda.device(device_index):
        return int(build().get_device_clock_rate_khz())


@dataclass
class CollectorState:
    range_begin: torch.Tensor
    range_end: torch.Tensor
    ready_end: torch.Tensor
    indices: torch.Tensor
    ready_count: torch.Tensor
    consumed_end: torch.Tensor
    status: torch.Tensor
    gather_table: torch.Tensor
    gather_ready_rows: torch.Tensor
    written_count: torch.Tensor
    gather_num_n_groups: int
    gather_group_size: int
    initialized: torch.cuda.Event
    clock_rate_khz: int
    _launched: bool = False

    def tensors(self):
        return (
            self.range_begin, self.range_end, self.ready_end, self.indices,
            self.ready_count, self.consumed_end, self.status,
            self.gather_table, self.gather_ready_rows, self.written_count,
        )


@dataclass
class CollectorRun:
    state: CollectorState
    event: torch.cuda.Event
    stream: torch.cuda.Stream
    recv_topk_idx: torch.Tensor  # Keep the asynchronously read input alive.

    def wait(self) -> CollectorState:
        """Host wait for final output and raise on protocol/capacity/timeout errors."""
        self.event.synchronize()
        code = int(self.state.status.item())
        if code != 1:
            raise RuntimeError(f"Ready-index collector: {_STATUS.get(code, f'unknown status {code}')}")
        return self.state


def allocate_state(
    num_ranges: int,
    num_rows: int,
    *,
    capacity: Optional[int] = None,
    device=None,
    gather_cluster_rows: int = 512,
    gather_num_n_groups: int = 2,
    gather_group_size: int = 8,
) -> CollectorState:
    """Allocate fresh state; producers must wait for state.initialized.

    By default each expert can hold every received row. Buffers must not alias
    each other or producer payloads. One state belongs to exactly one dispatch.
    Gather geometry defaults to QuACK tile_m=tile_n=256, cluster_m=2, N=4096,
    max_swizzle_size=8. The table allocation is a capacity, not a final work count;
    only its gather_ready_rows prefix is committed (see README.md).
    """
    if capacity is None:
        capacity = num_rows
    for name, value, minimum in (
        ("num_ranges", num_ranges, 1), ("num_rows", num_rows, 0), ("capacity", capacity, 0)
    ):
        if not isinstance(value, int) or not minimum <= value <= 2**31 - 1:
            raise ValueError(f"{name} must be an integer in [{minimum}, INT_MAX]")
    for name, value in (("gather_cluster_rows", gather_cluster_rows),
                        ("gather_num_n_groups", gather_num_n_groups),
                        ("gather_group_size", gather_group_size)):
        if not isinstance(value, int) or not 1 <= value <= 2**31 - 3:
            raise ValueError(f"{name} must be a positive int32 integer")
    if gather_cluster_rows == 2:
        raise ValueError("cluster_rows=2 produces width 4, which QuACK interprets as non-indexed")
    table_rows = 8 * ((capacity + gather_cluster_rows - 1) // gather_cluster_rows) * gather_num_n_groups
    if max(table_rows, gather_num_n_groups * gather_group_size) > 2**31 - 1:
        raise ValueError("gather table row count and N geometry must fit in int32")
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise ValueError("state must be allocated on a CUDA device")
    with torch.cuda.device(device):
        # Query device properties before dispatch is enqueued. On some Hopper
        # systems cudaGetDeviceProperties can block behind an executing
        # cooperative kernel, which would delay submission of the collector.
        clock_rate_khz = _device_clock_rate_khz(torch.cuda.current_device())
        options = dict(dtype=torch.int32, device=device)
        state = CollectorState(
            range_begin=torch.empty(num_ranges, **options),
            range_end=torch.empty(num_ranges, **options),
            ready_end=torch.full((num_ranges,), -1, **options),
            indices=torch.empty((8, capacity), **options),
            ready_count=torch.zeros(8, **options),
            consumed_end=torch.full((num_ranges,), -1, **options),
            status=torch.zeros(1, **options),
            gather_table=torch.empty((table_rows, 2 + gather_cluster_rows), **options),
            gather_ready_rows=torch.zeros(1, **options),
            written_count=torch.zeros(8, **options),
            gather_num_n_groups=gather_num_n_groups,
            gather_group_size=gather_group_size,
            initialized=torch.cuda.Event(),
            # clock_rate is kHz, equivalently cycles per millisecond.
            clock_rate_khz=clock_rate_khz,
        )
        state.initialized.record()
    return state


def launch(
    recv_topk_idx: torch.Tensor,
    state: CollectorState,
    *,
    stream: Optional[torch.cuda.Stream] = None,
    timeout_ms: float = 10000.0,
) -> CollectorRun:
    """Launch asynchronously, without waiting for dispatch completion.

    No dependency on the producer stream is inserted. The producer must publish
    descriptors and progress using collector.cuh's release-store protocol.
    Standard PyTorch reads of ready_count are NOT a live consumer protocol.
    """
    if state._launched:
        raise ValueError("allocate fresh state for each launch; reusing active/stale state is unsafe")
    if not isinstance(timeout_ms, (int, float)) or not math.isfinite(timeout_ms) or not 0 <= timeout_ms <= 3600000:
        raise ValueError("timeout_ms must be finite and in [0, 3600000]; zero disables the watchdog")
    timeout_cycles = int(timeout_ms * state.clock_rate_khz)
    extension = build()
    device = state.ready_end.device
    with torch.cuda.device(device):
        if stream is None:
            stream = torch.cuda.Stream(device=device)
        if stream.device != device:
            raise ValueError("collector stream and state must be on the same GPU")
        with torch.cuda.stream(stream):
            stream.wait_event(state.initialized)
            extension.launch(recv_topk_idx, *state.tensors(),
                             state.gather_num_n_groups, state.gather_group_size, timeout_cycles)
            state._launched = True
            for tensor in (recv_topk_idx, *state.tensors()):
                tensor.record_stream(stream)
            event = torch.cuda.Event()
            event.record()
    return CollectorRun(state, event, stream, recv_topk_idx)
