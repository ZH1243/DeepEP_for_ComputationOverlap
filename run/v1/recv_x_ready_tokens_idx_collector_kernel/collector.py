"""PyTorch launch wrapper. See README.md for the required producer protocol."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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


@dataclass
class CollectorState:
    range_begin: torch.Tensor
    range_end: torch.Tensor
    ready_end: torch.Tensor
    indices: torch.Tensor
    ready_count: torch.Tensor
    consumed_end: torch.Tensor
    status: torch.Tensor
    initialized: torch.cuda.Event
    _launched: bool = False

    def tensors(self):
        return (
            self.range_begin, self.range_end, self.ready_end, self.indices,
            self.ready_count, self.consumed_end, self.status,
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
) -> CollectorState:
    """Allocate fresh state; producers must wait for state.initialized.

    By default each expert can hold every received row. Buffers must not alias
    each other or producer payloads. One state belongs to exactly one dispatch.
    """
    if capacity is None:
        capacity = num_rows
    for name, value, minimum in (
        ("num_ranges", num_ranges, 1), ("num_rows", num_rows, 0), ("capacity", capacity, 0)
    ):
        if not isinstance(value, int) or not minimum <= value <= 2**31 - 1:
            raise ValueError(f"{name} must be an integer in [{minimum}, INT_MAX]")
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise ValueError("state must be allocated on a CUDA device")
    with torch.cuda.device(device):
        options = dict(dtype=torch.int32, device=device)
        state = CollectorState(
            range_begin=torch.empty(num_ranges, **options),
            range_end=torch.empty(num_ranges, **options),
            ready_end=torch.full((num_ranges,), -1, **options),
            indices=torch.empty((8, capacity), **options),
            ready_count=torch.zeros(8, **options),
            consumed_end=torch.full((num_ranges,), -1, **options),
            status=torch.zeros(1, **options),
            initialized=torch.cuda.Event(),
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
    extension = build()
    device = state.ready_end.device
    with torch.cuda.device(device):
        if stream is None:
            stream = torch.cuda.Stream(device=device)
        if stream.device != device:
            raise ValueError("collector stream and state must be on the same GPU")
        with torch.cuda.stream(stream):
            stream.wait_event(state.initialized)
            extension.launch(recv_topk_idx, *state.tensors(), timeout_ms)
            state._launched = True
            for tensor in (recv_topk_idx, *state.tensors()):
                tensor.record_stream(stream)
            event = torch.cuda.Event()
            event.record()
    return CollectorRun(state, event, stream, recv_topk_idx)
