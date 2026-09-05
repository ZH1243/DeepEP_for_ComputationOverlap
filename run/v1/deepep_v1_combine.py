#!/usr/bin/env python3
"""Direct DeepEP V1 combine benchmark with fake dispatch/expert inputs."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import torch
import torch.distributed as dist

from routing import (
    add_routing_arguments,
    make_fake_routing,
    routing_mode_summary,
    validate_routing_args,
)


_DEEPEP_BUFFER = None


def load_deepep():
    try:
        import deep_ep
        from deep_ep import Buffer
    except ImportError as exc:
        raise ImportError(
            "DeepEP is not importable. Install the NVSHMEM-based DeepEP V1 "
            "package before running this script."
        ) from exc
    return deep_ep, Buffer


def hidden_bytes(x: torch.Tensor) -> int:
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: dist.ProcessGroup, hidden_bytes_per_token: int):
    global _DEEPEP_BUFFER
    _, Buffer = load_deepep()
    group_size = dist.get_world_size(group)
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group_size), Buffer.get_combine_config(group_size)):
        num_nvl_bytes = max(
            num_nvl_bytes,
            config.get_nvl_buffer_size_hint(hidden_bytes_per_token, group_size),
        )
        num_rdma_bytes = max(
            num_rdma_bytes,
            config.get_rdma_buffer_size_hint(hidden_bytes_per_token, group_size),
        )
    if (
        _DEEPEP_BUFFER is None
        or _DEEPEP_BUFFER.group != group
        or _DEEPEP_BUFFER.num_nvl_bytes < num_nvl_bytes
        or _DEEPEP_BUFFER.num_rdma_bytes < num_rdma_bytes
    ):
        _DEEPEP_BUFFER = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _DEEPEP_BUFFER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DeepEP V1 combine directly. The script performs one fake dispatch "
            "to get the DeepEP handle, then benchmarks combine only."
        )
    )
    parser.add_argument("--num-local-tokens", type=int, required=True)
    parser.add_argument("--token-hidden", type=int, required=True)
    parser.add_argument("--num-of-experts", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument("--deepep-num-sms", type=int, default=24)
    parser.add_argument(
        "--fake-expert-output",
        choices=("identity", "random"),
        default="identity",
        help="identity reuses dispatched tokens; random uses random tensors with the dispatch shape.",
    )
    parser.add_argument("--print-timing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-correctness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--async-finish", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allocate-on-comm-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    add_routing_arguments(parser)
    return parser.parse_args()


def dtype_from_arg(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(dtype)


def validate_args(args: argparse.Namespace, world_size: int) -> None:
    checks = [
        (args.num_local_tokens > 0, "--num-local-tokens must be positive."),
        (args.token_hidden > 0, "--token-hidden must be positive."),
        (args.num_of_experts > 0, "--num-of-experts must be positive."),
        (args.topk > 0, "--topk must be positive."),
        (args.ep > 1, "DeepEP V1 combine requires EP > 1."),
        (world_size % args.ep == 0, "--ep must divide torchrun WORLD_SIZE."),
        (
            args.num_of_experts % args.ep == 0,
            "--num-of-experts must be divisible by --ep for TP=1 expert partitioning.",
        ),
        (args.topk <= args.num_of_experts, "--topk cannot exceed --num-of-experts."),
        (args.warmup_iters >= 0, "--warmup-iters cannot be negative."),
        (args.benchmark_iters >= 1, "--benchmark-iters must be at least 1."),
        (
            args.async_finish or not args.allocate_on_comm_stream,
            "--allocate-on-comm-stream requires --async-finish.",
        ),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(message)
    validate_routing_args(args)


def init_distributed() -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepEP.")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("LOCAL_RANK is not set. Run this script with torchrun.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), local_rank


def create_ep_group(ep_size: int) -> tuple[dist.ProcessGroup, int, int]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    group_count = world_size // ep_size
    ep_group_id = rank // ep_size
    ep_rank = rank % ep_size

    if ep_size == world_size:
        return dist.group.WORLD, ep_group_id, ep_rank

    ep_group = None
    for group_id in range(group_count):
        ranks = list(range(group_id * ep_size, (group_id + 1) * ep_size))
        group = dist.new_group(ranks=ranks, backend="nccl")
        if group_id == ep_group_id:
            ep_group = group
    if ep_group is None:
        raise RuntimeError("Failed to create EP process group.")
    return ep_group, ep_group_id, ep_rank


def wait_if_async(event: Any, async_finish: bool) -> None:
    if async_finish and event is not None:
        event.current_stream_wait()


def make_initial_event(async_finish: bool):
    if not async_finish:
        return None
    try:
        from deep_ep.utils import EventHandle, EventOverlap
    except ImportError:
        from deep_ep import EventHandle, EventOverlap
    return EventOverlap(EventHandle())


@torch.no_grad()
def setup_dispatch(
    args: argparse.Namespace,
    buffer,
    input_tokens: torch.Tensor,
    token_indices: torch.Tensor,
    token_probs: torch.Tensor,
) -> tuple[torch.Tensor, Any, torch.Tensor]:
    previous_event = make_initial_event(args.async_finish)
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        layout_event,
    ) = buffer.get_dispatch_layout(
        token_indices,
        args.num_of_experts,
        previous_event=previous_event,
        async_finish=args.async_finish,
        allocate_on_comm_stream=args.allocate_on_comm_stream,
    )
    recv_x, _recv_idx, _recv_probs, _tokens_per_expert, handle, dispatch_event = buffer.dispatch(
        input_tokens,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=layout_event,
        async_finish=args.async_finish,
        allocate_on_comm_stream=args.allocate_on_comm_stream,
    )
    wait_if_async(dispatch_event, args.async_finish)
    torch.cuda.synchronize()
    return recv_x, handle, is_token_in_rank


@torch.no_grad()
def run_combine(
    args: argparse.Namespace,
    buffer,
    fake_expert_output: torch.Tensor,
    handle,
) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    previous_event = make_initial_event(args.async_finish)

    start.record()
    combined_x, _combined_topk_weights, combine_event = buffer.combine(
        fake_expert_output,
        handle=handle,
        previous_event=previous_event,
        async_finish=args.async_finish,
        allocate_on_comm_stream=args.allocate_on_comm_stream,
    )
    wait_if_async(combine_event, args.async_finish)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), combined_x


def ordered_print(rank: int, world_size: int, message: str) -> None:
    for current in range(world_size):
        dist.barrier()
        if rank == current:
            print(message, flush=True)
    dist.barrier()


def all_ranks_boolean(ok: bool, device: torch.device) -> bool:
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def main() -> int:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    validate_args(args, world_size)
    ep_group, ep_group_id, ep_rank = create_ep_group(args.ep)
    deep_ep, Buffer = load_deepep()
    Buffer.set_num_sms(args.deepep_num_sms)

    dtype = dtype_from_arg(args.dtype)
    device = torch.device("cuda", local_rank)
    topk_dtype = getattr(deep_ep, "topk_idx_t", torch.int64)
    num_local_experts = args.num_of_experts // args.ep

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    input_tokens = torch.randn((args.num_local_tokens, args.token_hidden), device=device, dtype=dtype)
    token_indices, token_probs = make_fake_routing(
        args.num_local_tokens,
        args.num_of_experts,
        args.topk,
        device,
        topk_dtype,
        ep_size=args.ep,
        ep_rank=ep_rank,
        uniform=args.uniform_routing,
        exclude_local_node=args.exclude_local_node_routing,
        routing_ranks_per_node=args.routing_ranks_per_node,
    )
    buffer = get_buffer(ep_group, hidden_bytes(input_tokens))
    dispatched_tokens, handle, is_token_in_rank = setup_dispatch(
        args,
        buffer,
        input_tokens,
        token_indices,
        token_probs,
    )
    if args.fake_expert_output == "identity":
        fake_expert_output = dispatched_tokens.contiguous()
    else:
        fake_expert_output = torch.randn_like(dispatched_tokens).contiguous()

    ordered_print(
        rank,
        world_size,
        (
            f"[rank {rank}/{world_size}] local_rank={local_rank} ep_group={ep_group_id} "
            f"ep_rank={ep_rank}/{args.ep} mode={'inter-node' if args.ep > 8 else 'intra-node'} "
            f"tokens={args.num_local_tokens} hidden={args.token_hidden} "
            f"experts={args.num_of_experts} local_experts={num_local_experts} "
            f"topk={args.topk} dtype={args.dtype} fake_expert_output={args.fake_expert_output} "
            f"routing={routing_mode_summary(args)}"
        ),
    )

    dist.barrier(group=ep_group)
    for _ in range(args.warmup_iters):
        run_combine(args, buffer, fake_expert_output, handle)
    dist.barrier(group=ep_group)

    timings = []
    combined = None
    for _ in range(args.benchmark_iters):
        ms, combined = run_combine(args, buffer, fake_expert_output, handle)
        timings.append(ms)
    dist.barrier(group=ep_group)

    local_ok = True
    if args.check_correctness:
        local_ok = (
            combined is not None
            and tuple(combined.shape) == tuple(input_tokens.shape)
            and torch.isfinite(combined).all().item()
        )
        if local_ok and args.fake_expert_output == "identity":
            divisor = is_token_in_rank.sum(dim=1).clamp_min(1).unsqueeze(1)
            local_ok = torch.allclose(
                combined.float() / divisor,
                input_tokens.float(),
                atol=5.0e-3 if dtype != torch.float32 else 1.0e-5,
                rtol=5.0e-3 if dtype != torch.float32 else 1.0e-5,
            )
    global_ok = all_ranks_boolean(local_ok, device)

    assert combined is not None
    ordered_print(
        rank,
        world_size,
        (
            f"[rank {rank}] shapes: dispatched={tuple(dispatched_tokens.shape)} "
            f"fake_expert_output={tuple(fake_expert_output.shape)} "
            f"combined={tuple(combined.shape)} sanity_ok={local_ok}"
        ),
    )
    if args.print_timing:
        avg_ms = sum(timings) / len(timings)
        ordered_print(
            rank,
            world_size,
            f"[rank {rank}] avg combine over {args.benchmark_iters} iters: {avg_ms:.3f} ms",
        )

    dist.barrier()
    if rank == 0:
        print(
            "All distributed ranks completed successfully."
            if global_ok
            else "At least one distributed rank failed sanity checks.",
            flush=True,
        )
    dist.destroy_process_group()
    return 0 if global_ok else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if dist.is_available() and dist.is_initialized():
            rank_text = f"rank {dist.get_rank()}"
        else:
            rank_text = "uninitialized rank"
        print(f"[{rank_text}] ERROR: {exc}", file=sys.stderr, flush=True)
        time.sleep(0.2)
        raise
