#!/usr/bin/env python3
"""DeepEP V1 internode dispatch with optional live ready-token collection."""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import torch.distributed as dist

import deepep_v1_dispatch as base
from recv_x_ready_tokens_idx_collector_kernel import allocate_state, build, launch
from routing import make_fake_routing, routing_mode_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepEP V1 internode dispatch with an optional concurrent ready-token collector."
    )
    base.add_dispatch_arguments(parser)
    parser.add_argument(
        "--with-collector",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="start the collector before dispatch and enable receiver-frontier publication",
    )
    parser.add_argument(
        "--collector-timeout-ms",
        type=float,
        default=10000.0,
        help="collector idle timeout; zero disables it",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, world_size: int) -> None:
    base.validate_args(args, world_size)
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    node_count = int(os.environ.get("GROUP_WORLD_SIZE", 0))
    if node_count <= 0:
        node_count = world_size // local_world_size if local_world_size > 0 else 0
    if node_count <= 1 or world_size <= local_world_size:
        raise ValueError("This test requires torchrun with --nnodes greater than 1.")
    if args.ep <= 8 or args.ep % 8 != 0:
        raise ValueError("This test requires an internode EP group whose --ep is a multiple of 8 and greater than 8.")
    if args.num_of_experts // args.ep != 8:
        raise ValueError("The ready-token collector currently requires exactly 8 local experts per GPU.")
    if args.num_local_tokens * args.ep > 2**31 - 1:
        raise ValueError("The worst-case received-row capacity must fit in int32.")
    if args.deepep_num_sms <= 0 or args.deepep_num_sms % 2 != 0:
        raise ValueError("--deepep-num-sms must be a positive even number for internode dispatch.")
    if not 0.0 <= args.collector_timeout_ms <= 3600000.0:
        raise ValueError("--collector-timeout-ms must be in [0, 3600000].")
    if args.with_collector:
        sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        if args.deepep_num_sms >= sm_count:
            raise ValueError(
                "The collector needs one concurrently schedulable CTA; set --deepep-num-sms below "
                f"the GPU SM count ({sm_count})."
            )


def check_collector_output(
    recv_topk_idx: torch.Tensor,
    state,
) -> tuple[bool, list[int]]:
    descriptor_ok = bool(
        torch.all(state.range_begin >= 0).item()
        and torch.all(state.range_begin <= state.range_end).item()
        and torch.equal(state.ready_end, state.range_end)
        and int(state.range_end.max().item()) <= recv_topk_idx.size(0)
        and int((state.range_end - state.range_begin).sum().item()) == recv_topk_idx.size(0)
    )
    counts = state.ready_count.cpu().to(torch.int64).tolist()
    indices_ok = True
    for expert in range(8):
        expected = torch.nonzero((recv_topk_idx == expert).any(dim=1)).flatten().cpu().tolist()
        actual = state.indices[expert, :counts[expert]].cpu().tolist()
        indices_ok &= sorted(actual) == expected
    return descriptor_ok and indices_ok, counts


@torch.no_grad()
def run_dispatch(
    args: argparse.Namespace,
    buffer,
    input_tokens: torch.Tensor,
    token_indices: torch.Tensor,
    token_probs: torch.Tensor,
) -> tuple[float, dict[str, Any]]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    previous_event = base.make_initial_event(args.async_finish)

    start.record()
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

    collector_run = None
    collector_state = None
    recv_topk_idx_buffer = None
    dispatch_previous_event = layout_event
    if args.with_collector:
        # Join layout into the current stream. Keep layout_event as DeepEP's
        # previous event: allocate_on_comm_stream requires one, and the C++
        # publication path also waits on this compute stream so readiness-state
        # initialization precedes the producer without waiting for the collector.
        base.wait_if_async(layout_event, args.async_finish)
        max_recv_rows = args.num_local_tokens * args.ep
        num_rdma_ranks = args.ep // 8
        num_ranges = (args.deepep_num_sms // 2) * 8 * num_rdma_ranks
        recv_topk_idx_buffer = torch.empty(
            (max_recv_rows, args.topk), dtype=token_indices.dtype, device=token_indices.device
        )
        collector_state = allocate_state(
            num_ranges=num_ranges,
            num_rows=max_recv_rows,
            device=token_indices.device,
        )
        collector_run = launch(
            recv_topk_idx_buffer,
            collector_state,
            timeout_ms=args.collector_timeout_ms,
        )

    (
        recv_x,
        recv_token_indices,
        recv_token_probs,
        num_recv_tokens_per_expert,
        _handle,
        dispatch_event,
    ) = buffer.dispatch(
        input_tokens,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=dispatch_previous_event,
        async_finish=args.async_finish,
        allocate_on_comm_stream=args.allocate_on_comm_stream,
        publish_ready_tokens=args.with_collector,
        ready_token_state=(
            collector_state.range_begin,
            collector_state.range_end,
            collector_state.ready_end,
        ) if collector_state is not None else None,
        recv_topk_idx_buffer=recv_topk_idx_buffer,
    )
    base.wait_if_async(dispatch_event, args.async_finish)
    end.record()
    torch.cuda.synchronize()

    collector_ok = True
    collector_counts = None
    if collector_run is not None:
        collector_run.wait()
        collector_ok, collector_counts = check_collector_output(recv_token_indices, collector_state)

    if torch.is_tensor(num_recv_tokens_per_expert):
        tokens_per_expert = num_recv_tokens_per_expert.detach().cpu().to(torch.int64).tolist()
    else:
        tokens_per_expert = [int(x) for x in num_recv_tokens_per_expert]
    meta = {
        "input_shape": tuple(input_tokens.shape),
        "topk_indices_shape": tuple(token_indices.shape),
        "topk_probs_shape": tuple(token_probs.shape),
        "recv_x_shape": tuple(recv_x.shape),
        "recv_topk_indices_shape": tuple(recv_token_indices.shape),
        "recv_topk_probs_shape": tuple(recv_token_probs.shape),
        "tokens_per_local_expert": tokens_per_expert,
        "collector_ok": collector_ok,
        "collector_counts": collector_counts,
    }
    return float(start.elapsed_time(end)), meta


def main() -> int:
    args = parse_args()
    rank, world_size, local_rank = base.init_distributed()
    validate_args(args, world_size)
    ep_group, ep_group_id, ep_rank = base.create_ep_group(args.ep)
    deep_ep, Buffer = base.load_deepep()
    Buffer.set_num_sms(args.deepep_num_sms)

    if args.with_collector:
        # Compile before any producer can start, so JIT build time cannot trip
        # the persistent collector's watchdog.
        build()
    dist.barrier()

    dtype = base.dtype_from_arg(args.dtype)
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
    buffer = base.get_buffer(ep_group, base.hidden_bytes(input_tokens))

    base.ordered_print(
        rank,
        world_size,
        (
            f"[rank {rank}/{world_size}] local_rank={local_rank} ep_group={ep_group_id} "
            f"ep_rank={ep_rank}/{args.ep} mode=inter-node collector={args.with_collector} "
            f"tokens={args.num_local_tokens} hidden={args.token_hidden} "
            f"experts={args.num_of_experts} local_experts={num_local_experts} "
            f"topk={args.topk} dtype={args.dtype} routing={routing_mode_summary(args)}"
        ),
    )

    dist.barrier(group=ep_group)
    for _ in range(args.warmup_iters):
        if args.rerandomize_routing_each_iter:
            token_indices, token_probs = make_fake_routing(
                args.num_local_tokens, args.num_of_experts, args.topk, device, topk_dtype,
                ep_size=args.ep, ep_rank=ep_rank, uniform=args.uniform_routing,
                exclude_local_node=args.exclude_local_node_routing,
                routing_ranks_per_node=args.routing_ranks_per_node,
            )
        run_dispatch(args, buffer, input_tokens, token_indices, token_probs)
    dist.barrier(group=ep_group)

    timings = []
    last_meta = None
    for _ in range(args.benchmark_iters):
        if args.rerandomize_routing_each_iter:
            token_indices, token_probs = make_fake_routing(
                args.num_local_tokens, args.num_of_experts, args.topk, device, topk_dtype,
                ep_size=args.ep, ep_rank=ep_rank, uniform=args.uniform_routing,
                exclude_local_node=args.exclude_local_node_routing,
                routing_ranks_per_node=args.routing_ranks_per_node,
            )
        ms, last_meta = run_dispatch(args, buffer, input_tokens, token_indices, token_probs)
        timings.append(ms)
    dist.barrier(group=ep_group)

    assert last_meta is not None
    local_ok = bool(last_meta["collector_ok"])
    if args.check_correctness:
        local_ok = local_ok and (
            last_meta["input_shape"] == (args.num_local_tokens, args.token_hidden)
            and last_meta["topk_indices_shape"] == (args.num_local_tokens, args.topk)
            and last_meta["topk_probs_shape"] == (args.num_local_tokens, args.topk)
            and last_meta["recv_x_shape"][1] == args.token_hidden
            and len(last_meta["tokens_per_local_expert"]) == 8
            and token_indices.min().item() >= 0
            and token_indices.max().item() < args.num_of_experts
            and torch.allclose(
                token_probs.sum(dim=-1), torch.ones(args.num_local_tokens, device=device),
                atol=1.0e-6, rtol=1.0e-6,
            )
        )
    global_ok = base.all_ranks_boolean(local_ok, device)

    base.ordered_print(
        rank,
        world_size,
        (
            f"[rank {rank}] recv_x={last_meta['recv_x_shape']} "
            f"recv_topk_indices={last_meta['recv_topk_indices_shape']} "
            f"tokens_per_local_expert={last_meta['tokens_per_local_expert']} "
            f"collector_counts={last_meta['collector_counts']} sanity_ok={local_ok}"
        ),
    )
    if args.print_timing:
        avg_ms = sum(timings) / len(timings)
        base.ordered_print(
            rank, world_size,
            f"[rank {rank}] avg dispatch over {args.benchmark_iters} iters: {avg_ms:.3f} ms",
        )

    dist.barrier()
    if rank == 0:
        print(
            "All distributed ranks completed successfully."
            if global_ok else "At least one distributed rank failed sanity checks.",
            flush=True,
        )
    dist.destroy_process_group()
    return 0 if global_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
