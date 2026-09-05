"""Fake router metadata helpers for DeepEP V1 tests."""

from __future__ import annotations

import argparse

import torch


def add_routing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--uniform-routing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Route token i to consecutive experts, wrapping around the eligible "
            "expert set, so assignments are balanced across experts."
        ),
    )
    parser.add_argument(
        "--exclude-local-node-routing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Never route a token to experts owned by ranks in the same routing "
            "node as the source rank."
        ),
    )
    parser.add_argument(
        "--routing-ranks-per-node",
        type=int,
        default=8,
        help="Ranks per node used by --exclude-local-node-routing.",
    )


def validate_routing_args(args: argparse.Namespace) -> None:
    if args.routing_ranks_per_node <= 0:
        raise ValueError("--routing-ranks-per-node must be positive.")
    if not args.exclude_local_node_routing:
        return

    num_local_experts = args.num_of_experts // args.ep
    max_same_node_ranks = min(args.routing_ranks_per_node, args.ep)
    min_eligible_experts = args.num_of_experts - max_same_node_ranks * num_local_experts
    if min_eligible_experts <= 0:
        raise ValueError(
            "--exclude-local-node-routing leaves no eligible experts. Increase --ep "
            "or reduce --routing-ranks-per-node."
        )
    if args.topk > min_eligible_experts:
        raise ValueError(
            "--topk cannot exceed the minimum number of experts outside the local "
            f"routing node ({min_eligible_experts})."
        )


def routing_mode_summary(args: argparse.Namespace) -> str:
    modes = []
    if args.uniform_routing:
        modes.append("uniform")
    else:
        modes.append("random")
    if args.exclude_local_node_routing:
        modes.append(f"exclude-local-node/ranks-per-node={args.routing_ranks_per_node}")
    return "+".join(modes)


def _eligible_experts(
    num_experts: int,
    ep_size: int,
    ep_rank: int,
    routing_ranks_per_node: int,
    exclude_local_node: bool,
    device: torch.device,
) -> torch.Tensor:
    eligible = torch.arange(num_experts, device=device, dtype=torch.long)
    if not exclude_local_node:
        return eligible

    num_local_experts = num_experts // ep_size
    local_node_start = (ep_rank // routing_ranks_per_node) * routing_ranks_per_node
    local_node_end = min(local_node_start + routing_ranks_per_node, ep_size)
    expert_owner = eligible // num_local_experts
    return eligible[(expert_owner < local_node_start) | (expert_owner >= local_node_end)]


def make_fake_routing(
    num_tokens: int,
    num_experts: int,
    topk: int,
    device: torch.device,
    topk_dtype: torch.dtype,
    *,
    ep_size: int,
    ep_rank: int,
    uniform: bool,
    exclude_local_node: bool,
    routing_ranks_per_node: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    eligible = _eligible_experts(
        num_experts,
        ep_size,
        ep_rank,
        routing_ranks_per_node,
        exclude_local_node,
        device,
    )
    if topk > eligible.numel():
        raise ValueError(
            f"topk={topk} cannot exceed eligible experts={eligible.numel()} "
            f"for ep_rank={ep_rank}."
        )

    if uniform:
        token_offsets = torch.arange(num_tokens, device=device, dtype=torch.long).unsqueeze(1)
        topk_offsets = torch.arange(topk, device=device, dtype=torch.long).unsqueeze(0)
        token_indices = eligible[(token_offsets + topk_offsets) % eligible.numel()].to(topk_dtype)
        token_probs = torch.full(
            (num_tokens, topk),
            1.0 / topk,
            device=device,
            dtype=torch.float32,
        )
        return token_indices, token_probs

    scores = torch.rand((num_tokens, eligible.numel()), device=device, dtype=torch.float32)
    selected = torch.topk(scores, k=topk, dim=-1, sorted=False).indices
    token_indices = eligible[selected].to(topk_dtype)
    raw_probs = torch.rand((num_tokens, topk), device=device, dtype=torch.float32)
    token_probs = raw_probs / raw_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
    return token_indices, token_probs
