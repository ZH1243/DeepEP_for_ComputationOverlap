# Direct DeepEP V1 Dispatch/Combine Tests

This directory contains two standalone tests that call the DeepEP V1 `Buffer`
API directly. They do not import Megatron-LM, do not run a router, and do not
run expert computation. Instead, they fake local token tensors and top-k router
decisions so the communication path can be profiled in isolation.

## Tests

- `deepep_v1_dispatch.py`: benchmarks DeepEP V1 dispatch only.
- `deepep_v1_combine.py`: runs one setup dispatch to obtain the DeepEP handle,
  then benchmarks DeepEP V1 combine only.

The scripts support both cases:

- Intra-node EP: `--ep <= 8`
- Inter-node EP: `--ep > 8`

DeepEP chooses the corresponding communication path through its V1 buffer
configuration and runtime. Expert tensor parallelism is assumed to be 1.

## Single Node

Dispatch:

```bash
torchrun \
  --nproc_per_node=8 \
  DeepEP_v1/deepep_v1_dispatch.py \
  --num-local-tokens 4096 \
  --token-hidden 7168 \
  --num-of-experts 256 \
  --topk 8 \
  --ep 8
```

Combine:

```bash
torchrun \
  --nproc_per_node=8 \
  DeepEP_v1/deepep_v1_combine.py \
  --num-local-tokens 4096 \
  --token-hidden 7168 \
  --num-of-experts 256 \
  --topk 8 \
  --ep 8
```

## Multiple Nodes

Example for 2 nodes, 8 GPUs per node, and one EP group of 16 ranks:

```bash
torchrun \
  --nnodes=2 \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=8 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  DeepEP_v1/deepep_v1_dispatch.py \
  --num-local-tokens 4096 \
  --token-hidden 7168 \
  --num-of-experts 256 \
  --topk 8 \
  --ep 16
```

Use the same launch shape for `deepep_v1_combine.py`.

If `--ep` is smaller than `WORLD_SIZE`, the scripts create independent
contiguous EP groups. For example, with `WORLD_SIZE=32` and `--ep 16`, ranks
`0..15` form one EP group and ranks `16..31` form another.

## What The Inputs Mean

- `--num-local-tokens`: number of local tokens created on each rank.
- `--token-hidden`: hidden size per token. A value of `7168` means each token is
  a vector with 7168 elements.
- `--num-of-experts`: total experts in each EP group.
- `--topk`: number of selected experts per token.
- `--ep`: expert-parallel group size.
- `--dtype`: token dtype, default `bf16`.

For default `bf16`, each hidden element is 2 bytes. With
`--token-hidden 7168`, each token is `7168 * 2 = 14336` bytes.

## What The Tests Do

`deepep_v1_dispatch.py`:

1. Initializes NCCL distributed state.
2. Creates a contiguous EP process group.
3. Creates fake local token hidden states with shape
   `[num_local_tokens, token_hidden]`.
4. Creates fake top-k router metadata:
   - `token_indices`: `[num_local_tokens, topk]`
   - `token_probs`: `[num_local_tokens, topk]`
   By default, the fake router randomly selects experts.
5. Allocates a DeepEP V1 `Buffer`.
6. Calls `buffer.get_dispatch_layout(...)`.
7. Calls `buffer.dispatch(...)`.
8. Prints output shapes, tokens per local expert, sanity status, and average
   dispatch time.

`deepep_v1_combine.py`:

1. Initializes NCCL distributed state.
2. Creates fake local tokens and router metadata.
3. Runs one setup dispatch to obtain the DeepEP combine handle.
4. Fakes expert output:
   - `identity`: reuse the dispatched tokens.
   - `random`: create random tensors with the dispatched shape.
5. Calls `buffer.combine(...)`.
6. Prints output shapes, sanity status, and average combine time.

## Useful Arguments

- `--warmup-iters`: warmup iterations before timing.
- `--benchmark-iters`: timed iterations.
- `--deepep-num-sms`: value passed to `Buffer.set_num_sms`.
- `--async-finish / --no-async-finish`: enable or disable DeepEP async finish.
- `--allocate-on-comm-stream / --no-allocate-on-comm-stream`: enable or disable
  communication-stream allocation.
- `--check-correctness / --no-check-correctness`: enable or disable lightweight
  sanity checks.
- `--rerandomize-routing-each-iter`: dispatch test only; regenerate fake router
  metadata every iteration.
- `--fake-expert-output {identity,random}`: combine test only.
- `--uniform-routing / --no-uniform-routing`: route local token `i` to
  consecutive eligible expert IDs. Expert IDs are zero-based in the tensors, so
  with `--topk 2`, token 0 routes to experts `[0, 1]`, token 1 to `[1, 2]`,
  and so on, wrapping at the end.
- `--exclude-local-node-routing / --no-exclude-local-node-routing`: exclude
  experts owned by ranks in the same routing node as the source token. When this
  is combined with `--uniform-routing`, tokens are uniformly distributed only
  across experts on other routing nodes.
- `--routing-ranks-per-node`: number of ranks treated as one routing node for
  `--exclude-local-node-routing`, default `8`.

`--allocate-on-comm-stream` requires `--async-finish`, matching the DeepEP V1
event dependency requirements.

## Requirements And Assumptions

- Run with `torchrun`.
- CUDA and NCCL are required.
- The `deep_ep` Python package must be importable.
- Inter-node EP requires a DeepEP V1 build and environment with working
  NVSHMEM/RDMA support.
- `--ep` must divide `WORLD_SIZE`.
- `--num-of-experts` must divide `--ep`.
- `--topk` must be less than or equal to `--num-of-experts`.
- With `--exclude-local-node-routing`, `--topk` must fit in the minimum number
  of experts outside any local routing node.
- Expert TP is fixed to 1.
- Expert computation is intentionally skipped.
