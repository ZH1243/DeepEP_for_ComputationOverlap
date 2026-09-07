NUM_NODES=2
MASTER_ADDR=28.49.38.169
MASTER_PORT=9080

NODE_RANK=0


export NCCL_IB_HCA="mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_8,mlx5_8,mlx5_9"
export NVSHMEM_HCA_LIST="mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_8:1"

torchrun \
  --nnodes=${NUM_NODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=8 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  deepep_v1_dispatch.py \
  --num-local-tokens 4096 \
  --token-hidden 7168 \
  --num-of-experts 128 \
  --topk 8 \
  --ep 16 \
  --deepep-num-sms 24 \
  --exclude-local-node-routing \
  --routing-ranks-per-node 8