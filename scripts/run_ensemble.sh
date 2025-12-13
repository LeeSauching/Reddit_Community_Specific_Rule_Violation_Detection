#!/bin/bash
# scripts/run_ensemble.sh

# GPU 0 跑这两个
(
  export CUDA_VISIBLE_DEVICES=0
  python train.py --config configs/phi4_14b.yaml
  python train.py --config configs/qwen3_8b.yaml
) &

# GPU 1 跑这两个
(
  export CUDA_VISIBLE_DEVICES=1
  python train.py --config configs/qwen3_14b.yaml
  python train.py --config configs/qwen25_7b.yaml
) &

wait
python ensemble.py # 运行融合脚本
