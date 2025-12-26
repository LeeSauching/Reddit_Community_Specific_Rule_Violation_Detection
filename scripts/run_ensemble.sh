#!/bin/bash
# scripts/run_ensemble.sh

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,moheader | wc -l)
echo "Detected $GPU_COUNT GPUs."

run_group_0(){
    echo "Starting Group 0 on GPU $1..."
    export CUDA_VISIBLE_DEVICES=$1
    python train.py --config configs/phi4.yaml
    python train.py --config configs/qwen25_7b.yaml

}

run_group_1() {
    echo "Starting Group 1 on GPU $1..."
    export CUDA_VISIBLE_DEVICES=$1
    python train.py --config configs/qwen3_14b.yaml
    python train.py --config configs/qwen3_8b.yaml
}

if [ "$GPU_COUNT" ge 2 ]; then
    echo "Multi-GPU Environment"
    # GPU 0 跑第一组任务 (后台运行 &)
    (run_group_0 0) &
    # GPU 1 跑第二组任务 (后台运行 &)
    (run_group_1 1) &
    wait
else
    echo "Single-GPU Environment."
    run_group_0 0
    run_group_1 0
fi

echo "All training finished. Starting Ensemble..."
python ensemble.py
