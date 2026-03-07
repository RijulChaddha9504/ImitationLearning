#!/bin/bash

echo "Starting Isaac GR00T Imitation Learning Training..."

CONFIG_PATH="$(pwd)/configuration/gr00t_train_config.yaml"
echo "Using configuration: $CONFIG_PATH"

cd /workspace/Isaac-GR00T

source .venv/bin/activate
source $HOME/.local/bin/env

# Disable DeepSpeed CUDA op compilation
export DS_BUILD_OPS=0
export DEEPSPEED_SKIP_CUDA_CHECK=1

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

uv run python gr00t/experiment/launch_finetune.py \
    --experiment-dir "/workspace/isaaclab/ImitationLearning/checkpoints" \
    --dataset-path "/workspace/isaaclab/ImitationLearning/demonstrations/robomimic_dataset.hdf5" \
    --dataset-val-path "/workspace/isaaclab/ImitationLearning/demonstrations/robomimic_dataset.hdf5" \
    --model-type "transformer_policy" \
    --training-batch-size 64 \
    --training-learning-rate 1e-4 \
    --training-num-epochs 200

echo "Training command finished."