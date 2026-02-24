#!/bin/bash
# Script to launch Isaac GR00T training

echo "Starting Isaac GR00T Imitation Learning Training..."

CONFIG_PATH="$(pwd)/configuration/gr00t_train_config.yaml"
echo "Using configuration: $CONFIG_PATH"

# Go to GR00T repo
cd ~/Isaac-GR00T || { echo "Error: ~/Isaac-GR00T not found."; exit 1; }

# Activate environment
source .venv/bin/activate
source $HOME/.local/bin/env

# Verify GPU before starting
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Launch training
uv run python gr00t/experiment/launch_finetune.py \
    --experiment-dir "/workspace/isaaclab/ImitationLearning/checkpoints" \
    --dataset-path "/workspace/isaaclab/ImitationLearning/demonstrations/robomimic_dataset.hdf5" \
    --dataset-val-path "/workspace/isaaclab/ImitationLearning/demonstrations/robomimic_dataset.hdf5" \
    --model-type "transformer_policy" \
    --training-batch-size 64 \
    --training-learning-rate 1e-4 \
    --training-num-epochs 200

echo "Training command finished."