#!/bin/bash
# Script to launch Isaac GR00T training from the WatCloud Docker container

echo "Starting Isaac GR00T Imitation Learning Training..."

# The config file needs an absolute path within the container. 
# We assume this script is running from the ImitationLearning directory.
CONFIG_PATH="$(pwd)/configuration/gr00t_train_config.yaml"

echo "Using configuration: $CONFIG_PATH"

# Navigate to the Isaac-GR00T installation directory
cd ~/Isaac-GR00T || { echo "Error: ~/Isaac-GR00T not found. Did you run the installation steps?"; exit 1; }

# Activate the uv environment
source .venv/bin/activate
source $HOME/.local/bin/env

# Set CUDA_HOME so DeepSpeed can find the newly installed CUDA compiler
export CUDA_HOME=/usr

# Run the training script provided by Isaac GR00T (using launch_finetune.py)
uv run python gr00t/experiment/launch_finetune.py \
    --experiment_dir "/workspace/isaaclab/ImitationLearning/checkpoints" \
    --dataset.path "/workspace/isaaclab/ImitationLearning/demonstrations/robomimic_dataset.hdf5" \
    --dataset.val_path "/workspace/isaaclab/ImitationLearning/demonstrations/robomimic_dataset.hdf5" \
    --model.type "transformer_policy" \
    --training.batch_size 64 \
    --training.learning_rate 1e-4 \
    --training.num_epochs 200

echo "Training command finished."
