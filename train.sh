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

# Run the training script provided by Isaac GR00T
# Note: For multi-GPU, Isaac-GR00T usually uses torchrun. For a single GPU or debug, python works but torchrun is safer if available.
python scripts/train.py --config-name="$CONFIG_PATH"

echo "Training command finished."
