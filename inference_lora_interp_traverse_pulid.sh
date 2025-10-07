#!/bin/bash

SEED=8165
METHOD="direct_linear"
YOUNG_LORA="path/to/young_lora"
OLD_LORA="path/to/old_lora"
IMAGE_PATH="./example_inputs/hinton.jpeg"
SEX="m"

ID_SCALE=0.3
FOLDER="outputs/$METHOD/idscale_${ID_SCALE}"
MODEL_PATH="RunDiffusion/Juggrnaut-XL-v9"

echo "Running with seed=$SEED -> Output: $FOLDER"
mkdir -p "$FOLDER"

python inference_lora_interp_traverse_pulid.py \
    --pretrained_model_name_or_path=${MODEL_PATH} \
    --young_lora_path=${YOUNG_LORA} \
    --old_lora_path=${OLD_LORA} \
    --image_path=${IMAGE_PATH} \
    --sex=${SEX} \
    --output_folder="$FOLDER" \
    --id_scale=$ID_SCALE \
    --alpha_step=0.1 \
    --num_samples=1 \
    --method="$METHOD" \
    --seed=$SEED \
    --prompt="" \