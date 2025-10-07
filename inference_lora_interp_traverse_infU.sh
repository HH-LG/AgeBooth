#!/bin/bash

SEED=39582
METHOD="svdmix"
# MODEL_VERSION="sim_stage1"
MODEL_VERSION="aes_stage2"
YOUNG_LORA="path/to/young_lora"
OLD_LORA="path/to/old_lora"
LORA="_b1/checkpoint-1000"
IMAGE_PATH="./example_inputs/hinton.jpeg"
SEX="m"

ID_SCALE=0.6
FOLDER="outputs/$METHOD/idscale_${ID_SCALE}"


echo "Running with seed=$SEED -> Output: $FOLDER"
mkdir -p "$FOLDER"
ENABLE_REALISM_LORA=1
MODEL_STR=${MODEL_VERSION}
EXTRA_ARGS=""
if [ $ENABLE_REALISM_LORA -eq 1 ]; then
    # model_version + realism lora
    MODEL_STR="${MODEL_VERSION}_realism"
    EXTRA_ARGS="--enable_realism_lora"
fi
FLUX_PATH="./models/FLUX.1-dev"
InfU_PATH="./models/InfiniteYou"

python inference_lora_interp_traverse_infU.py \
    --base_model_path ${FLUX_PATH} \
    --model_dir ${InfU_PATH} \
    --lora_name_or_path="/home/share/fluxage/models/ageLoRA/flux/5_15${LORA}" \
    --lora_name_or_path_2="/home/share/fluxage/models/ageLoRA/flux/80_90${LORA}" \
    --image_path=${IMAGE_PATH} \
    --sex=${SEX} \
    --output_folder="$FOLDER" \
    --model_version="$MODEL_VERSION" \
    --id_scale=$ID_SCALE \
    --alpha_step=0.1 \
    --num_samples=1 \
    --method="$METHOD" \
    --prompt="" \
    --seed=$SEED \
    debug \
    $EXTRA_ARGS \