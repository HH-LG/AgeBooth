export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export OUTPUT_DIR="models/ageLoRA/flux/80_90_b1"
export INSTANCE_DIR="data/80-90"


BATCH_SIZE=1
export PROMPT="A person in sbu age."
export VALID_PROMPT="A person at the age of sbu, portrait, in white background"
accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="${PROMPT}" \
  --rank=4 \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=${BATCH_SIZE} \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=40 \
  --checkpointing_steps=100 \
  --gradient_checkpointing \
  --seed="0" \
  #--push_to_hub