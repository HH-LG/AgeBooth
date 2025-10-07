export MODEL_NAME="RunDiffusion/Juggrnaut-XL-v9"
export OUTPUT_DIR="models/ageLoRA/jgr/10_20_b1"
export INSTANCE_DIR="data/10-20"


BATCH_SIZE=1
export PROMPT="A person in sbu age."
export VALID_PROMPT="A person at the age of sbu, portrait, in white background"
accelerate launch \
  --machine_rank 0 \
  --num_machines 1 \
  --main_process_port 11131 \
  --num_processes 1 \
  train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --rank=16 \
  --resolution=512 \
  --train_batch_size=${BATCH_SIZE} \
  --learning_rate=5e-5 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=200 \
  --seed="0" \
  --gradient_checkpointing \
  --variant="fp16" \
  --mixed_precision="fp16" \
  --checkpointing_steps=200 \
  # --use_8bit_adam \
  # -enable_xformers_memory_efficient_attention \
  # -push_to_hub \