export EXPERIMENT_NAME="FSMG"
export MODEL_PATH="./stable-diffusion/stable-diffusion-2-1-base"
export CLASS_DIR="data/class-person"
export CLEAN_TRAIN_DIR="$data_path/$dataset_name/$data_id/set_A" 
export REF_MODEL_PATH="outputs/$EXPERIMENT_NAME/tmp_ADVERSARIAL"

target_dir=$(ls $REF_MODEL_PATH -1 | grep -E 'checkpoint-[0-9]+-id[0-9]+')
# 检查是否找到符合命名规则的文件夹
if [[ -n "$target_dir" ]]; then
    # 提取文件夹中的 data_id
    extracted_data_id=$(echo "$target_dir" | grep -oP '(?<=-id)\d+')
    # 判断提取的 data_id 是否与当前环境变量 data_id 不相等
    echo "now target id $data_id ,we have ckp of id $extracted_data_id"
    if [[ "$extracted_data_id" != "$data_id" ]]; then
        echo "删除目录: $target_dir"
        rm -rf "$REF_MODEL_PATH"  # 删除不匹配的目录
        # ------------------------- Train DreamBooth model on set A -------------------------
        accelerate launch train_dreambooth.py \
          --pretrained_model_name_or_path=$MODEL_PATH  \
          --enable_xformers_memory_efficient_attention \
          --train_text_encoder \
          --instance_data_dir=$CLEAN_TRAIN_DIR\
          --class_data_dir=$CLASS_DIR \
          --output_dir=$REF_MODEL_PATH \
          --with_prior_preservation \
          --prior_loss_weight=1.0 \
          --instance_prompt="a photo of sks person" \
          --class_prompt="a photo of person" \
          --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
          --resolution=512 \
          --train_batch_size=2 \
          --gradient_accumulation_steps=1 \
          --learning_rate=5e-7 \
          --lr_scheduler="constant" \
          --lr_warmup_steps=0 \
          --num_class_images=200 \
          --max_train_steps=$max_train_steps \
          --checkpointing_steps=$max_train_steps \
          --center_crop \
          --mixed_precision=bf16 \
          --prior_generation_precision=bf16 \
          --sample_batch_size=16
        echo "rename $REF_MODEL_PATH/checkpoint-${max_train_steps} to $REF_MODEL_PATH/checkpoint-${max_train_steps}-id${data_id}"
        mv $REF_MODEL_PATH/checkpoint-${max_train_steps} $REF_MODEL_PATH/checkpoint-${max_train_steps}-id${data_id}
    else
        echo "ckp匹配，无须再次生成: $target_dir"
    fi
else
    echo "未找到符合命名规则的文件夹。"
    accelerate launch train_dreambooth.py \
      --pretrained_model_name_or_path=$MODEL_PATH  \
      --enable_xformers_memory_efficient_attention \
      --train_text_encoder \
      --instance_data_dir=$CLEAN_TRAIN_DIR\
      --class_data_dir=$CLASS_DIR \
      --output_dir=$REF_MODEL_PATH \
      --with_prior_preservation \
      --prior_loss_weight=1.0 \
      --instance_prompt="a photo of sks person" \
      --class_prompt="a photo of person" \
      --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
      --resolution=512 \
      --train_batch_size=2 \
      --gradient_accumulation_steps=1 \
      --learning_rate=5e-7 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --num_class_images=200 \
      --max_train_steps=$max_train_steps \
      --checkpointing_steps=$max_train_steps \
      --center_crop \
      --mixed_precision=bf16 \
      --prior_generation_precision=bf16 \
      --sample_batch_size=16
    echo "rename $REF_MODEL_PATH/checkpoint-${max_train_steps} to $REF_MODEL_PATH/checkpoint-${max_train_steps}-id${data_id}"
    mv $REF_MODEL_PATH/checkpoint-${max_train_steps} $REF_MODEL_PATH/checkpoint-${max_train_steps}-id${data_id}
fi



# ------------------------- Train DreamBooth model on set A -------------------------
# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_PATH  \
#   --enable_xformers_memory_efficient_attention \
#   --train_text_encoder \
#   --instance_data_dir=$CLEAN_TRAIN_DIR\
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$REF_MODEL_PATH \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --instance_prompt="a photo of sks person" \
#   --class_prompt="a photo of person" \
#   --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-7 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=200 \
#   --max_train_steps=$max_train_steps \
#   --checkpointing_steps=$max_train_steps \
#   --center_crop \
#   --mixed_precision=bf16 \
#   --prior_generation_precision=bf16 \
#   --sample_batch_size=16
# mv $REF_MODEL_PATH/checkpoint-${max_train_steps} $REF_MODEL_PATH/checkpoint-${max_train_steps}-id${data_id}
# 

# ------------------------- Train FSMG on set B -------------------------
export CLEAN_ADV_DIR="$data_path/$dataset_name/$data_id/set_B"
export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/tmp_ADVERSARIAL"

mkdir -p $OUTPUT_DIR
cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise
max_r=$(echo "scale=6; ($r + 0.1) / 127.5" | bc -l)
echo $max_r
accelerate launch attacks/fsmg.py \
  --pretrained_model_name_or_path="$REF_MODEL_PATH/checkpoint-${max_train_steps}-id${data_id}"  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$CLEAN_ADV_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks person" \
  --resolution=512 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=$attack_steps \
  --checkpointing_steps=$attack_steps \
  --center_crop \
  --pgd_alpha=5e-3 \
  --pgd_eps=$max_r


# ------------------------- Train DreamBooth on perturbed examples -------------------------
# export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/$attack_steps"
# export DREAMBOOTH_OUTPUT_DIR="outputs/$EXPERIMENT_NAME/n000050_DREAMBOOTH"

# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_PATH  \
#   --enable_xformers_memory_efficient_attention \
#   --train_text_encoder \
#   --instance_data_dir=$INSTANCE_DIR \
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --instance_prompt="a photo of sks person" \
#   --class_prompt="a photo of person" \
#   --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-7 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=200 \
#   --max_train_steps=1000 \
#   --checkpointing_steps=500 \
#   --center_crop \
#   --mixed_precision=bf16 \
#   --prior_generation_precision=bf16 \
#   --sample_batch_size=16

