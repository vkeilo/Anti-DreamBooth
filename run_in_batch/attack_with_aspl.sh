export EXPERIMENT_NAME="ASPL"
export MODEL_PATH=$model_path
export CLEAN_TRAIN_DIR="$data_path/$dataset_name/$data_id/set_A" 
export CLEAN_ADV_DIR="$data_path/$dataset_name/$data_id/set_B"
export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/tmp_ADVERSARIAL"
export CLASS_DIR=$class_dir


# ------------------------- Train ASPL on set B -------------------------
rm -r $OUTPUT_DIR/*
mkdir -p $OUTPUT_DIR
cp -r $CLEAN_TRAIN_DIR $OUTPUT_DIR/image_clean_ref
cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise
echo $r
max_r=$(echo "scale=6; ($r + 0.1) / 127.5" | bc -l)
echo $max_r
aspl_cmd="""accelerate launch attacks/aspl.py \
--pretrained_model_name_or_path=$MODEL_PATH  \
--enable_xformers_memory_efficient_attention \
--instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
--instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
--instance_prompt='a photo of sks person' \
--class_data_dir=$class_data_dir \
--num_class_images=200 \
--class_prompt='a photo of person' \
--output_dir=$OUTPUT_DIR \
--center_crop \
--with_prior_preservation \
--prior_loss_weight=1.0 \
--resolution=512 \
--train_text_encoder \
--train_batch_size=1 \
--max_train_steps=$attack_steps \
--max_f_train_steps=3 \
--max_adv_train_steps=6 \
--checkpointing_iterations=$attack_steps \
--learning_rate=5e-7 \
--pgd_alpha=5e-3 \
--pgd_eps=$max_r \
--mixed_precision=$mixed_precision  \
--report_to=$report_to
"""
echo $aspl_cmd
eval $aspl_cmd


# ------------------------- Train DreamBooth on perturbed examples -------------------------
# export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/$attack_steps"
# export DREAMBOOTH_OUTPUT_DIR="outputs/$EXPERIMENT_NAME/tmp_DREAMBOOTH"

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
#   --sample_batch_size=8

