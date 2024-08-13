pertrained_model='/mnt/cfs/NLP/zcl/huggingface/models/chinese-lert-base'

train_file='/mnt/cfs/NLP/zcl/subjects_classification/datasets/train_data_hard.json'
valid_file='/mnt/cfs/NLP/zcl/subjects_classification/datasets/eval_data_run.json'
test_file='/mnt/cfs/NLP/zcl/subjects_classification/datasets_badclean/test_data.json'
 
output_dir='/mnt/cfs/NLP/zcl/subjects_classification/lert/output_badclean_pretrain_rl_hard_512'
log_dir='/mnt/cfs/NLP/zcl/subjects_classification/lert/log_badclean_pretrain_rl_hard_512'
cache_dir='/mnt/cfs/NLP/zcl/subjects_classification/lert/cache_badclean_pretrain_rl_hard_512'

max_seq_length=512
per_device_train_batch_size=120
per_device_eval_batch_size=120
learning_rate=5e-5
num_train_epochs=2

preprocessing_num_workers=128

# save_total_limit=5
warmup_steps=2000
warmup_ratio=0.1
weight_decay=0.001

eval_steps=2000
evaluation_strategy="steps"
save_steps=2000
save_strategy="steps"
logging_steps=5

label_smoothing_factor=0.1
neftune_noise_alpha=0.1

classifier_dropout=0.1

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 
python train.py \
    --model_name_or_path ${pertrained_model} \
    --train_file ${train_file} \
    --validation_file ${valid_file} \
    --test_file ${test_file} \
    --output_dir ${output_dir} \
    --logging_dir ${log_dir} \
    --logging_first_step True \
    --logging_steps ${logging_steps} \
    --cache_dir ${cache_dir} \
    --max_seq_length ${max_seq_length} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --learning_rate ${learning_rate} \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --num_train_epochs ${num_train_epochs} \
    --save_steps ${save_steps} \
    --save_strategy ${save_strategy} \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --evaluation_strategy ${evaluation_strategy} \
    --eval_steps ${eval_steps} \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --overwrite_output_dir True \
    --overwrite_cache False \
    --label_column_name subject_id \
    --text_column_names question_clean \
    --save_safetensors False \
    --label_column_name subject_id \
    --remove_columns question_cut \
    --report_to tensorboard \
    --run_name lert_512 \
    --shuffle_train_dataset True \
    --neftune_noise_alpha ${neftune_noise_alpha} \
    --label_smoothing_factor ${label_smoothing_factor} \
    --classifier_dropout ${classifier_dropout} \
    --preprocessing_num_workers ${preprocessing_num_workers} \
    --lr_scheduler_type cosine \