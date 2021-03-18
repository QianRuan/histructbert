#!/bin/bash
: '
#######################################----------------------------cnndm_hs_bert_s_sin_sum_bs400ts50K_4gpu
'
: '
#################------------------------------------------------------------TRAIN
'
export CUDA_VISIBLE_DEVICES=1,5,6,7
python histruct/src/train.py -task ext \
-mode train \
-add_tok_struct_emb false \
-add_sent_struct_emb true \
-sent_pos_emb_type sinusoidal \
-sent_se_comb_mode sum \
-bert_data_path bert_data_cnndm/cnndm \
-ext_dropout 0.1 \
-model_path models/cnndm_hs_bert_s_sin_sum_bs400ts50K_4gpu \
-lr 2e-3 \
-visible_gpus 1,5,6,7 \
-report_every 50 \
-save_checkpoint_steps 1000 \
-batch_size 500 \
-train_steps 50000 \
-accum_count 2 \
-log_file models/cnndm_hs_bert_s_sin_sum_bs400ts50K_4gpu/train.log \
-temp_dir temp \
-use_interval true \
-warmup_steps 10000 \
-max_pos 512
: '
#################------------------------------------------------------------EVAL
'
python histruct/src/train.py -task ext \-mode validate \
-test_all true \
-select_top_n_sent 3 \
-batch_size 3000 \
-test_batch_size 500 \
-bert_data_path bert_data_cnndm/cnndm \
-eval_folder eval \
-model_path models/cnndm_hs_bert_s_sin_sum_bs400ts50K_4gpu \
-log_file models/cnndm_hs_bert_s_sin_sum_bs400ts50K_4gpu/eval/eval.log \
-result_path models/cnndm_hs_bert_s_sin_sum_bs400ts50K_4gpu/eval/cnndm.test \
-sep_optim true \
-visible_gpus 5,6,7 \
-max_pos 512 \
-max_length 200 \
-alpha 0.95 \
-min_length 50 
