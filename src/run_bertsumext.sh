export BERT_DATA_PATH=../bert_data/
export MODEL_PATH=../models/bertsumext_cnndm
python -m train.py \
-task ext \
-mode train \
-bert_data_path BERT_DATA_PATH \
-ext_dropout 0.1 \
-model_path MODEL_PATH \
-lr 2e-3 \
-visible_gpus 0,1,2 \
-report_every 5 \
-save_checkpoint_steps 100 \
-batch_size 3000 \
-train_steps 500 \
-accum_count 2 \
-log_file ../logs/ext_bert_cnndm \
-use_interval true \
-warmup_steps 100 \
-max_pos 512