#!/bin/sh

python run_qa_lm_finetune.py \
  --bert_model bert-base-uncased \
  --do_train \
  --train_file ../../Squad2Generative/data/train-v2.0.json \
  --output_dir ../models_3l \
  --num_train_epochs 3.0 \
  --learning_rate 1e-5 \
  --train_batch_size 32 \
  --max_seq_length 128
