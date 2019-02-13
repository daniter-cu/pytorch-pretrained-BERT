#!/bin/sh

python run_autoreg_train.py \
  --bert_model bert-base-uncased \
  --do_train \
  --train_file ../dataset/train-v2.0.json \
  --eval_file ../dataset/dev-v2.0.json \
  --output_dir ../autoreg_model_deleteme \
  --num_train_epochs 1.0 \
  --learning_rate 3e-5 \
  --train_batch_size 32 \
  --max_seq_length 128 \
  --test_run
