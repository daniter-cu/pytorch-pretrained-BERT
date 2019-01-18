#!/bin/sh

python run_qa_lm_finetune.py \
  --bert_model bert-base-uncased \
  --do_train \
  --train_file ../dataset/train-v2.0.json \
  --output_dir ../model_local \
  --num_train_epochs 1.0 \
  --learning_rate 3e-5 \
  --train_batch_size 32 \
  --max_seq_length 128 \
  --test_run
