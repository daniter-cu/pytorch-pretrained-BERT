#!/bin/sh

python train_cond_with_hint.py \
  --bert_model bert-base-uncased \
  --do_train \
  --multihint \
  --train_file ../dataset/train-v2.0.json \
  --dev_file ../dataset/dev-v2.0.json \
  --question_parts_file ../qparts/part_labels%s.pkl \
  --output_dir ../model_hint \
  --num_train_epochs 1.0 \
  --learning_rate 3e-5 \
  --train_batch_size 16 \
  --max_seq_length 256 \
  --test_run
