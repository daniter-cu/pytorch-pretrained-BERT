#!/bin/sh

python run_ent_lm.py \
  --bert_model bert-base-uncased \
  --do_train \
  --train_file ../dataset/train-v2.0.json \
  --dev_file ../dataset/dev-v2.0.json \
  --chunk_file_train ../entity_labels_v2.pkl \
  --chunk_file_test ../entity_labels_v2.pkl \
  --output_dir ../model_ents \
  --num_train_epochs 1.0 \
  --learning_rate 3e-5 \
  --train_batch_size 16 \
  --max_seq_length 256 \
  --test_run
