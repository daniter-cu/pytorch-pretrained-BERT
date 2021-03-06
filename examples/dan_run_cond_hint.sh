#!/bin/sh

python train_cond_with_copy.py \
  --bert_model bert-base-uncased \
  --do_train \
  --train_file ../../Squad2Generative/data/train-v2.0.json \
  --dev_file ../../Squad2Generative/data//dev-v2.0.json \
  --question_parts_file ../qparses2/parsed_qs_labels%s.pkl \
  --output_dir ../model_copy \
  --num_train_epochs 10.0 \
  --learning_rate 3e-5 \
  --train_batch_size 16 \
  --max_seq_length 256
