#!/bin/sh

python run_chunk_lm_finetune.py \
  --bert_model bert-base-uncased \
  --do_train \
  --train_file ../../Squad2Generative/data/train-v2.0.json \
  --output_dir ../model_chunk \
  --num_train_epochs 10.0 \
  --learning_rate 3e-5 \
  --train_batch_size 16 \
  --max_seq_length 256 
