#!/bin/sh

python run_autoreg_train.py \
  --bert_model bert-base-uncased \
  --do_train \
  --train_file ../../Squad2Generative/data/train-v2.0.json \
   --eval_file ../../Squad2Generative/data/dev-v2.0.json \
  --output_dir ../autoreg_model_lm_noft \
  --num_train_epochs 10.0 \
  --learning_rate 3e-5 \
  --train_batch_size 16 \
  --max_seq_length 128 
