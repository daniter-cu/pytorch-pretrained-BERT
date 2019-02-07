#!/bin/sh

python run_autoreg_eval.py \
  --bert_model ../autoreg_model/pytorch_model.bin \
  --eval_file ../../Squad2Generative/data/dev-v2.0.json \
  --output_dir deleteme \
  --output_file losses.csv \
  --train_batch_size 1 \
  --max_seq_length 128 
