#!/bin/sh

python run_autoreg_eval.py \
  --bert_model ../autoreg_model/pytorch_model.bin \
  --eval_file ../dataset/dev-v2.0.json \
  --output_file deleteme \
  --train_batch_size 1 \
  --max_seq_length 128 
