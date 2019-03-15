#!/bin/sh

python run_chunk_lm_eval.py \
  --bert_model bert-base-uncased \
  --dev_file ../../Squad2Generative/data//dev-v2.0.json \
  --chunk_file_test ../val_data_chunks.pkl \
  --output_file chunk_losses.csv \
  --model_weights ../model_chunk_bak/pytorch_model.bin \
  --learning_rate 3e-5 \
  --train_batch_size 1 \
  --max_seq_length 256
