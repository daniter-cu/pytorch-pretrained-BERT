#!/bin/sh

python train_negation_exp.py \
  --bert_model bert-base-uncased \
  --do_train \
  --train_file ../../negation_training_data.pkl \
  --dev_file ../../negation_test_data.pkl \
  --output_dir ../../model_neg_exp \
  --num_train_epochs 1.0 \
  --learning_rate 3e-5 \
  --train_batch_size 16 \
  --max_seq_length 256 \
  --test_run
