
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../")

import os
import logging
import argparse
import json
from tqdm import tqdm, trange
import csv

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel #BertForPreTraining
from pytorch_pretrained_bert.optimization import BertAdam
from autoreg_util import RNNModel, InputExample, InputFeatures, BERTDataset

from torch.utils.data import Dataset
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--eval_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input eval corpus.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file where the scores will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        action='store_true',
                        default=True,
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--test_run",
                        action='store_true',
                        default=False,
                        help="If true, shortcut the input data.")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # Load eval_data
    eval_dataset_answerable = BERTDataset(args.eval_file, tokenizer, seq_len=args.max_seq_length,
                                on_memory=args.on_memory, answerable=True)
    eval_dataset_unanswerable = BERTDataset(args.eval_file, tokenizer, seq_len=args.max_seq_length,
                               on_memory=args.on_memory, answerable=False)

    # Prepare model
    if n_gpu > 0:
        model_state_dict = torch.load(args.bert_model)
    else:
        model_state_dict = torch.load(args.bert_model, map_location='cpu')
    context_model = BertModel.from_pretrained("bert-base-uncased")#, state_dict=model_state_dict)
    question_model = BertModel.from_pretrained("bert-base-uncased")#, state_dict=model_state_dict)
    context_model.to(device)
    question_model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        context_model = DDP(context_model)
        question_model = DDP(question_model)
    elif n_gpu > 1:
        context_model = torch.nn.DataParallel(context_model)
        question_model = torch.nn.DataParallel(question_model)

    # Prepare optimizer
    print("Checking the vocab size:", len(tokenizer.vocab))
    # 768 is bert hidden size, 256 is GRU hidden size, 1 is the layers in the GRU
    model = RNNModel("GRU", len(tokenizer.vocab), 768, 768, 1, context_model, question_model, ngpu=n_gpu)
    model.load_state_dict(model_state_dict)
    model.to(device)

    # eval loader
    eval_sampler_ans = SequentialSampler(eval_dataset_answerable)
    eval_dataloader_ans = DataLoader(eval_dataset_answerable, sampler=eval_sampler_ans,
                                     batch_size=args.train_batch_size)
    eval_sampler_unans = SequentialSampler(eval_dataset_unanswerable)
    eval_dataloader_unans = DataLoader(eval_dataset_unanswerable, sampler=eval_sampler_unans,
                                       batch_size=args.train_batch_size)


    criterion = nn.CrossEntropyLoss()
    model.init_hidden(args.train_batch_size)
    with torch.no_grad():
        model.eval()
        with open(args.output_file, "w") as handle:
            loss_writer = csv.writer(handle, delimiter=',')

            eval_loss_ans = 0
            for batch_i, eval_batch in enumerate(eval_dataloader_ans):
                if batch_i % 1000 == 0:
                    print("#### DANITER completed answerable", batch_i)
                eids = eval_batch[-1]
                eval_batch = tuple(t.to(device) for t in eval_batch[:-1])
                question_ids, question_mask, context_ids, context_mask, targets = eval_batch
                output, _ = model(context_ids, context_mask, question_ids, question_mask)
                loss = criterion(output.view(-1, len(tokenizer.vocab)), question_ids.view(-1))
                loss_writer.writerow([eids[0], loss.item(), "ANS"])
                eval_loss_ans += loss.item()
            print("##### DANITER EVAL LOSS IS (ANSWERABLE) : ", eval_loss_ans)

            eval_loss_unans = 0
            for batch_i, eval_batch in enumerate(eval_dataloader_unans):
                if batch_i % 1000 == 0:
                    print("#### DANITER completed unanswerable", batch_i)
                eids = eval_batch[-1]
                eval_batch = tuple(t.to(device) for t in eval_batch[:-1])
                question_ids, question_mask, context_ids, context_mask, targets = eval_batch
                output, _ = model(context_ids, context_mask, question_ids, question_mask)
                loss = criterion(output.view(-1, len(tokenizer.vocab)), question_ids.view(-1))
                loss_writer.writerow([eids[0], loss.item(), "UNANS"])
                eval_loss_unans += loss.item()
            print("##### DANITER EVAL LOSS IS (UNANSWERABLE) : ", eval_loss_unans)





if __name__ == "__main__":
    main()
