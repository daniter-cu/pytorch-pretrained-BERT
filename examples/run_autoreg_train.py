
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

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel #BertForPreTraining
from pytorch_pretrained_bert.optimization import BertAdam
from autoreg_util import BERTDataset, RNNModel, warmup_linear

from torch.utils.data import Dataset
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus.")
    parser.add_argument("--eval_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input eval corpus.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
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
    parser.add_argument("--ft_bert",
                        action='store_true',
                        default=False,
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

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    #train_examples = None
    num_train_steps = None
    if args.do_train:
        print("Loading Train Dataset", args.train_file)
        train_dataset = BERTDataset(args.train_file, tokenizer, seq_len=args.max_seq_length,
                                     on_memory=args.on_memory)
        num_train_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Load eval_data
    eval_dataset_answerable = BERTDataset(args.eval_file, tokenizer, seq_len=args.max_seq_length,
                                on_memory=args.on_memory, answerable=True)
    eval_dataset_unanswerable = BERTDataset(args.eval_file, tokenizer, seq_len=args.max_seq_length,
                               on_memory=args.on_memory, answerable=False)

    # Prepare model
    context_model = BertModel.from_pretrained(args.bert_model)# BertForPreTraining.from_pretrained(args.bert_model)
    question_model = BertModel.from_pretrained(args.bert_model)
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
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and "bert" in n], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and "bert" in n], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if "bert" not in n]}
        ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on file.__next__
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        # eval loader
        eval_sampler_ans = SequentialSampler(eval_dataset_answerable)
        eval_dataloader_ans = DataLoader(eval_dataset_answerable, sampler=eval_sampler_ans,
                                         batch_size=args.train_batch_size)
        eval_sampler_unans = SequentialSampler(eval_dataset_unanswerable)
        eval_dataloader_unans = DataLoader(eval_dataset_unanswerable, sampler=eval_sampler_unans,
                                           batch_size=args.train_batch_size)



        model.train()
        criterion = nn.CrossEntropyLoss()#len(tokenizer.vocab))
        model.init_hidden(args.train_batch_size)
        for epoch_i in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                eids = batch[-1]
                batch = tuple(t.to(device) for t in batch[:-1])
                question_ids, question_mask, context_ids, context_mask, targets = batch
                # daniter todo : fix inputs and change model to include some head
                output, _ = model(context_ids, context_mask, question_ids, question_mask)
                loss = criterion(output.view(-1, len(tokenizer.vocab)), targets.view(-1))
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += question_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        if len(param_group['params']) < 10:
                            param_group['lr'] = lr_this_step * 100
                        else:
                            if args.ft_bert:
                                param_group['lr'] = lr_this_step
                            else:
                                param_group['lr'] = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if global_step % 10 == 0:
                        print("Current Loss: ", tr_loss)
                        tr_loss = 0

                # eval
                if global_step % 20 == 0:
                    with torch.no_grad():
                        model.eval()
                        eval_loss_ans = 0
                        for batch_i, eval_batch in enumerate(eval_dataloader_ans):
                            if batch_i > 40:
                                break
                            eids = eval_batch[-1]
                            eval_batch = tuple(t.to(device) for t in eval_batch[:-1])
                            question_ids, question_mask, context_ids, context_mask, targets = eval_batch
                            output, _ = model(context_ids, context_mask, question_ids, question_mask)
                            loss = criterion(output.view(-1, len(tokenizer.vocab)), targets.view(-1))
                            eval_loss_ans += loss.item()
                        print("##### DANITER EVAL LOSS IS (ANSWERABLE) : ", eval_loss_ans)

                        eval_loss_unans = 0
                        for batch_i, eval_batch in enumerate(eval_dataloader_unans):
                            if batch_i > 40:
                                break
                            eids = eval_batch[-1]
                            eval_batch = tuple(t.to(device) for t in eval_batch[:-1])
                            eval_batch = tuple(t.to(device) for t in eval_batch)
                            question_ids, question_mask, context_ids, context_mask, targets = eval_batch
                            output, _ = model(context_ids, context_mask, question_ids, question_mask)
                            loss = criterion(output.view(-1, len(tokenizer.vocab)), targets.view(-1))
                            eval_loss_unans += loss.item()
                        print("##### DANITER EVAL LOSS IS (UNANSWERABLE) : ", eval_loss_unans)
                        model.train()



                    if args.test_run and global_step == 20:
                        logger.info("** ** * Saving fine - tuned model ** ** * ")
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                        if args.do_train:
                            torch.save(model_to_save.state_dict(), output_model_file)
                        return

            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "pytorch_model"+str(epoch_i) +".bin")
            if args.do_train:
                torch.save(model_to_save.state_dict(), output_model_file)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        tokens_a.pop()

    while True:
        total_length = len(tokens_b)
        if total_length <= max_length:
            break
        tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


if __name__ == "__main__":
    main()
