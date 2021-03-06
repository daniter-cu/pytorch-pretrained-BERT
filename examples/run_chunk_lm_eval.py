
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../")

import os
import logging
import argparse
import json, csv
import pickle
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining
from pytorch_pretrained_bert.optimization import BertAdam

from torch.utils.data import Dataset
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


class BERTDataset(Dataset):
    def __init__(self, corpus_path, chunk_path, tokenizer, seq_len, encoding="utf-8", on_memory=True, keep_answerable=True):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_path = corpus_path
        self.encoding = encoding

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.questions = []
        self.contexts = []
        self.examples = []
        self.example_ids = []
        self.mask_ids = []

        # load samples into memory
        if on_memory:
            # DANITER: Load Squad data
            with open(corpus_path, 'r') as handle:
                jdata = json.load(handle)
                data = jdata['data']

            for i in tqdm(range(len(data)), "Loading Squad", total=len(data)):
                section = data[i]['paragraphs']
                for sec in section:
                    context = sec['context']
                    self.contexts.append(context)
                    qas = sec['qas']
                    for j in range(len(qas)):
                        question = qas[j]['question']
                        unanswerable = qas[j]['is_impossible']
                        self.questions.append(question)
                        if unanswerable and keep_answerable:
                            continue
                        if not keep_answerable and not unanswerable:
                            continue
                        self.examples.append((len(self.contexts)-1, len(self.questions)-1))
                        self.example_ids.append(qas[j]['id'])
                        self.mask_ids.append(-1)


            with open(chunk_path, "rb") as handle:
                self.training_data_map = pickle.load(handle)


        # load samples later lazily from disk
        else:
            raise Exception("No supported")

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return len(self.examples)

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            raise Exception("No supported")

        while True:
            t1, t2, target, is_next_label = self.get_example(item)

            # tokenize
            tokens_a = self.tokenizer.tokenize(t1)
            tokens_b = self.tokenizer.tokenize(t2)
            if target[0] is None or len(tokens_a) + len(target[1]) + 3 > self.seq_len :
                item += 1
            else:
                break

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label, target=target)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.is_next),
                       self.example_ids[item],
                       self.mask_ids[item])

        return cur_tensors

    def get_example(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_line(index)


        candidate_targs = list(self.training_data_map[self.examples[index][1]])
        cand_targ_with_index = list(zip(candidate_targs, range(len(candidate_targs))))
        random.shuffle(cand_targ_with_index)
        targets = [(tg, targ_id) for tg, targ_id in
                   cand_targ_with_index
                   if tg[0] is not None]

        if targets:
            target = targets[0][0]
            self.mask_ids[index] = targets[0][1]
        else:
            target = (None, None) # keep same shape
        # Daniter we do not do next sentence prediction

        assert len(t1) > 0
        assert len(t2) > 0
        return t1, t2, target, 1

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert item < len(self.examples)
        if self.on_memory:
            # DANITER - get the context and question pair based on the example indexes
            context_idx, question_idx = self.examples[item]
            t1 = self.contexts[context_idx]
            t2 = self.questions[question_idx]

            # used later to avoid random nextSentence from same doc
            return t1, t2
        else:
            raise Exception("No supported")


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None, target=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.target = target


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids


def random_word(tokens, tokenizer, question=False, tokens_question=None, tokens_target=None):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    if not question:
        return tokens, [-1] * len(tokens)
    else:
        assert tokens_question is not None
        assert tokens_target is not None

        return tokens_question, tokenizer.convert_tokens_to_ids(tokens_target)

    return tokens, output_label


def convert_example_to_features(example, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    tokens_question = example.target[0]
    tokens_target = example.target[1]
    for tokb, tokq in zip(tokens_b, tokens_question):
        if tokq != '[MASK]':
            assert tokq == tokb, (tokens_b, tokens_question)
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    t1_random, t1_label = random_word(tokens_a, tokenizer, question=False)
    t2_random, t2_label = random_word(tokens_b, tokenizer, question=True, tokens_question=tokens_question, tokens_target=tokens_target)
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(t2_random) > 0
    for token in t2_random:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    while len(lm_label_ids) < max_seq_length:
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length, len(input_ids)
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if len(lm_label_ids) != max_seq_length:
        print("ERROR COULD NOT COMPUTE...!",len(lm_label_ids) )
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
            [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("LM label: %s " % (lm_label_ids))
        logger.info("lm label tokens: %s" % (tokenizer.convert_ids_to_tokens([t for t in lm_label_ids if t != -1])))

    assert len(lm_label_ids) == max_seq_length

    # if example.guid < 5:
    #     logger.info("*** Example ***")
    #     logger.info("guid: %s" % (example.guid))
    #     logger.info("tokens: %s" % " ".join(
    #             [str(x) for x in tokens]))
    #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #     logger.info(
    #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #     logger.info("LM label: %s " % (lm_label_ids))
    #     logger.info("lm label tokens: %s" % (tokenizer.convert_ids_to_tokens([ t for t in lm_label_ids if t != -1])))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next)
    return features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_weights",
                        default=None,
                        type=str,
                        required=True,
                        help="model weights.")
    parser.add_argument("--dev_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input dev corpus.")
    parser.add_argument("--chunk_file_test",
                        default=None,
                        type=str,
                        required=True,
                        help="Chunk file.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file where the model losses will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=16,
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

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    val_dataset_ans = BERTDataset(args.dev_file, args.chunk_file_test, tokenizer, seq_len=args.max_seq_length,
                                on_memory=args.on_memory)
    val_dataset_unans = BERTDataset(args.dev_file, args.chunk_file_test, tokenizer, seq_len=args.max_seq_length,
                                on_memory=args.on_memory, keep_answerable=False)

    # Prepare model
    if n_gpu > 0:
        model_state_dict = torch.load(args.model_weights)
    else:
        model_state_dict = torch.load(args.model_weights, map_location='cpu')
    model = BertForPreTraining.from_pretrained(args.bert_model, state_dict=model_state_dict)
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    global_step = 0
    eval_sampler_ans = RandomSampler(val_dataset_ans)
    eval_sampler_unans = RandomSampler(val_dataset_unans)
    eval_dataloader_ans = DataLoader(val_dataset_ans, sampler=eval_sampler_ans, batch_size=args.train_batch_size)
    eval_dataloader_unans = DataLoader(val_dataset_unans, sampler=eval_sampler_unans, batch_size=args.train_batch_size)

    with torch.no_grad():
        model.eval()
        with open(args.output_file, "w") as handle:
            loss_writer = csv.writer(handle, delimiter=',')

            answerable_loss = 0
            for batch_i, eval_batch in enumerate(eval_dataloader_ans):
                if args.test_run and batch_i > 3:
                    break
                eids = eval_batch[-2]
                mask_id = eval_batch[-1].data.numpy()
                eval_batch = tuple(t.to(device) for t in eval_batch[:-2])
                input_ids, input_mask, segment_ids, lm_label_ids, is_next = eval_batch
                loss = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)
                loss_writer.writerow([eids[0], mask_id[0], loss.item(), "ANS"])
                answerable_loss += loss.item()
            print("###### DANITER EVAL LOSS (ANSWERABLE): ", answerable_loss)

            unanswerable_loss = 0
            for batch_i, eval_batch in enumerate(eval_dataloader_unans):
                if args.test_run and batch_i > 3:
                    break
                eids = eval_batch[-2]
                mask_id = eval_batch[-1].data.numpy()
                eval_batch = tuple(t.to(device) for t in eval_batch[:-2])
                input_ids, input_mask, segment_ids, lm_label_ids, is_next = eval_batch
                loss = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)
                loss_writer.writerow([eids[0], mask_id[0], loss.item(), "UNANS"])
                unanswerable_loss += loss.item()
            print("###### DANITER EVAL LOSS (UNANSWERABLE): ", unanswerable_loss)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


if __name__ == "__main__":
    main()
