
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


# LSTM LM HEAD

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, context_bert, question_bert, dropout=0.5,
                 tie_weights=False, ngpu=1):
        super(RNNModel, self).__init__()
        self.ngpu = ngpu
        self.drop = nn.Dropout(dropout)
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.context_bert = context_bert
        self.question_bert = question_bert

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            #self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, context_ids, context_mask, question_ids, question_mask):
        #emb = self.drop(self.encoder(input))
        _, hidden = self.context_bert(context_ids, token_type_ids=None,
                                      attention_mask=context_mask, output_all_encoded_layers=False)
        emb, _ = self.question_bert(question_ids, token_type_ids=None, attention_mask=question_mask,
                                    output_all_encoded_layers=False)
        output, hidden = self.rnn(emb, hidden.unsqueeze(0))
        #output = self.drop(output) # todo daniter: probably put back dropout
        if self.ngpu > 10:
            decoded = self.decoder(output.view(-1, output.size(2)))
        else:
            decoded = self.decoder(output.contiguous().view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class BERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", on_memory=True, answerable=True):
        self.use_answerable = answerable
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
                        id = qas[j]['id']
                        if self.use_answerable and unanswerable:
                            continue
                        if not self.use_answerable and not unanswerable:
                            continue
                        self.questions.append(question)
                        self.examples.append((len(self.contexts)-1, len(self.questions)-1))
                        self.example_ids.append(id)

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

        t1, t2, is_next_label = self.random_sent(item)

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        tokens_b = self.tokenizer.tokenize(t2)

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.question_ids),
                       torch.tensor(cur_features.question_mask),
                       torch.tensor(cur_features.context_ids),
                       torch.tensor(cur_features.context_mask),
                       torch.tensor(cur_features.targets),
                       self.example_ids[item])

        return cur_tensors

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_line(index)
        # Daniter we do not do next sentence prediction

        assert len(t1) > 0
        assert len(t2) > 0
        return t1, t2, 0

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
            raise Exception("Not supported")


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, context_ids, context_mask, question_ids, question_mask, targets):
        self.context_ids = context_ids
        self.context_mask = context_mask
        self.question_ids = question_ids
        self.question_mask = question_mask
        self.targets = targets


def random_word(tokens, tokenizer, question=False):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    #output_label = []
    return tokens, [-1] * len(tokens)

    # if not question:
    #     return tokens, [-1] * len(tokens)
    #
    # for i, token in enumerate(tokens):
    #     prob = random.random()
    #     # mask token with 15% probability
    #     if prob < 0.15:
    #         prob /= 0.15
    #
    #         # 80% randomly change token to mask token
    #         if prob < 0.8:
    #             tokens[i] = "[MASK]"
    #
    #         # 10% randomly change token to random token
    #         elif prob < 0.9:
    #             tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]
    #
    #         # -> rest 10% randomly keep current token
    #
    #         # append current token to output (we will predict these later)
    #         try:
    #             output_label.append(tokenizer.vocab[token])
    #         except KeyError:
    #             # For unknown words (should not occur with BPE vocab)
    #             output_label.append(tokenizer.vocab["[UNK]"])
    #             logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
    #     else:
    #         # no masking token (will be ignored by loss function later)
    #         output_label.append(-1)
    #
    # return tokens, output_label


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
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    t1_random, t1_label = random_word(tokens_a, tokenizer, question=False)
    t2_random, t2_label = random_word(tokens_b, tokenizer, question=True)
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
    context_tokens = []
    context_tokens.append("[CLS]")
    context_tokens += tokens_a

    question_tokens = []
    question_tokens.append("[CLS]")
    question_tokens += tokens_b

    context_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    question_ids = tokenizer.convert_tokens_to_ids(question_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    context_mask = [1] * len(context_ids)
    question_mask = [1] * len(question_ids)
    targets = list(question_ids)


    # Zero-pad up to the sequence length.
    while len(context_ids) < max_seq_length:
        context_ids.append(0)
        context_mask.append(0)

    while len(question_ids) < max_seq_length:
        question_ids.append(0)
        question_mask.append(0)
        targets.append(-1)

    assert len(question_ids) == max_seq_length
    assert len(question_mask) == max_seq_length
    assert len(targets) == max_seq_length
    assert len(context_ids) == max_seq_length
    assert len(context_mask) == max_seq_length

    # if example.guid < 5:
    #     logger.info("*** Example ***")
    #     logger.info("guid: %s" % (example.guid))
    #     logger.info("tokens: %s" % " ".join(
    #             [str(x) for x in question_tokens]))
    #     logger.info("input_ids: %s" % " ".join([str(x) for x in question_ids]))
    #     logger.info("input_mask: %s" % " ".join([str(x) for x in question_mask]))

    features = InputFeatures(context_ids, context_mask, question_ids, question_mask, targets)
    return features


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
    model_state_dict = torch.load(args.bert_model, map_location='cpu') #TODO daniter: remove this map_location
    ## TODO daniter: check if bert model is being loaded correctly
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
    import pdb;
    pdb.set_trace()
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
