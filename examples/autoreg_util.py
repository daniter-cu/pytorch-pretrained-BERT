
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

# LSTM LM HEAD

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

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
    targets = list(question_ids[1:])


    # Zero-pad up to the sequence length.
    while len(context_ids) < max_seq_length:
        context_ids.append(0)
        context_mask.append(0)

    while len(question_ids) < max_seq_length:
        question_ids.append(0)
        question_mask.append(0)

    while len(targets) < max_seq_length:
        targets.append(0)

    assert len(question_ids) == max_seq_length
    assert len(question_mask) == max_seq_length
    assert len(targets) == max_seq_length
    assert len(context_ids) == max_seq_length
    assert len(context_mask) == max_seq_length

    if False:#example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in question_tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in question_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in question_mask]))

    features = InputFeatures(context_ids, context_mask, question_ids, question_mask, targets)
    return features


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