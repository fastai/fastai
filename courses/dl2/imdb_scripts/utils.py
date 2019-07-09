"""
Utility methods for data processing.
"""
import pandas as pd
import numpy as np
from fastai import F, to_device
import torch
from tqdm import tqdm
import re
import csv

EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'
PAD_TOKEN_ID = 1

number_match_re = re.compile(r'^[0-9]+(?:[,.][0-9]*)*$')
number_split_re = re.compile(r'([,.])')


def replace_number(token):
    """Replaces a number and returns a list of one or multiple tokens."""
    if number_match_re.match(token):
        return number_split_re.sub(r' @\1@ ', token)
    return token


def read_file(file_path, outname):
    """Reads a text file and writes it to a .csv."""
    with open(file_path, encoding='utf8') as f:
        text = f.readlines()
    df = pd.DataFrame(
        {'text': np.array(text), 'labels': np.zeros(len(text))},
        columns=['labels', 'text'])
    df.to_csv(file_path.parent / f'{outname}.csv', header=False, index=False)


def read_whitespace_file(filepath):
    """Reads a file and prepares the tokens."""
    tokens = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            # newlines are replaced with EOS
            tokens.append(line.split() + [EOS])
    return np.array(tokens)


def read_imdb(file_path, mt):
    toks, lbls = [], []
    print(f'Reading {file_path}...')
    with open(file_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            label, text = row
            lbls.append(label)
            raw_tokens = mt.tokenize(text, return_str=True).split(' ') + [EOS]
            tokens = []
            for token in raw_tokens:
                if number_match_re.match(token):
                    tokens += number_split_re.sub(r' @\1@ ', token).split()
                else:
                    tokens.append(token)
            toks.append(tokens)
    return np.array(toks), np.array(lbls)


class DataStump:
    """Placeholder class as LanguageModelLoader requires object with ids attribute."""
    def __init__(self, ids):
        self.ids = ids
        self.loss_func = F.cross_entropy


def validate(model, ids, bptt=2000):
    """
    Return the validation loss and perplexity of a model
    :param model: model to test
    :param ids: data on which to evaluate the model
    :param bptt: bptt for this evaluation (doesn't change the result, only the speed)
    From https://github.com/sgugger/Adam-experiments/blob/master/lm_val_fns.py#L34
    """
    data = TextReader(np.concatenate(ids), bptt)
    model.eval()
    model.reset()
    total_loss, num_examples = 0., 0
    for inputs, targets in tqdm(data):
        outputs, raws, outs = model(to_device(inputs, None))
        p_vocab = F.softmax(outputs, 1)
        for i, pv in enumerate(p_vocab):
            targ_pred = pv[targets[i]]
            total_loss -= torch.log(targ_pred.detach())
        num_examples += len(inputs)
    mean = total_loss / num_examples  # divide by total number of tokens
    return mean, np.exp(mean)


class TextReader():
    """ Returns a language model iterator that iterates through batches that are of length N(bptt,5)
    The first batch returned is always bptt+25; the max possible width.  This is done because of they way that pytorch
    allocates cuda memory in order to prevent multiple buffers from being created as the batch width grows.
    From: https://github.com/sgugger/Adam-experiments/blob/master/lm_val_fns.py#L3
    """
    def __init__(self, nums, bptt, backwards=False):
        self.bptt,self.backwards = bptt,backwards
        self.data = self.batchify(nums)
        self.i,self.iter = 0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter<len(self):
            res = self.get_batch(self.i, self.bptt)
            self.i += self.bptt
            self.iter += 1
            yield res

    def __len__(self): return self.n // self.bptt

    def batchify(self, data):
        data = np.array(data)[:,None]
        if self.backwards: data=data[::-1]
        return torch.LongTensor(data)

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)