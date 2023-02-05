#!/usr/bin/python

#The data preprocss script is adopted from DLKcat, author Le Yuan
#https://github.com/SysBioChalmers/DLKcat

import math
import json
import pickle
import numpy as np
from collections import defaultdict
import torch

word_dict = defaultdict(lambda: len(word_dict))
sst_dict = defaultdict(lambda: len(sst_dict))

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    return np.array(words)

def split_topology(topolgy, ngram):
    topolgy = '-' + topolgy + '='
    ssts = [sst_dict[topolgy[i:i+ngram]] for i in range(len(topolgy)-ngram+1)]
    return np.array(ssts)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

