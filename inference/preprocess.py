#!/usr/bin/python
# coding: utf-8

# Author: LE YUAN
# Date: 2020-10-03

import math
import json
import pickle
import numpy as np
from collections import defaultdict
import torch

word_dict = defaultdict(lambda: len(word_dict))
sst_dict = defaultdict(lambda: len(sst_dict))

proteins = list()
regression = list()
ssts = list()
mutants = list()

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    # print(sequence)
    words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    return np.array(words)
    # return word_dict

def split_topology(topolgy, ngram):
    topolgy = '-' + topolgy + '='
    ssts = [sst_dict[topolgy[i:i+ngram]] for i in range(len(topolgy)-ngram+1)]
    return np.array(ssts)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def main() :
    with open('./test_data_1133.csv', 'r') as infile :
        enrich_data = infile.readlines()

    ngram = 3

    i = 0
    for line in enrich_data[1:] :
        line_item = list()
        data = line.strip().split(',')
        sequence = data[0]
        enrich = data[1]
        ss = data[-1]
        """protein sequence embedding"""
        words = split_sequence(sequence,ngram)
        proteins.append(words)
        """secondary topology embedding"""
        ss_words = split_topology(ss, 3) #ss topology for ngram=3
        ssts.append(ss_words)
        """regression"""
        #mutants.append(name)
        regression.append(np.array([float(enrich)]))
        print(float(enrich))

    np.save('./test/'+'proteins', proteins)
    np.save('./test/'+'ssts', ssts)
    np.save('./test/' + 'regression', regression)
    #np.save('./input_top/' + 'mutants', mutants)
    dump_dictionary(word_dict, './test/sequence_dict.pickle')
    dump_dictionary(sst_dict, './test/topolgy_dict.pickle')



if __name__ == '__main__' :
    main()
