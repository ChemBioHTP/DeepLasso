#!/usr/bin/python
# coding: utf-8

import pickle
import numpy as np
import torch
import pandas as pd
from model import *
from data import *


class Predictor(object):
    def __init__(self, model):
        self.model = model
    def inference(self, data):
        predicted_values = self.model.forward(data)
        return predicted_values

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_csv(name:list, content:list, setting:str):
    tmp_df = pd.DataFrame(columns=name, data=content)
    tmp_df.to_csv(f'./output/{setting}_data.csv', encoding='gbk')

if __name__ == "__main__":

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('./inference/case/')
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    sst = load_tensor(dir_input + 'ssts', torch.LongTensor)
    word_dict = load_pickle(dir_input + 'sequence_dict.pickle')
    sst_dict = load_pickle(dir_input + 'topolgy_dict.pickle')
    n_word = len(word_dict)
    n_top = len(sst_dict)
    
    dataset = list(zip(proteins, sst)) #exempl the sst (proteins, sst,interaction)
    
    model =  DeepLasso(1770,8, 20, 5, 11, 3, 1).to(device)
    model.load_state_dict(torch.load("./params_trained/model.pt", map_location = device))
    predict_enrichment = Predictor(model)

    """Start predicting."""
    with open("./inference/case.csv", 'r') as infiles:
        lines = infiles.readlines()
        i=0
        for line in lines[1:] :
            line_item = list()
            data = line.strip().split('\t')
            sequence = data[0]
            sst = data[-1]
            words = torch.LongTensor(split_sequence(sequence, 3))
            ssts= torch.LongTensor(split_topology(sst, 3))
            inputs = [words, ssts]
            prediction_values = predict_enrichment.inference(inputs)
            enrichvalues = prediction_values.item()
            enrichvalues = '%.9f'%(enrichvalues)
            print(enrichvalues)


