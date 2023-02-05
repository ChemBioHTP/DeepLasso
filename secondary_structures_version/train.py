#!/usr/bin/python
# coding: utf-8

import pickle
import sys
import timeit
import math
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error,r2_score
#import scipy.stats as sps
import pandas as pd
from model import * 

class Trainer(object):
    def __init__(self, model, lr,weight_decay):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        y, y_predict = [], []
        for data in dataset:
            loss, correct_values, predicted_values = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
            y.append(correct_values)
            y_predict.append(predicted_values)
        rmse_train = np.sqrt(mean_squared_error(y,y_predict))
        r2_train = r2_score(y,y_predict)
        return loss_total, rmse_train, r2_train


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        y, y_predict = [], []
        for data in dataset :
            (correct_values, predicted_values) = self.model(data, train=False)
            SAE += np.abs(predicted_values-correct_values)
            y.append(correct_values)
            y_predict.append(predicted_values)
        MAE = SAE / N  # mean absolute error.
        rmse = np.sqrt(mean_squared_error(y,y_predict))
        r2 = r2_score(y,y_predict)
        return MAE, rmse, r2, y, y_predict

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, MAEs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def save_csv(name:list, content:list, setting:str):
    tmp_df = pd.DataFrame(columns=name, data=content)
    tmp_df.to_csv(f'./output/{setting}_data.csv', encoding='gbk')

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    """Hyperparameters."""
    ngram=3
    dim=20
    side=5
    window=11
    layer_cnn=3
    layer_output=2
    lr=1e-3
    lr_decay = 0.5
    decay_interval=10
    weight_decay=1e-5
    iteration=15
    setting ='train_preprocess_dim_20_ss_top_model_31th_layer_2_iteration_15_2'

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('./input_31th/')
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    sst = load_tensor(dir_input + 'ssts', torch.LongTensor)
    enrichments = load_tensor(dir_input + 'regression', torch.FloatTensor)
    word_dict = load_pickle(dir_input + 'sequence_dict.pickle')
    sst_dict = load_pickle(dir_input + 'topolgy_dict.pickle')
    n_word = len(word_dict)
    n_top = len(sst_dict)
    
    dataset = list(zip(proteins, sst, enrichments)) #exempl the sst (proteins, sst,enrichments)
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_tmp = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_tmp, 0.5) 
    
    torch.manual_seed(1234)
    model = DeepLasso(n_word,n_top, dim, side, window, layer_cnn, layer_output).to(device)
    #model = torch.nn.DataParallel(model)
    trainer = Trainer(model,lr, weight_decay )
    tester = Tester(model)

    """Output files."""
    file_MAEs = './predicted/MAEs--' + setting + '.csv'
    file_model = './predicted/' + setting + '.pt'
    MAEs = ('Epoch\tTime(sec)\tRMSE_train\tR2_train\tMAE_dev\tMAE_test\tRMSE_dev\tRMSE_test\tR2_dev\tR2_test')
    with open(file_MAEs, 'w') as f:
        f.write(MAEs + '\n')

    """Start training."""
    print('Training...')
    print(MAEs)
    start = timeit.default_timer()

    for epoch in range(1, iteration+1):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train, rmse_train, r2_train = trainer.train(dataset_train)
        MAE_dev, RMSE_dev, R2_dev, Ydev, Preddev = tester.test(dataset_dev)
        MAE_test, RMSE_test, R2_test, Ytest, Predtest  = tester.test(dataset_test)
        name_dev = ['Ydev', 'Preddev']
        list_dev = []
        for i in zip(Ydev, Preddev):
            list_dev.append(i)
        name_test = ['Ytest', 'Predtest']
        list_test = []
        for a in zip(Ytest, Predtest):
            list_test.append(a)
        save_csv(name_dev, list_dev, 'valid_ss_iteration_50'+setting)
        save_csv(name_test, list_test, 'test_ss_iteration_50'+setting)

        end = timeit.default_timer()
        time = end - start

        MAEs = [epoch, time, rmse_train, r2_train, MAE_dev, MAE_test, RMSE_dev, RMSE_test, R2_dev, R2_test]
        tester.save_MAEs(MAEs, file_MAEs)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, MAEs)))
