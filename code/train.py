#!/usr/bin/python
# coding: utf-8


import pickle
import sys
import timeit
import math
import numpy as np
import torch
from nn.attention import MultiheadAttention
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error,r2_score
#import scipy.stats as sps
import pandas as pd

def bn_conv2d(in_planes, out_planes, kernel_size, dilated):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, dilation=dilated,padding=(dilated * (kernel_size -1)+1)//2), nn.BatchNorm2d(out_planes), nn.ReLU())

class deeplasso(nn.Module):
    def __init__(self):
        super(deeplasso, self).__init__()
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_cnn = nn.ModuleList([bn_conv2d(in_planes=1, out_planes=1, kernel_size=2*window+1, dilated=1) for _ in range(layer_cnn)])
        #self.W_attention = nn.Linear(dim, dim)
        self.W_attention = MultiheadAttention(dim,dim,num_heads=5)
        self.W_out = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_output)])
        # self.W_interaction = nn.Linear(2*dim, 2)
        self.W_linear = nn.Linear(dim, 1)
    
    def attention_cnn(self, xs, layer):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        residuals = xs
        for i in range(layer):
            xs = residuals + self.W_cnn[i](xs)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        #h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        #weights = torch.tanh(F.linear(h, hs))
        #print(weights.shape, hs.shape)
        #ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(hs, 0), 0)

    def forward(self, inputs):

        words = inputs

        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(word_vectors, layer_cnn)

        for j in range(layer_output):
            output_vector = torch.relu(self.W_out[j](protein_vector))
        output = self.W_linear(output_vector)

        return output

    def __call__(self, data, train=True):

        inputs, correct_enrichment = data[0], data[-1]
        predicted_enrichment = self.forward(inputs)
        # print(predicted_interaction)

        if train:
            loss = F.mse_loss(predicted_enrichment, correct_enrichment)
            correct_values = correct_enrichment.to('cpu').data.numpy()
            predicted_values = predicted_enrichment.to('cpu').data.numpy()[0]
            return loss, correct_values, predicted_values
        else:
            correct_values = correct_enrichment.to('cpu').data.numpy()
            predicted_values = predicted_enrichment.to('cpu').data.numpy()[0]
            return correct_values, predicted_values


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        trainCorrect, trainPredict = [], []
        for data in dataset:
            loss, correct_values, predicted_values = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
            trainCorrect.append(correct_values)
            trainPredict.append(predicted_values)
        rmse_train = np.sqrt(mean_squared_error(trainCorrect,trainPredict))
        r2_train = r2_score(trainCorrect,trainPredict)
        return loss_total, rmse_train, r2_train


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        testY, testPredict = [], []
        for data in dataset :
            (correct_values, predicted_values) = self.model(data, train=False)
            SAE += np.abs(predicted_values-correct_values)
            testY.append(correct_values)
            testPredict.append(predicted_values)
        MAE = SAE / N  # mean absolute error.
        rmse = np.sqrt(mean_squared_error(testY,testPredict))
        r2 = r2_score(testY,testPredict)
        return MAE, rmse, r2, testY, testPredict

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

def save_csv(name:list, content:list, setting):
    tmp_df = pd.DataFrame(columns=name, data=content)
    tmp_df.to_csv(r'./output/{setting}_data.csv', encoding='gbk')

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    """Hyperparameters."""
    ngram=21
    dim=10
    side=5
    window=11
    layer_cnn=6
    layer_output=6
    lr=1e-3
    lr_decay = 0.5
    decay_interval=10
    weight_decay=1e-6
    iteration=200
    setting = 'train_preprocess_ngrams_3_ngram_21Multiheads_attention_denovo_10330'
    # print(type(radius))

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('./input/')
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'regression', torch.FloatTensor)
    word_dict = load_pickle(dir_input + 'sequence_dict.pickle')
    n_word = len(word_dict)

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(proteins, interactions))
    dataset = shuffle_dataset(dataset, 42)
    dataset_train, dataset_vt = split_dataset(dataset, 0.8)
    dataset_valid, dataset_test = split_dataset(dataset_vt, 0.5)

    """Set a model."""
    torch.manual_seed(1234)
    model = deeplasso().to(device)
    #model = torch.nn.DataParallel(model)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_MAEs = './output/MAEs--' + setting + '.txt'
    file_model = './output/' + setting
    MAEs = ('Epoch\tTime(sec)\tRMSE_train\tR2_train\tSpearman_R_train\tPearson_R_train\tMAE_dev\tMAE_test\tRMSE_dev\tRMSE_test\tR2_dev\tR2_test\tSpearman_R_dev\tSpearman_R_test')
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
        MAE_dev, RMSE_dev, R2_dev, Ydev, Preddev = tester.test(dataset_valid)
        MAE_test, RMSE_test, R2_test, Ytest, Predtest  = tester.test(dataset_test)
        name_dev = ['Ydev', 'Preddev']
        list_dev = []
        for i in zip(Ydev, Preddev):
            list_dev.append(i)
        name_test = ['Ytest', 'Predtest']
        list_test = []
        for a in zip(Ytest, Predtest):
            list_test.append(a)
        save_csv(name_dev, list_dev, 'dev')
        save_csv(name_test, list_test, 'test')

        end = timeit.default_timer()
        time = end - start

        MAEs = [epoch, time, rmse_train, r2_train, MAE_dev, MAE_test, RMSE_dev, RMSE_test, R2_dev, R2_test]
        tester.save_MAEs(MAEs, file_MAEs)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, MAEs)))
