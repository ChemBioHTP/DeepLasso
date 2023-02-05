#!/usr/bin/python
# coding: utf-8
import sys
import timeit
import math
import numpy as np
import torch
from layers.attention import MultiheadAttention
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error,r2_score
import scipy.stats as sps
import pandas as pd
from model import *

def bn_conv2d(in_planes, out_planes, kernel_size, dilate_size,  relu=True):
    layers = [nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, dilation=dilate_size, padding=(dilate_size * (kernel_size-1)+1)//2), nn.BatchNorm2d(out_planes)]
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes,size, dilated):
        super(ResBlock, self).__init__()
        self.inplanes = inplanes
        self.planes =  planes
        self.size = size
        self.dilated = dilated
        self.conv1 = bn_conv2d(inplanes, planes, size, dilated)
        self.conv2 = bn_conv2d(planes, planes, size, dilated,relu=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = F.relu(out)
        return out


class DeepLasso(nn.Module):
    def __init__(self,n_word, n_top, dim, side, window,layer_cnn, layer_output):
        super(DeepLasso, self).__init__()
        self.n_word = n_word
        self.n_top = n_top
        self.dim = dim
        self.side = side
        self.window = window
        self.layer_cnn = layer_cnn
        self.layer_output = layer_output
        self.embed_word = nn.Embedding(n_word, dim)
        self.embed_top = nn.Embedding(n_top, dim)
        self.W_cnn = nn.ModuleList([bn_conv2d(in_planes=1, out_planes=1, kernel_size=2*window+1, dilate_size=1) for _ in range(layer_cnn)])
        self.resblock1 = nn.ModuleList([ResBlock(1, 1, 5, 1) for _ in range(2)])
        self.resblock2 = ResBlock(1, 1, 5, 1)
        self.W_attention = MultiheadAttention(dim,dim,num_heads=10)
        self.W_out = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_output)])
        self.linear1 = nn.Linear(dim, dim)
        self.lstm = nn.LSTM(dim, 100, num_layers=2, bidirectional=True, batch_first=True)
        self.clsf_output = nn.Linear(dim, 2)
        self.regr_output = nn.Linear(dim, 1)
    
    def top_encoder(self, x, layer=3):
        x = x.reshape(1, 1, x.shape[0], x.shape[1])
        residuals = x
        for i in range(layer):
            x = residuals + self.W_cnn[i](x)
        t = x.reshape(x.shape[2], x.shape[3])
        return t

    def sequence_encoder(self, xs, layer):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        residuals = xs
        for i in range(layer):
            xs = residuals + self.W_cnn[i](xs)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        output, hidden_cells = self.lstm(xs)
        xs1 = output.reshape(-1, self.dim)
        hs = torch.relu(self.W_attention(xs1))
        return torch.unsqueeze(torch.mean(hs, 0), 0)

    def classifier(self, x, layer=2):
        x = x.reshape(1, 1, x.shape[0], x.shape[1])
        for i in range(layer):
            x = self.resblock1[i](x)
        res_out = x.reshape(x.shape[2], x.shape[3])
        hs = torch.relu(self.W_attention(res_out))
        hs = torch.unsqueeze(torch.mean(hs, 0), 0)
        outputs = self.linear1(hs)
        return outputs

    def regressor(self, x):
        x = x.reshape(1, 1, x.shape[0], x.shape[1])
        x = self.resblock2(x)
        res_out = x.reshape(x.shape[2], x.shape[3])
        return res_out

    def forward(self, inputs):
        words, tops  = inputs
        #words = inputs
        """Protein vector with CNN_BiLSTM_attention."""
        word_vectors = self.embed_word(words)
        protein_vector = self.sequence_encoder(word_vectors, self.layer_cnn)
        #cat_vector = protein_vector #exemple structures
        """Topology encoder with CNN"""
        tops_vectors = self.embed_top(tops)
        struc_vectors = self.top_encoder(tops_vectors)
        

        cat_vector = torch.cat((protein_vector, tops_vectors), 0)

        outputs1 = self.classifier(cat_vector)
        outputs = self.regressor(outputs1)

        """Concatenate the above two vectors and output the interaction."""
        #cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(self.layer_output):
            output_vector = torch.relu(self.W_out[j](outputs))
        output = self.regr_output(output_vector)

        return output.view(-1)

    def __call__(self, data, train=True):

        inputs, correct_enrichment = data[:-1], data[-1]
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

