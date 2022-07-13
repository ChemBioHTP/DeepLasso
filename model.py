import logging1
from scipy.stats.stats import pearsonr
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#def conv1d(in_planes, out_planes, size):
 #   return nn.Conv1d(in_planes, out_planes, kernel_size=size, padding=size // 2, bias=False)


#def bn_conv1d(in_planes, out_planes, size, relu=True):
#    layers = [conv1d(in_planes, out_planes, size), nn.BatchNorm1d(out_planes)]
#    if relu:
#        layers.append(nn.ReLU())
#    return nn.Sequential(*layers)


#class ResBlockV1(nn.Module):
#    def __init__(self, inplanes, planes):
#    #    super(ResBlockV1, self).__init__()
#   #     self.inplanes = inplanes
#  #      self.planes = planes
# #       self.conv1 = bn_conv1d(inplanes, planes, 3)
#        self.conv2 = bn_conv1d(planes, planes, 3, relu=False)

#    def forward(self, x):
#        residual = x
#        out = self.conv1(x)
#        out = self.conv2(out)
#
#        if self.inplanes != self.planes:
#            residual = F.pad(x, (0, 0, 0, self.planes - self.inplanes), "constant", 0)
#
#        out += residual
#        out = F.relu(out)
#
#        return out



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb1 = nn.Embedding(30, 30)
        self.emb1.weight.data = torch.eye(30)
        self.emb1.weight.requires_grad = False
        self.emb2 = nn.Embedding(20, 20)
        self.emb2.weight.data = torch.eye(20)
        self.emb2.weight.requires_grad = False

        self.inplanes = 30 + 20
       # self.msa_block = self._make_layers([96, 128], [8, 0], ResBlockV1)
       # self.ss_block = self._make_layers([96, 128, 256, 512], [16, 0, 0, 1], ResBlockV1)
        self.cn1 = nn.Conv1d(self.inplanes, 32, 129)
        self.cn2 = nn.Conv1d(self.inplanes, 32, 257)
        self.lstm = nn.LSTM(114, 1024, batch_first=True, bidirectional=True)
       # self.lstm = nn.LSTM(114, 1024, batch_first=True, bidirectional=True)
       # self.fc2 = nn.Conv1d(self.inplanes, 512, kernel_size=1, bias=False) #full contact
        self.fc3 = nn.Conv1d(2048, 20, kernel_size=1,  bias=True) #full contact
       # self.fc4 = nn.Conv1d(2048, 5, kernel_size=1, bias=True) #full contact
       # self.lstm = nn.LSTM(114, 1024, batch_first=True) 
       # self.lstm = nn.LSTM(114, 1024, batch_first=True)


#    def _make_layers(self, channels, repeats, block):
#        layers = []
#        for c, r in zip(channels, repeats):
#            for _ in range(r):
#                layers.append(block(self.inplanes, c))
#                self.inplanes = c
       # return nn.Sequential(*layers)

    def forward(self, data):
        x = data['pssm']
        x = self.emb1(x)
        x = x.permute(0, 3, 1, 2)
        seq = self.emb2(data['seq'])
        seq = seq.permute(0, 2, 1)
        seq = torch.cat([seq, data['mask1d'].float()], dim=1)

        k = x.shape[2]
        C, L = seq.shape[1], seq.shape[2]
        seq = seq.repeat(1, k, 1).reshape(-1, k, C, L)
        seq = seq.transpose(1, 2)
        x = torch.cat([x, seq, data['mask2d'].float()], dim=1)

        x = x.transpose(1, 2)
        in_shape = x.shape
        x = x.contiguous().view(-1, in_shape[2], in_shape[3])
       #x = self.mse_block(x)
        a = self.cn1(x)
        b = self.cn2(x)
       # x = x.contiguous().view(in_shape[0], in_shape[1], -1, in_shape[3])

       # x = x.permute(0, 2, 3, 1)
       # x, pool_map = torch.max(x, dim=3)
        x = a + b + self.inplanes
        x = self.lstm(x)
        x = self.lstm(x)
       # x = F.relu(self.fc2(x))
       # y1 = F.relu(self.fc3(x))
        y2 = self.fc4(x)
       # y1 = y1.permute(0, 2, 1).contiguous()
        y2 = y2.permute(0, 2, 1).contiguous()
        outputs = {
            'ss3': y1[:, :, :3],
            'ss8': y1[:, :, 3:11],
            'rsa': y2[:, :, 0],
           # 'sin_phi': y2[:, :, 1],
           # 'cos_phi': y2[:, :, 2],
           # 'sin_psi': y2[:, :, 3],
           # 'cos_psi': y2[:, :, 4],
           # 'pool_map': pool_map
        }
        return outputs

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Trainable parameters: {}'.format(params))
        logging.info(self)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(nn.Module, self).__str__() + '\nTrainable parameters: {}'.format(params)


def ss3_cross_entropy_loss(output, target):
    output = output.view(-1, 3)
    target = target.view(-1)
    return F.cross_entropy(output[target >= 0], target[target >= 0])


def ss8_cross_entropy_loss(output, target):
    output = output.view(-1, 8)
    target = target.view(-1)
    return F.cross_entropy(output[target >= 0], target[target >= 0])

"""
def l2_loss(output, target, key):
    mask = target['mask'].view(-1)
    output = output[key].view(-1)
    target = target[key].view(-1)
    return F.mse_loss(output[mask > 0], target[mask > 0])
"""

def rsa_loss(output, target):
    output = output['rsa'].view(-1)
    target = target['rsa'].view(-1)
    return F.mse_loss(output[target != -1], target[target != -1])

"""
def angle_loss(output, target, key):
    true = target[key].view(-1)
    sin = output['sin_' + key].view(-1)[true != 360]
    cos = output['cos_' + key].view(-1)[true != 360]
    true = true[true != 360]

    return F.mse_loss(sin, torch.sin(true * math.pi / 180.0)), F.mse_loss(cos, torch.cos(true * math.pi / 180))
"""

def gen_loss(output, target):
    ssl, csl = angle_loss(output, target, 'psi')
    shl, chl = angle_loss(output, target, 'phi')
    return {
        'ss3_loss': ss3_cross_entropy_loss(output['ss3'], target['ss3']),
        'ss8_loss': ss8_cross_entropy_loss(output['ss8'], target['ss8']),
        'rsa_loss': rsa_loss(output, target) * 20,
        'asa_loss': l2_loss(output, target, 'asa'),
       # 'sin_psi_loss': ssl * 5,
       # 'cos_psi_loss': csl * 5,
       # 'sin_phi_loss': shl * 5,
       # 'cos_phi_loss': chl * 5,
    }


def ss3_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output.view(-1, 3), dim=1)
        target = target.view(-1)
        correct = torch.sum((pred == target) & (target >= 0)).item()
        cnt = torch.sum(target >= 0).item()
    return correct / cnt

def ss8_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output.view(-1, 8), dim=1)
        target = target.view(-1)
        correct = torch.sum((pred == target) & (target >= 0)).item()
        cnt = torch.sum(target >= 0).item()
    return correct / cnt

"""
def angle_metric(output, target, key):
    with torch.no_grad():
        true = target[key].view(-1)
        sin_k = output['sin_' + key].view(-1)[true != 360]
        cos_k = output['cos_' + key].view(-1)[true != 360]
        true = true[true != 360]
        angle = torch.atan2(sin_k, cos_k) * 180 / math.pi
        d = torch.abs(true - angle)
        d = torch.min(d, 360 - d)
        return torch.mean(d).item()
"""


def asa_metric(output, target):
    with torch.no_grad():
        masa = target['masa'].view(-1)
        pred = output['rsa'].view(-1)[masa != -1].cpu().numpy()
        asa = target['asa'].view(-1)[masa != -1].cpu().numpy()
        masa = masa[masa != -1].cpu().numpy()
        return pearsonr(pred * masa, asa)[0]


def gen_metric(output, target):
    return {
        'ss3_acc': ss3_metric(output['ss3'], target['ss3']),
        'ss8_acc': ss8_metric(output['ss8'], target['ss8']),
        'asa_pearson': asa_metric(output, target),
       # 'phi_MAE': angle_metric(output, target, 'phi'),
       # 'psi_MAE': angle_metric(output, target, 'psi')
    }

    
