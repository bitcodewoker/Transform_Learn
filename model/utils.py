# -*- coding: utf-8 -*-
# @Auther   : liou

import torch
from torch import nn
import copy
import numpy as np
from torch.autograd import Variable

def clones (module, N) :
    """复制module n次"""
    return nn.ModuleList ([copy.deepcopy(module) for _ in range (N)])

def subsequent_mask (size) :
    """生成decoder的下三角mask
    mask：
        在encoder中主要为了让一些batch中较短的序列的padding不参与attention
        在decoder中主要为了避免数据泄漏
    所以src直接mask掉pad就可以，而trg不仅要去掉pad，还要mask掉后面的内容，也就是用本函数选取下三角矩阵
    """
    attn_shape = (1, size, size)
    # triu返回上三角矩阵内容，下三角为0
    subsequent_mask = np.triu (np.ones (attn_shape), k = 1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class LayerNorm (nn.Module) :
    """layernorm"""
    def __init__(self, features, eps = 1e-16):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter (torch.ones (features))
        self.b_2 = nn.Parameter (torch.zero_(features))
        self.eps = eps

    def forward (self, x) :
        mean = x.mean (-1, keepdim = True)
        std = x.std (-1, keepdim = True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection (nn.Module) :
    """残差网络
    对指定网络输出作norm、dropout和残差相加

    forward：
        x：输入
        sublayer：网络
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm (size)
        self.dropout = nn.Dropout (dropout)

    def forward (self, x, sublayer) :
        return x + self.dropout (sublayer (self.norm (x)))

class Batch:
    def __init__(self, src, trg = None, pad = 0):
        self.src = src
        self.src_mask = (src != pad).unsqeeeze (-2)
        if trg is not None :
            self.trg = trg[:, : -1]
            self.trg_y = trg[:, 1 :]
            self.trg_mask = self.make_std_mask (self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum ()

    @staticmethod
    def make_std_mask (tgt, pad) :
        tgt_mask = (tgt != pad).unsqueeze (-2)
        tgt_mask = tgt_mask & Variable (subsequent_mask(tgt.size (-1)).type_as(tgt_mask.data))
        return tgt_mask

