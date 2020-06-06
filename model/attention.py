# -*- coding: utf-8 -*-
# @Auther   : liou

import math
import torch
from torch import nn
import torch.nn.functional as F
from .utils import clones

def attention (query, key, value, mask = None, dropout = None) :
    """attention模型
    输出为：softmax (q * k / d_k^0.5) * v
    d_k为词向量维度。

    这个公式与乘性attention计算方式的唯一不同就在于使用了一个缩放因子d_k^0.5

    论文中对缩放的解释：
        在d_k比较小的时候，不加缩放的效果和加性attention效果差不多。
        在d_k比较大的时候，不加缩放效果明显更差。
    怀疑是当d_k增长的时候，内积的量级也会增长，导致softmax函数会被推向梯度较小的区域。
    """
    d_k = query.size (-1)
    scores = torch.matmul(query, key.transpose (-2, -1)) / math.sqrt(d_k)

    if mask is not None :
        # 将为0的地方替换为-1e9
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None :
        p_attn = dropout (p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention (nn.Module) :
    def __init__(self, h, d_model, dropout = 0.1):
        super(MultiHeadedAttention, self).__init__()
        # 默认八个头即h=8，序列长度为64，所以隐藏层即d_model长度为512
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear (d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout (p = dropout)

    def forward (self, query, key, value, mask = None) :
        if mask is not None :
            mask = mask.unsqueeze (1)
        nbatches = query.size (0)

        # 此处zip返回项数最少的，linears有四层而qkv只有三个，所以只返回三个值，即为qkv的linears
        query, key, value = [l(x).view(nbatches,
                                       -1,
                                       self.h,
                                       self.d_k).transpose (1, 2)
                             for l, x in zip (self.linears, (query, key, value))]
        # 多头attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.traspose (1, 2).contiguout ().view (nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)