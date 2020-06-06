# -*- coding: utf-8 -*-
# @Auther   : liou

import math
import copy
import torch
from torch import nn
from .utils import LayerNorm
import torch.nn.functional as F
from torch.autograd import Variable
from .attention import MultiHeadedAttention
from .EncoderDecoder import EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer, Generator

class PositionwiseFeedForward (nn.Module) :
    """模型中的feed forward部分
    此处的feed forward为逐个位置的，即：
        attention部分针对每个位置都生成一个d_model大小的内容，所以总的输出为B * Length * d_model
        按位置代表对每一个x，即length上每一个单位，使用相同的变换矩阵

    就是两个全连接层和relu
    公式：
        FFN (x) = max (0, xW1 + b1) W2 + b2
        默认论文中中间隐藏层d_ff = 2048
    """
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear (d_model, d_ff)
        self.w_2 = nn.Linear (d_ff, d_model)
        self.dropout = nn.Dropout (dropout)

    def forward (self, x) :
        return self.w_2 (self.dropout (F.relu(self.w_1 (x))))

class Embedding (nn.Module) :
    """对词作embedding
    返回长度为d_model的embedding，并且乘了d_model^0.5
    """
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding (vocab, d_model)
        self.d_model = d_model

    def forward (self, x) :
        return self.lut (x) * math.sqrt(self.d_model)


class PositionalEncoding (nn.Module) :
    """位置编码
    具体内容见论文
    """
    def __init__(self, d_model, dropout, max_len = 500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout (p = dropout)

        pe = torch.zeros (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0 :: 2] = torch.sin (position * div_term)
        pe[:, 1 :: 2] = torch.cos (position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer ('pe', pe)

    def forward (self, x) :
        x = x + Variable (self.pe[:, : x.size(1)], requires_grad = False)
        return self.dropout (x)

def make_model (src_vocab, tgt_vocab, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1) :
    c = copy.deepcopy
    attn = MultiHeadedAttention (h, d_model)
    ff = PositionwiseFeedForward (d_model, d_ff, dropout)
    position = PositionalEncoding (d_model, dropout)
    model = EncoderDecoder (
        Encoder (EncoderLayer (d_model, c(attn), c(ff), dropout), N),
        Decoder (DecoderLayer (d_model, c(attn), c(ff), dropout), N),
        nn.Sequential (Embedding (d_model, src_vocab, c(position))),
        nn.Sequential (Embedding (d_model, tgt_vocab, c(position))),
        Generator (d_model, tgt_vocab)
    )

    for p in model.parameters () :
        if p.dim () > 1 :
            nn.init.xavier_uniform_(p)
    return model
