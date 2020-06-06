# -*- coding: utf-8 -*-
# @Auther   : liou

from torch import nn
import torch.nn.functional as F
from .utils import clones, LayerNorm, SublayerConnection

class EncoderDecoder (nn.Module) :
    """encoder and decoder"""
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward (self, src, tgt, src_mask, tgt_mask) :
        return self.decoder (self.encoder (src, src_mask),
                             src_mask,
                             tgt,
                             tgt_mask)

    def encode (self, src, src_mask) :
        return self.encoder (self.src_embed (src),
                             src_mask)

    def decode (self, memory, src_mask, tgt, tgt_mask) :
        return self.decoder (self.tgt_embed (tgt),
                             memory,
                             src_mask,
                             tgt_mask)

class Generator (nn.Module) :
    """线性分类器加softmax"""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear (d_model, vocab)

    def forward (self, x) :
        return F.log_softmax(self.proj (x), dim = -1)

class Encoder (nn.Module) :
    """Encoder with layernorm"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layer = clones(layer, N)
        self.norm = LayerNorm (layer.size)

    def forward (self, x, mask) :
        for layer in self.layer:
            x = layer (x, mask)
        return self.norm (x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layer = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm

class EncoderLayer (nn.Module) :
    """transformer 左边的encoder的单个层
    包括两层：
        attention后先norm再dropout，最后和残差相加
        feed forward后县norm在dropout，最后和残差相加
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward  = feed_forward
        self.sublayer = clones(SublayerConnection (size, dropout), 2)
        self.size = size

    def forward (self, x, mask) :
        x = self.sublayer[0](x, lambda x: self.self_attn (x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer (nn.Module):
    """decoder的单个层
    包括三层:
        对x的attention
        用decoder输出作为query去查询encoder的输出
        feed forward
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self._attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones (SublayerConnection (size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask) :
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn (x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.self.attn (x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

