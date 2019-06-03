"""
Code/Comments By Hrituraj Singh
Code reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html
June 2019
"""

import torch.nn as nn
from Model import *
from Attention import MultiHeadedAttention
from 

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters.

    [Inputs]
    src_vocab : no of words in source vocabulary
    tgt_vocab : no of words in target vocabulary
    N : no of layers in the model
    d_model : no of dimensions in model
    d_ff : no of hidden layer dimensions in feed forward part of encoder layer
    h : no of heads to be used in attention
    dropout : dropout rate

    [Outputs]
    model : returns the whole transformer model with given configuration"""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model