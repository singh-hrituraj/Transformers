"""
Code/Comments By Hrituraj Singh
Code reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html
June 2019
"""

import torch.nn as nn
from utils import *

class Decoder(nn.Module):
	"""Base class for generic Decoder"""

	def __init__(self, layer, N):
		"""Initializes the class

		[Inputs]
		layer : the core decoder layer
		N : Number of core layers to be incorporated in whole decoder architecture
		"""

		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, memory, src_mask, tgt_mask):
		"""Performs the forward operation on input

		[Inputs]
		x : tgt input for the encoder
		memory : the output of encoder - encoded representation of input/source
		src_mask : mask to be applied over source
		tgt_mask : mask to be applied over target
		"""

		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)

		return self.norm(x)


class DecoderLayer(nn.Module):
	"""Base class for decoder layer"""

	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		"""Initializes the class

		[Inputs]
    	size : size of the output of layer [FINISH]
    	self_attn : self attention performing function/object
    	src_attn : Applying attention on the source input
    	feed_forward : feed forwarding function/object
    	droput : dropout rate
    	"""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

	def forward(self, x, memory, src_mask, tgt_mask):
        """Feed forward operations of decoder layer

        [Inputs]
        x : input target sequence
		memory : the output of encoder - encoded representation of input/source
		src_mask : mask to be applied over source
		tgt_mask : mask to be applied over target"""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
        
