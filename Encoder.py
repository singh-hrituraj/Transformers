"""
Code/Comments By Hrituraj Singh
Code reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html
June 2019
"""

from utils import *
import torch.nn as nn

class Encoder(nn.Module):
	"""Base class for the encoder architecture

	It is nothing but a stack of N core encoder layers"""

	def __init__(self, layer, N):
		"""Initializes the class

		[Inputs]
		layer : the core encoder layer
		N : Number of core layers to be incorporated in whole encoder architecture
		"""
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, mask):
		"""Performs the forward operation on input

		[Inputs]
		x : src input for the encoder
		mask : mask to be applied over source
		"""
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)





class EncoderLayer(nn.Module):
    """Encoder layer is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
    	"""Initializes the class

    	[Inputs]
    	size : size of the output of layer [FINISH]
    	self_attn : self attention performing function/object
    	feed_forward : feed forwarding function/object
    	dropout : dropout rate
    	"""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Implementation of the layer as described by Vaswani et al."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
