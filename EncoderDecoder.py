"""
Code/Comments By Hrituraj Singh
Code reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html
June 2019
"""

import torch.nn as nn

class EncoderDecoder(nn.Module):
	"""Base class for standard encoder-decoder architecture."""

	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		""" Initializer for the class

		[Inputs]
		encoder : An instantiation of encoder class - For eg. An RNN for RNN Encoder
		decoder : An instantiation of decoder class - For eg. An RNN for RNN Decoder
		src_embed : Embedding layer for the source vocabulary
		tgt_embed : Embedding layer for the target vocabulary
		generator : 
		"""
		super(EncoderDecoder, self).__init__():
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator

	def forward(self, src, tgt, src_mask, tgt_mask):
		"""Performs a forward pass on the input

		[Inputs]
		src : Source Sentence of shape [FINISH]
		tgt : Target Sentence of shape [FINISH]
		src_mask : Mask over the source tokens [FINISH]
		tgt_mask : Mask over the target tokens [FINISH]
		"""
		return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

	def encode(self, src, src_mask):
		"""Performs the encoding function

		[Inputs]
		src : Source Sentence of shape [FINISH]
		tgt : Target Sentence of shape [FINISH]
		"""

		return self.encoder(self.src_embed(src), src_mask)

	def decode(self, memory, src_mask, tgt, tgt_mask):
		"""Performs the decoding function

		[Inputs]
		memory : the output of encoder - encoded representation of input/source
		tgt : Target Sentence of shape [FINISH]
		src_mask : Mask over the source tokens [FINISH]
		tgt_mask : Mask over the target tokens [FINISH]
		"""

		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)








