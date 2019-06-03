"""
Code/Comments By Hrituraj Singh
Code reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html
June 2019
"""
import math, copy
import torch
import torch.nn as nn
import torch.nn.Functional as F


def attention(query, key, value, mask=None, dropout=None):
	"""Application of generalised attention

	[Inputs]
	query : standard query matrix of size:(None, no_query, head_dim)
	key : standary key matrix of size : (None, no_keys, head_dim
	values  : standardn value matrix of size : (None, no_keys=no_values, model_dim)
	mask : mask matrix of shape (None, no_query, no_keys)
	dropout : dropout rate

	[Output]
	context_vectors : context results after attention of size : (None, no_query, model_dim)
	p_attn : matrix of attention probabilities to help in visualisation of size : (None, no_query, no_keys)"""

	d_k = query.size(-1)
	scores = torch.matmul(quer, key.transpose(-2,-1)) / math.sqrt(d_k)

	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)

	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)

	return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
	"""Implementation of multi-headed attention as described in Vaswani et al."""

	def __init__(self, h, d_model, dropout=0.1):
		"""Initialize the class

		[Inputs]
		h : No of heads to be used
		d_model : No of model dimensions
		dropout : dropout rate"""
		super(MultiHeadedAttention, self).__init__()
		assert d_model%h==0

		d_k = d_model//h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.dropout(p=dropout)

	def forward(self, query, key, value, mask=None):
		"""Forward pass for the module

		[Inputs]
		query : standard query matrix of size:(None, no_query, head_dim)
		key : standary key matrix of size : (None, no_keys, head_dim
		values  : standardn value matrix of size : (None, no_keys=no_values, model_dim)
		mask : mask matrix of shape (None, no_query, no_keys)

		[Outputs]
		context_vector : Output context vector after applying the attention of shape : (None, model_dim)"""

		if mask is not None:
			mask = mask.unsqueeze(1)

		n_batches = query.size(0)
		query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) 
							for l, x in zip(self.linears, (query, key, value))]


		x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
		x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)

