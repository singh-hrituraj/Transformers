"""
Code/Comments By Hrituraj Singh
Code reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html
June 2019
"""

import torch.nn as nn
import torch.nn.Functional as F

class PositionWiseFeedForward(nn.Module):
	"""Performs the position wise feed forward as described in Vaswani et al."""

	def __init__(self, d_model, d_ff, dropout=0.1):
		"""Initialize the class

		[Inputs]
		d_model : No of dimensions in model
		d_ff : no of hidden layer neurons in feed forward
		dropout : dropout rate"""
		super(PositionWiseFeedForward, self).__init__()
		self.w_1 = nn.linear(d_model, d_ff)
		self.w_2 = nn.linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		"""Performs the feed forward"""

		return self.w_2(self.dropout(F.relu(self.w_1(x))))