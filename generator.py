"""
Code/Comments By Hrituraj Singh
Code reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html
June 2019
"""

import torch.nn.Functional as F
import torch.nn as nn

class Generator(nn.Module):
	"""Performs standard linear projection followed by softmax"""

	def __init__(self, d_model, vocab):
		"""Initializes the class

		[Inputs]
		d_model : Number of dimensions being used in the model
		vocab : : vocabulary size

		"""
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		"""Performs generator step on input"""

		return F.log_softmax(self.proj(x), dim=-1)
	

