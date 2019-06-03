"""
Code/Comments By Hrituraj Singh
Code reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html
June 2019
"""

import copy
import torch.nn as nn


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    """Mask out subsequent positions. Can be used in decoder for masking"""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class LayerNorm(nn.Module):
    """Construct a layernorm module (Reference: https://arxiv.org/abs/1607.06450"""
    def __init__(self, features, eps=1e-6):
    	"""Initializes the class

    	[Inputs]
    	features : Shape of the features/mapping
    	eps : small value for calculation stabilization
    	"""
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
    	"""Performs the layer normalization"""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm."""
    def __init__(self, size, dropout):
    	"""Initializes the class

    	[Inputs]
    	size : shape of the features map
    	dropout : dropout rate
    	"""

        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))