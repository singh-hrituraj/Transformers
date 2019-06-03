"""
Code/Comments By Hrituraj Singh
Code reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html
June 2019
"""

class Embeddings(nn.Module):
	"""Base class for implementing embeddings"""
    def __init__(self, d_model, vocab):
    	"""Initialize the class

    	[Inputs]
    	d_model : No of dimensions in model
    	vocab : vocabulary size
    	"""
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
    	"""Converts the one-hot vocab vectors to correponding embeddings"""
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """Implements the positional encoding function as described in Vaswani et al."""
    def __init__(self, d_model, dropout, max_len=5000):
    	"""Initialize the class

    	[Inputs]
    	d_model : no of dimensions in model
    	dropout : dropout rate
    	max_len : maximum length of the sequence"""

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
    	"""Performs the feed forward for positional encodings"""
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)