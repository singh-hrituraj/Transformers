"""
Code/Comments By Hrituraj Singh
Code reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html
June 2019
"""

import copy
import torch.nn as nn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])