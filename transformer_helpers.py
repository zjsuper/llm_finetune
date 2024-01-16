import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import numpy as np
import torch.nn.functional as F
import copy

def clones(module, n = 1):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size,d_model)
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)
    

def subsequent_mask(size):
    # return a size * size masked matrix?
    attention_size = (1, size, size)
    subseq_mask = np.triu(np.ones(attention_size), k = 1)
    return torch.from_numpy(1 - subseq_mask)
    