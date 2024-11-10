import torch 
import numpy as np 
from torch import nn 

class PositionalEncoding(nn.Module):
    """
    Unlike models like LSTM's which take embeddings in sequentially,
    transformers take the embeddings in parallel. Transformers are 
    not aware of what order the sequenences are supposed t be in. This 
    will be a problem because changing the order of the sequnece will
    alter it's meaning. So, positional encoding need to be added to the 
    embeddings. Each positional encodinng is unique with it's position
    that it represents which allows model to identify which position
    each embed should go in.
    
    """

    def __init__(self, width, max_sequence):
        super(PositionalEncoding, self).__init__()
        positional_encode = torch.zeros(max_sequence, width)

        for position in range(max_sequence):
            for i in range(width):
                if i % 2 == 0:
                    positional_encode[position][i] = np.sin(position / 10000 ** (i / width)) 
                else:
                    positional_encode[position][i] = np.cos(position / 10000 ** ((i - 1) / width))
            
            self.register_buffer('positional_encode', positional_encode.unsqueeze(0)) 

    def forward(self, input):
        return input + self.positional_encode