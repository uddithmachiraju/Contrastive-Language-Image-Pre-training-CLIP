import torch 
from torch import nn
from multi_head import MultiHeadAttention

class Transformer(nn.Module):
    """
    Transformer encoder is consists of two sub-layers:
    1 sub layer performs multi-head attention and the second
    sub layer contains a multi-layer perceptron. The multi head 
    attention sub-layer performs communication between tokens 
    while the multi layer perceptron sub-layer allows the tokens 
    to indually think on what was communicated to them.
    
    """


    def __init__(self, width, n_heads, r_mlp = 4):
        super(Transformer, self).__init__()
        self.width = width 
        self.n_heads = n_heads 

        self.norm_layer_1 = nn.LayerNorm(width)
        self.multi_head = MultiHeadAttention(width, n_heads) 
        self.norm_layer_2 = nn.LayerNorm(width) 

        self.mlp = nn.Sequential(
            nn.Linear(self.width, self.width * r_mlp),
            nn.GELU(),
            nn.Linear(self.width * r_mlp, self.width) 
        )

    def forward(self, input, mask = None):
        input = input + self.multi_head(self.norm_layer_1(input), mask = mask)
        input = input + self.mlp(self.norm_layer_2(input))
        return input