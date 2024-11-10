import torch 
from torch import nn 

class AttentionHead(nn.Module):
    """
    Transformers use attention which is a communication mechanism 
    that allows the model to focus on important parts of the image.
    
    It consists of 
        1. Query - What the token is looking for?
        2. Key - What the token contains?
        3. Value - What is communicated between tokens? 

    Attention mask is required to decoders to avoid seeing into the
    next token. Since CLIP is a encoder only model we need attention 
    due to the padding that is applied to the input text during 
    tokenization.
    """


    def __init__(self, width, head_size):
        super(AttentionHead, self).__init__()
        self.head_size = head_size 

        self.query = nn.Linear(width, head_size)
        self.key = nn.Linear(width, head_size)
        self.value = nn.Linear(width, head_size)

    def forward(self, input, mask = None):
        Q = self.query(input) 
        K = self.key(input)
        V = self.value(input)

        # Dot product of Querys and keys
        attention = Q @ K.transpose(-2, -1)
        
        # Scalling attention
        attention = attention / (self.head_size ** 0.5) 

        # Apply mask for decoder
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))
        
        # Softmax 
        attention = torch.softmax(attention, dim = -1)

        attention = attention @ V 

        return attention 
    
class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention is just running multiple heads of 
    single head attention in parallel and combining them. 
    
    """


    def __init__(self, width, n_heads):
        super(MultiHeadAttention, self).__init__() 
        self.head_size = width // n_heads 
        self.output_layer = nn.Linear(width, width) 
        self.attention_heads = nn.ModuleList(
            [
                AttentionHead(width, self.head_size) 
                for _ in range(n_heads)
            ]
        )

    def forward(self, input, mask = None):
        output = torch.cat(
            [head(input, mask) for head in self.attention_heads],
            dim = -1
        )

        output = self.output_layer(output)

        return output
