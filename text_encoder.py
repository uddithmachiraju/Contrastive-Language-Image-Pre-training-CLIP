import torch 
from torch import nn
from pos_embeds import PositionalEncoding
from transformer_encoder import Transformer

def tokenizer(text, encode = True, mask = None, max_sequence = 32):
    """
    Transformers are unable to process raw text, so the first thing is to do
    is tokenize the input strings before passing them through the transformers.
    
    """
    
    if encode:
        # Adding SOT and EOT tokens
        out = chr(2) + text + chr(3) 
        # Adding padding
        out = out + "".join(chr(0) for _ in range(max_sequence - len(out)))
        # Encode the text
        out = torch.IntTensor(list(out.encode("utf-8")))
        mask = torch.ones(len(out.nonzero())) 
        mask = torch.cat((mask, torch.zeros(max_sequence - len(mask)))).type(torch.IntTensor)
    else:
        out = [chr(x) for x in text[1:len(mask.nonzero()) - 1]] 
        out = "".join(out)
        mask = None 

    return out, mask

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, width, max_sequence, n_heads, n_layers, emb_dim):
        super(TextEncoder, self).__init__()
        self.max_sequence = max_sequence 
        self.enocoder_embeds = nn.Embedding(vocab_size, width)
        self.positional_encoding = PositionalEncoding(width, max_sequence)
        self.encoder = nn.ModuleList(
            [
                Transformer(width, n_heads)
                for _ in range(n_layers)
            ]
        )

        self.projection_layer = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, text, mask = None):
        text = self.enocoder_embeds(text) 
        text = self.positional_encoding(text) 
        for encoder_layer in self.encoder:
            text = encoder_layer(text, mask = mask) 
        text = text[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:, 0], dim = 1), 1)] 

        if self.projection_layer is not None:
            text = text @ self.projection_layer
        
        text = text / torch.norm(text, dim = -1, keepdim = True)

        return text 