import torch 
import numpy as np
from torch import nn 
from image_encoder import ImageEncoder 
from text_encoder import TextEncoder

class CLIP(nn.Module):
    """
    Given a batch of images and captions, CLIP is supposed to tell 
    us which captions goes with which images. It does this by training
    the text and image encoder together to maxiize the pairwise
    xoisne similarity scores of the pairs that are supposed to go 
    together and minimizing the pairs that are not supposed to go together.
    
    In order to maximize the cosine similarity between related images,
    CLIP uses symmetric/contrastive loss.
    """


    def __init__(self, emb_dim, vit_width, image_size, 
                 patch_size, n_channels, vit_layers, 
                 vit_heads, vocab_size, text_width, max_sequence,
                 text_heads, text_layers): 
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder(
            vit_width, image_size, patch_size, n_channels,
            vit_layers, vit_heads, emb_dim
        )
        self.text_encoder = TextEncoder(
            vocab_size, text_width, max_sequence,
            text_heads, text_layers, emb_dim
        )
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    def forward(self,image,text, mask=None):
        I_e = self.image_encoder(image)
        T_e = self.text_encoder(text, mask=mask)

        # scaled pairwise cosine similarities [n, n]
        logits = (I_e @ T_e.transpose(-2, -1)) * torch.exp(self.temperature)

        # symmetric loss function
        labels = torch.arange(logits.shape[0]).to(self.device)
        # print(logits) 

        loss_i = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)
        loss_t = nn.functional.cross_entropy(logits, labels)

        loss = (loss_i + loss_t) / 2 
        # print(loss.item(), loss_i.item(), loss_t.item())

        return loss 