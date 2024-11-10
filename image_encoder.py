import torch
from torch import nn 
from pos_embeds import PositionalEncoding
from transformer_encoder import Transformer

class ImageEncoder(nn.Module):
    """
    We will be using vision transformer when creating our image 
    encoder, we first need to make sure that the input images can 
    be split evently into patches of patch_sizea and that dimensionally
    of the model is divisible by the number of attention heads.

    Split the image into patches and create a sequence of linear
    embeddings of these patches by Conv2d method.

    """

    def __init__(self, width, image_size, patch_size, n_channels, n_layers, n_heads, emb_dim):
        super(ImageEncoder, self).__init__() 
        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, "image_size dimentions must match with patch_dimnetions"
        assert width % n_heads == 0, "width must divisible by n_heads"
        self.n_patches = (image_size[0] * image_size[1]) // (patch_size[0] * patch_size[1])
        self.max_sequence = self.n_patches + 1
        self.linear_projection = nn.Conv2d(
            n_channels, width, kernel_size =patch_size, stride = patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.positionaL_encods = PositionalEncoding(width, self.max_sequence)
        self.encoder = nn.ModuleList(
            [
                Transformer(width, n_heads)
                for _ in range(n_layers)
            ]
        )
        self.projection_layer = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, input):
        # Patch Embedding (B, C, H, W) -> (B, width, P_col, P_row)
        input = self.linear_projection(input) 
        # (B, width, P_col, P_row) -> (B, width, P) -> (B, P, width) 
        input = input.flatten(2).transpose(1, 2)

        # Positional Encoding 
        input = torch.cat((self.cls_token.expand(input.size()[0], -1, -1), input), dim = 1)
        input = self.positionaL_encods(input) 

        # Transformer Encoder
        for encoder_layer in self.encoder:
            input = encoder_layer(input) 

        # Getting class tokens 
        input = input[:, 0, :] 

        # Join multimodel embeddings
        if self.projection_layer is not None:
            input = input @ self.projection_layer 

        input = input / torch.norm(input, dim = -1, keepdim = True)

        return input 