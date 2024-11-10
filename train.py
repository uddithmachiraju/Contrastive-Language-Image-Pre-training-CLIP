import torch 
import numpy as np
from config import * 
from tqdm import tqdm 
from model import CLIP 
from dataset import FashionMNIST
from torch.utils.data import DataLoader

train_set = FashionMNIST(train = True)
train_dataloader = DataLoader(train_set, shuffle = True, batch_size = parameters['batch_size']) 

device = 'cpu' 
model = CLIP(
    parameters['emb_dim'], 
    parameters['vit_width'],
    parameters['image_size'],
    parameters['patch_size'],
    parameters['n_channels'],
    parameters['vit_layers'],
    parameters['vit_heads'],
    parameters['vocab_size'],
    parameters['text_width'],
    parameters['max_sequence'],
    parameters['text_heads'],
    parameters['text_layers']
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = parameters['lr']) 

best_loss = np.inf
for epoch in range(parameters['epochs']):
    for index, data in enumerate(tqdm(train_dataloader), 0):
        image, caption, mask = data['image'].to(device), data['caption'].to(device), data['mask'].to(device)
        loss = model(image, caption, mask) 
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 

    if epoch % 10 == 0:
        print(f"Epoch [{epoch + 1} / {parameters['epochs']}, Batch Loss : {loss.item():.3f}")
    
    if loss.item() <= best_loss:
        best_loss = loss.item() 
        torch.save(model.state_dict(), 'CLIP.pt') 
        print(f'Model Saved at {loss.item()}') 