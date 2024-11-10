import torch 
from model import CLIP 
from config import parameters 
from dataset import FashionMNIST
from text_encoder import tokenizer
from torch.utils.data import DataLoader

test_set = FashionMNIST(train = False)
test_dataloader = DataLoader(test_set, shuffle = True, batch_size = parameters['batch_size']) 
device = 'cpu'

eval_model = CLIP(
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

eval_model.load_state_dict(torch.load('CLIP.pt', map_location = device)) 
text = torch.stack(
    [
        tokenizer(x)[0] 
        for x in test_set.captions.values()
    ]
).to(device) 
mask = torch.stack(
    [
        tokenizer(x)[1]
        for x in test_set.captions.values()
    ]
)
mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)

correct, total = 0, 0 
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data['image'].to(device), data['caption'].to(device)
        image_features = eval_model.image_encoder(images) 
        text_features = eval_model.text_encoder(text, mask = mask) 

        image_features /= image_features.norm(dim = -1, keepdim = True) 
        text_features /= text_features.norm(dim = -1, keepdim = True) 
        similarity = (100.0 * image_features @ text_features.T).softmax(dim = -1)
        _, indices = torch.max(similarity, 1) 
        pred = torch.stack(
            [
                tokenizer(test_set.captions[int(i)])[0]
                for i in indices
            ]
        ).to(device) 
        correct += int(sum(torch.sum((pred==labels),dim=1)//len(pred[0])))
        total += len(labels)

print(f'Model Accuracy: {100 * correct // total} %')