import torch 
from model import CLIP 
from config import parameters
import matplotlib.pyplot as plt 
from dataset import FashionMNIST
from text_encoder import tokenizer
from torch.utils.data import DataLoader

test_set = FashionMNIST(train = False)
test_dataloader = DataLoader(test_set, shuffle = True, batch_size = parameters['batch_size']) 
device = 'cpu'
# print(len(test_set))

# Loading Best Model
test_model = CLIP(
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

test_model.load_state_dict(torch.load("CLIP.pt", map_location=device))


# Captions to compare images to
class_names =[
   "t-shirt/top",
    "trousers",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot"
]

text = torch.stack([tokenizer(x)[0] for x in class_names]).to(device)
mask = torch.stack([tokenizer(x)[1] for x in class_names])
mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)

idx = torch.randint(0, len(test_set), size = [1])
# print(idx.item()

img = test_set[idx.item()]["image"][None,:]
plt.imshow(img[0].permute(1, 2, 0)  ,cmap="gray")
plt.title(tokenizer(test_set[idx.item()]["caption"], encode=False, mask=test_set[idx.item()]["mask"][0])[0])
plt.show()

img = img.to(device)
with torch.no_grad():
  image_features = test_model.image_encoder(img)
  text_features = test_model.text_encoder(text, mask=mask)


image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{class_names[int(index)]:>16s}: {100 * value.item():.2f}%")