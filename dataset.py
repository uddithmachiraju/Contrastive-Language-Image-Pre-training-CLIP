import torch 
from datasets import load_dataset
from text_encoder import tokenizer
import torchvision.transforms as transform
from torch.utils.data import Dataset

class FashionMNIST(Dataset):
    def __init__(self, train = True):
        super().__init__() 
        self.dataset = load_dataset('fashion_mnist')
        self.transform = transform.ToTensor()
        if train: self.split = 'train'
        else: self.split = 'test' 

        self.captions = {
            0 : 'An image of a T-Shirt/Top',
            1 : 'An image of a Trousers',
            2 : 'An image of a Pullover',
            3 : 'An image of a Dress',
            4 : 'An image of a Coat',
            5 : 'An image of a Sandal',
            6 : 'An image of a Shirt',
            7 : 'An image os a Sneaker',
            8 : 'An image of a Bag',
            9 : 'An image of a Ankle Boot'
        }
    def __len__(self):
        return self.dataset.num_rows[self.split] 
    
    def __getitem__(self, index):
        image = self.dataset[self.split][index]['image'] 
        image = self.transform(image) 

        caption, mask = tokenizer(
            self.captions[self.dataset[self.split][index]['label']]
        )

        mask = mask.repeat(len(mask), 1)

        return {
            'image' : image ,
            'caption' : caption,
            'mask' : mask
        }