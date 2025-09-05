import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
from src.data_preprocessing import transformation

class Food101Dataset(Dataset):
    def __init__(self, csv_file, split='train', transform=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.img_paths = self.df['image']
        self.labels = self.df['label']
        self.transform = transformation()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)


def dataloader(csv_file, batch_size, split='train'):
    dataset = Food101Dataset(csv_file, split, transform=transformation())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

if __name__ == '__main__':
    train_dataloader = dataloader('./data/processed/food101_train.csv', 32, 'train')
    test_dataloader = dataloader('./data/processed/food101_test.csv', 32, 'test')

    print(len(train_dataloader))
    print(len(test_dataloader))
