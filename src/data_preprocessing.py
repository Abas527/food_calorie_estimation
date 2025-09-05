import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import os
import json


image_dir='./data/raw/images'
train_file='./data/raw/meta/train.json'
test_file='./data/raw/meta/test.json'
labels_file='./data/raw/meta/labels.txt'
classes_file='./data/raw/meta/classes.txt'

def transformation():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def index_to_label(index):
    current_dir = os.path.dirname(__file__)
    classes_file = os.path.abspath(os.path.join(current_dir, '..', 'data', 'raw', 'meta', 'classes.txt'))
    with open(classes_file, 'r') as f:
        labels = f.read().splitlines()
    return labels[index]


def create_food101_dataframe(data_dir):

    data_dr=data_dir.replace('\\', '/')



    with open(classes_file, 'r') as f:
        classes = f.read().splitlines()
    cls_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    with open(train_file, 'r') as f:
        train_data = json.load(f)

    with open(test_file, 'r') as f:
        test_data = json.load(f)

    rows=[]
    for classes in train_data:
        image_list=train_data[classes]
        for image in image_list:
            image_path=os.path.join(data_dr, image+'.jpg')
            image_path=image_path.replace('\\', '/')
            rows.append({
                'image': image_path,
                'label': cls_to_idx[classes],
                'class_name': classes,
                'split': 'train'
            })
    
    rows_test=[]
    for classes in test_data:
        image_list=test_data[classes]
        for image in image_list:
            image_path=os.path.join(data_dr, image+'.jpg')
            image_path=image_path.replace('\\', '/')
            rows_test.append({
                'image': image_path,
                'label': cls_to_idx[classes],
                'class_name': classes,
                'split': 'test'
            })
    return pd.DataFrame(rows), pd.DataFrame(rows_test)
    

def main():
    df_train, df_test = create_food101_dataframe(image_dir)
    df_train.to_csv('./data/processed/food101_train.csv', index=False)
    df_test.to_csv('./data/processed/food101_test.csv', index=False)

if __name__ == "__main__":
    main()
    
