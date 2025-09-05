import os
import sys


import torch
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
from src.data_preprocessing import index_to_label,transformation
import pandas as pd


def predict_class(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return index_to_label(predicted[0].item())


def class_infos(class_name):
    current_dir = os.path.dirname(__file__)  # points to src/
    csv_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'processed', 'nutrition_lookup_food101.csv'))

    lookup_table = pd.read_csv(csv_path)

    return {
        "Class": class_name,
        "Calories": lookup_table.loc[lookup_table['food'] == class_name, 'calories'].values[0],
        "Protein": lookup_table.loc[lookup_table['food'] == class_name, 'protein'].values[0],
        "carbohydrates": lookup_table.loc[lookup_table['food'] == class_name, 'carbs'].values[0],
        "fat": lookup_table.loc[lookup_table['food'] == class_name, 'fat'].values[0]
    }

def load_model_and_preprocess_image(image_path):
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(__file__) 
    model_path = os.path.abspath(os.path.join(current_dir, '..', 'models', 'model.pth'))



    model=resnet18(pretrained=True)
    model.fc=nn.Linear(512,101)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model=model.to(device)

    image = Image.open(image_path)
    trns=transformation()
    image=trns(image).unsqueeze(0)
    return image,model,device


def main():
    pass

if __name__ == "__main__":
    main()