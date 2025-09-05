import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import pandas as pd
from src.predict import predict_class, class_infos, load_model_and_preprocess_image


def main():

    image_path = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if image_path is not None:
        image=st.image(image_path)
        
        image,model,device=load_model_and_preprocess_image(image_path)
        
        
        predicted_class = predict_class(model, image, device)
        st.write(f" predicted food is {predicted_class}")

        class_info = class_infos(predicted_class)
        st.write(f"Calories: {class_info['Calories']}")
        st.write(f"Protein: {class_info['Protein']}g")
        st.write(f"Carbohydrates: {class_info['carbohydrates']}g")
        st.write(f"Fat: {class_info['fat']}g")

        st.write("Note: all nutrients are measured in per 100 grams")
    
if __name__ == "__main__":
    main()