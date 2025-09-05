import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_loader import dataloader
from torchvision.models import resnet18
import mlflow
import json

def evaluate(model, dataloader, device):
    with mlflow.start_run():
        mlflow.set_tag("model", "resnet18")
        mlflow.set_tag("dataset", "food101")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        mlflow.log_metric("accuracy", accuracy)
        return accuracy


def main():
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataloader = dataloader('./data/processed/food101_test.csv', batch_size, 'test')

    model=resnet18(pretrained=True)
    model.fc=nn.Linear(512,101)
    state_dict = torch.load('models/model.pth', map_location=device)
    model.load_state_dict(state_dict)
    model=model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    accuracy = evaluate(model, test_dataloader, device)
    
    with open('evaluation.json', 'w') as f:
        f.write(str(accuracy))


if __name__ == '__main__':
    main()