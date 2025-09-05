import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_loader import dataloader
from torchvision.models import resnet18
import mlflow
import json


def train_model(model, train_dataloader, num_epochs, learning_rate, device,criterion, optimizer,batch_size):
    with mlflow.start_run():
        mlflow.set_tag("model", "resnet18")
        mlflow.set_tag("dataset", "food101")

        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)





        model.train()
        for epoch in range(num_epochs):
            running_loss,correct,total = 0.0,0,0
            for i, data in enumerate(train_dataloader, 0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss=criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
            mlflow.log_metric("loss", running_loss/len(train_dataloader))
            mlflow.log_metric("accuracy", correct/total)

            metrics={
                "loss":running_loss/total,
                "accuracy":correct/total
            }
            with open('metrics.json', 'w') as f:
                json.dump(metrics, f)
    
        torch.save(model.state_dict(), 'models/model.pth')
        mlflow.pytorch.log_model(model, "model")
    



def main():
    #defining the parameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataloader = dataloader('./data/processed/food101_train.csv', batch_size, 'train')
    test_dataloader = dataloader('./data/processed/food101_test.csv', batch_size, 'test')
    model = resnet18(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.fc=nn.Linear(512,101)
    model=model.to(device)
    train_model(model, train_dataloader, num_epochs, learning_rate, device, criterion, optimizer,batch_size)


if __name__ == "__main__":
    main()

