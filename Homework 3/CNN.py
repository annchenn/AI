import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        # (TODO) Design your CNN, it can only be less than 3 convolution layers
        super(CNN, self).__init__()
        #in_channel, out_channel, kernel_size, padding
        self.conv1 = nn.Conv2d(3,32,3,1)
        self.bn1 = nn.BatchNorm2d(32)#normalize data
        self.relu = nn.ReLU()
        #kernel_size
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32,64,3,1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128,3,1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128*26*26, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # (TODO) Forward the model
        #conv1,2,3
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        #flatten
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    # (TODO) Train the model and return the average loss of the data, we suggest use tqdm to know the progress
    model.train()
    
    all_loss=0.0
    n=0
    
    for images, labels in tqdm(train_loader, desc='training'):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()#清空梯度
        
        #forward pass:預測結果並計算loss
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()#計算loss的梯度
        optimizer.step()#根據梯度更新模型參數
        
        num = images.size(0)
        all_loss += loss.item()*num #.item return scalar
        n+=num
    
    avg_loss = all_loss/n
    return avg_loss


def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    model.eval()#evaluation mode
    
    all_loss = 0.0
    correct = 0#correct prediction
    n=0#all sample num
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='validate'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)#為batch_size*class num的tensor
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)#找出每個output預測最大值的label的index
            
            num=images.size(0)#batch size
            all_loss+=loss.item()*num
            correct+=(predicted==labels).sum().item()
            n+=num
    avg_loss = all_loss/n
    accuracy = correct/n
    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, criterion, device):
    # (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
    model.eval()#evaluation mode
    
    predict = []
    imageID = []
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc = 'test'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            
            predict.extend(predicted.cpu().numpy())
            imageID.extend(ids)
            
    results = {'id':imageID, 'prediction':predict}
    df = pd.DataFrame(results)
    df.to_csv('CNN.csv', index = False)
    print(f"Predictions saved to 'CNN.csv'")
    return