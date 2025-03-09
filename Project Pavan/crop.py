import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opti
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import datetime
import os

data = pd.read_csv("Crop_recommendation.csv")

features = data.iloc[:,:-1].values
input_size = len(features[0])
labels = data.iloc[:,-1].values

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
num_classes = len(np.unique(labels)) 

features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels ,dtype=torch.long)

mean = features.mean(dim=0)
std = features.std(dim = 0)
features = (features - mean)/std
np.savez("normalization.npz", mean=mean, std=std)

with open("../web/pickle/encoder.pkl", "wb") as file:
    pickle.dump(encoder,file)

class CustomDataset(Dataset):
    def __init__(self,features,labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self,index):
        feature = self.features[index]
        label = self.labels[index]
        return feature , label
        
dataset = CustomDataset(features,labels)
train_size = int(0.8*len(dataset))
val_size = len(dataset) - train_size
train_dataset,val_dataset = torch.utils.data.random_split(dataset,[train_size , val_size])

class Neural_network(nn.Module):
    def __init__(self,input_size,num_classes):
        super(Neural_network,self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x)

network = Neural_network(7,22)
loss_function = nn.CrossEntropyLoss()
optimizer = opti.Adam(network.parameters(),lr= 0.0001)

train_losses = []
val_losses = []
EPOCH = 100
train_accuracies = []
val_accuracies = []

for epoch in range(EPOCH):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataset):
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_dataset)
    train_losses.append(train_loss)
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels in val_dataset:
            outputs = network(inputs)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
    if epoch % 10 == 9:
        print('Epoch :',epoch+1) 
print('Finished training')


model_name = "crop_prediction_model"+".hdf5"
torch.save(network.state_dict(),model_name)
model = Neural_network(7,22)

normalization_data = np.load("normalization.npz")
mean = torch.tensor(normalization_data["mean"])
std = torch.tensor(normalization_data["std"])

with open("../web/pickle/encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

model = Neural_network(input_size, num_classes)
model.load_state_dict(torch.load('crop_prediction_model.hdf5'))
model.eval()


def predict():
    print("Enter Soil Information :")
    P = float(input("Enter Phosphorus level :"))
    K = float(input("Enter Pottasium level :"))
    N = float(input("Enter Nitrogen level :"))
    temperature  = float(input("Temperature (in celcius):"))
    humidity = float(input("Humidity % :"))
    ph = float(input("ph level:"))
    rainfall = float(input("Rainfall in (mm) :"))

    in_features = torch.tensor([N,P,K,temperature,humidity,ph,rainfall])
    in_features = (in_features - mean)/std

    with torch.no_grad():
        output = model(in_features)
        prediction = output.argmax().item()
    crop = encoder.inverse_transform([prediction])[0]
    print("Crop -",crop)
predict()