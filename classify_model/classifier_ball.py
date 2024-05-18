from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import glob
import os
import cv2
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.utils.data import random_split
# set the numpy seed for better reproducibility
import numpy as np
# import the necessary packages
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
import argparse
import imutils
import torch
import cv2
import time
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix

# BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

# Conv1DNet model
class Conv1DNet(nn.Module):
    def __init__(self, num_classes):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(10, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6400, 64)  # Adjust the input size based on the output shape of the last conv layer
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # print('-------------------------------')
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu3(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x
    
#Load Data

train_data_list = []

for i in range(3):

    cls_0_txt_path = f'/home/chenzy/FastInst-main/mediapipe/data/{i}.txt'
    f_0 = open(cls_0_txt_path,"r")
    data_0 = f_0.read()
    data_0_paddle_list = data_0.split(',')
    npy_pose = np.load(f'/home/chenzy/FastInst-main/mediapipe/data/{i}.npy')
    # print(len(npy_pose))

    result = []
    times = len(npy_pose) -(len(npy_pose)%10)
    for j in range(times):
        temp = npy_pose[j].ravel().tolist()
        temp.append(float(data_0_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10, 100)
    # print(len(np_train_data_temp))
    
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], i])

# print(train_data_list)
    
dataset = train_data_list
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print('------')
print(train_dataset[0][0].shape)
print(train_dataset[0])

train_size2 = int(0.9 * len(train_dataset))
test_size = len(train_dataset) - train_size2
train_dataset, test_dataset = random_split(train_dataset, [train_size2, test_size])

print(len(train_dataset), len(val_dataset), len(test_dataset))

# define training hyperparameters
INIT_LR = 1e-4
BATCH_SIZE = 5
EPOCHS = 100

# initialize the train, validation, and test data loaders
# trainDataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
trainDataLoader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

trainSteps = len(train_dataset) // BATCH_SIZE
valSteps = len(val_dataset) // BATCH_SIZE

# initialize the Conv3DNet model
print("[INFO] initializing the Conv3DNet model...")
num_classes = 3
labels = [0,1,2]
# model = BiLSTM(10,64,100,num_classes)
model = Conv1DNet(num_classes)
print(model)


# initialize our optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr = INIT_LR)
criterion = nn.CrossEntropyLoss()
# initialize a dictionary to store training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
model = model.cuda()
# model = model.to(device)
criterion = criterion.to(device)

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

# loop over our epochs
for idx, e in enumerate(range(0, EPOCHS)):
    print('------------------------------------------------------------------------------------------------')
    print(f'Epoch {e + 1}/{EPOCHS}:', end = ' ')

    # set the model in training mode
    model = model.cuda()
    model.train()
    
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    
    # initialize the number of correct predictions in the training and validation step
    trainCorrect = 0
    valCorrect = 0
    
    # loop over the training set
    for x, y in trainDataLoader:
        # send the input to the device
#         (x, y) = (x.to(device), y.to(device))
        x = x.float().cuda(non_blocking = True)
        y = y.view(-1).cuda(non_blocking = True)
        
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = criterion(pred, y)
        # print(pred)
        # print('--------------------------')
        # print(labels)
        
        # zero out the gradients, perform the backpropagation step, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # add the loss to the total training loss so far and calculate the number of correct predictions
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        
      
        
        
        
    # switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model = model.cuda()
        model.eval()
        # loop over the validation set
        for (x, y) in valDataLoader:
            # send the input to the device
#             (x, y) = (x.to(device), y.to(device))
            x = x.float().cuda(non_blocking = True)
            y = y.view(-1).cuda(non_blocking = True)
        
            # make the predictions and calculate the validation loss
            pred = model(x)
            totalValLoss += criterion(pred, y)
            
            # calculate the number of correct predictions
            valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item() 
        
#     torch.save(model.state_dict(), f'/home/yoson/SparseInst/official/SparseInst/table-tennis/pose_data/model/modelS_splitorder_pose_model_allframe_{idx}.pth')
        
    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    # calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)

    print(len(trainDataLoader.dataset))

    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
    
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
        avgValLoss, valCorrect))
        
        
# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

# plot the training loss and accuracy8x4194304 
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# plt.savefig(args["plot"])
# serialize the model to disk
# torch.save(model, args["model"])

torch.save(model.state_dict(), "./classify_model/model.pth")

from sklearn.metrics import classification_report
# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
# turn off autograd for testing evaluation
print("===============Train_dataset=====================")

with torch.no_grad():
    # set the model in evaluation mode
    model.eval()
    
    # initialize a list to store our predictions
    preds = []
    # loop over the test set
    for (x, y) in trainDataLoader:
        # send the input to the device
#         x = x.to(device)
        x = x.float().cuda(non_blocking=True)
        # make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())
print([y for x,y in train_dataset], preds, labels)
print(classification_report([y for x,y in train_dataset], preds, labels=labels))

print("============val_dataset===================")
from sklearn.metrics import classification_report
# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
# turn off autograd for testing evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()
    
    # initialize a list to store our predictions
    preds = []
    # loop over the test set
    for (x, y) in valDataLoader:
        # send the input to the device
#         x = x.to(device)
        x = x.float().cuda(non_blocking=True)
        # make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())
        
# generate a classification report
# print([y for x,y in val_dataset], preds, labels)
print(classification_report([y for x,y in val_dataset], preds, labels=labels))

from sklearn.metrics import classification_report
# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
print("============Test_dataset===================")
# turn off autograd for testing evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()
    
    # initialize a list to store our predictions
    preds = []
    # loop over the test set
    for (x, y) in testDataLoader:
        # send the input to the device
#         print(x)
#         x = x.to(device)
        x = x.float().cuda(non_blocking=True)
        # make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())
        
# generate a classification report
print(len(train_dataset), len(val_dataset), len(test_dataset))
# print([y for x,y in test_dataset], preds, labels)
print(classification_report([y for x,y in test_dataset], preds, labels=labels))



