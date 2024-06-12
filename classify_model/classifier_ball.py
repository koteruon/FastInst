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
        # self.conv1 = nn.Conv1d(10, 64, kernel_size=3, padding=1) #100d
        # self.conv1 = nn.Conv1d(10, 64, kernel_size=3, padding=1) #99d
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
      
        # self.fc1 = nn.Linear(6400, 64)  # Adjust the input size based on the output shape of the last conv layer
        self.fc1 = nn.Linear(6144, 64)  # Adjust the input size based on the output shape of the last conv layer
        # self.fc1 = nn.Linear(256 * 3, 64)
        # self.fc1 = nn.Linear(2, 64)
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
class Conv1DNet_ori(nn.Module):
    def __init__(self, num_classes):
        super(Conv1DNet_ori, self).__init__()
        self.conv1 = nn.Conv1d(10, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 64)  # Adjust the input size based on the output shape of the last conv layer
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        print('-------------------------------')
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        # x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x 
#Load Data

train_data_list = []
# cls_0 = ['IMG_7386','IMG_7388','IMG_7390']
# cls_1 = ['IMG_7394','IMG_7397']
# cls_2 = ['IMG_7404','IMG_7406','IMG_7408']
# cls_3 = ['IMG_7407','IMG_7409'] 
cls_0 = ['mediapipe/dataset_new/bhc_left_1.npy','mediapipe/dataset_new/bhc_left_2.npy']
cls_1 = ['mediapipe/dataset_new/bhpull_left_1.npy']
cls_2 = ['mediapipe/dataset_new/bhpush_left_1.npy', 'mediapipe/dataset_new/bhpush_left_2.npy']
cls_3 = ['mediapipe/dataset_new/bht_left_1.npy', 'mediapipe/dataset_new/bht_left_2.npy']
cls_4 = ['mediapipe/dataset_new/fhc_left_1.npy']
cls_5 = ['mediapipe/dataset_new/fhpull_left_1.npy']
cls_6 = ['mediapipe/dataset_new/fhpull_left_2.npy']
cls_7 = ['mediapipe/dataset_new/fhs_left_1.npy', 'mediapipe/dataset_new/fhs_left_2.npy']

'''
data_path = r'/home/chenzy/FastInst-main/output/data_test_1_10f.txt'
f_0 = open(data_path,"r")
data_0 = f_0.read()
# print(data_0.shape)
train_arr = np.genfromtxt(data_0.splitlines(), delimiter=' ', dtype=float)
# 將浮點數轉換成整數
train_arr = train_arr.astype(int)
# 印出 NumPy 陣列中的第一列
print(train_arr.shape)

data_label_path = r'/home/chenzy/FastInst-main/output/data_test_1_label_10f_3cls.txt'
data_label = open(data_label_path,"r")
data_1 = data_label.read()
# print(data_0.shape)
label_arr = np.genfromtxt(data_1.splitlines(), delimiter=' ', dtype=float)
# 將浮點數轉換成整數
label_arr = label_arr.astype(int)
# 印出 NumPy 陣列中的第一列
print(label_arr.shape)

train_data_list = []
# test_data_list = []
for i in range(len(train_arr)):
#     print(np_train_data_reshape[i].shape())
    train_data_list.append([train_arr[i], label_arr[i]])
# train_data_list
print(train_data_list[1])
# exit()
# train_dataset_np = np.array(train_data_list)
'''
'''
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
        # temp.append(float(data_0_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10,99 )
    # print(len(np_train_data_temp))
    
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], i])

# print(train_data_list)

'''

'''for i_0 in range(len(cls_0)):
    cls_0_txt_path = f'/home/chenzy/FastInst-main/output/video_output/{cls_0[i_0]}_R_paddle.txt'
    f_0 = open(cls_0_txt_path,"r")
    data_0 = f_0.read()
    data_0_paddle_list = data_0.split(',')
    f_0.close()
    # /home/chenzy/FastInst-main/mediapipe/IMG_7386_Trim_mediaP.npy
    npy_pose = np.load(f'/home/chenzy/FastInst-main/mediapipe/dataset/{cls_0[i_0]}_Trim_mediaP.npy')
    # print(len(npy_pose))

    result = []
    times = len(npy_pose) -(len(npy_pose)%10)
    for j in range(times):
        temp = npy_pose[j].ravel().tolist() 
        # temp = [] # area only
        temp.append(float(data_0_paddle_list[j]))
        result.append(temp)

    # print(len(result))
    np_train_data_temp = np.array(result) .reshape(-1, 10, 100)
    # np_train_data_temp = np.array(result) .reshape(-1, 10, 1)
   
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], 0])

for i_1 in range(len(cls_1)):
    cls_1_txt_path = f'/home/chenzy/FastInst-main/output/video_output/{cls_1[i_1]}_R_paddle.txt'
    f_1 = open(cls_1_txt_path,"r")
    data_1 = f_1.read()
    data_1_paddle_list = data_1.split(',')
    f_1.close()
    npy_pose = np.load(f'/home/chenzy/FastInst-main/mediapipe/dataset/{cls_1[i_1]}_Trim_mediaP.npy')
    # print(len(npy_pose))

    result = []
    times = len(npy_pose) -(len(npy_pose)%10)
    for j in range(times):
        temp = npy_pose[j].ravel().tolist()
        # temp = [] # area only
        temp.append(float(data_1_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10, 100)
    # print(len(np_train_data_temp))
    # np_train_data_temp = np.array(result) .reshape(-1, 10, 1)

    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], 1])

for i_2 in range(len(cls_2)):
    cls_2_txt_path = f'/home/chenzy/FastInst-main/output/video_output/{cls_2[i_2]}_R_paddle.txt'
    f_2 = open(cls_2_txt_path,"r")
    data_2 = f_2.read()
    data_2_paddle_list = data_2.split(',')
    f_2.close()
    npy_pose = np.load(f'/home/chenzy/FastInst-main/mediapipe/dataset/{cls_2[i_2]}_Trim_mediaP.npy')
    # print(len(npy_pose))

    result = []
    times = len(npy_pose) -(len(npy_pose)%10)
    for j in range(times):
        temp = npy_pose[j].ravel().tolist()
        # temp = [] # area only
        temp.append(float(data_0_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10, 100)
    # np_train_data_temp = np.array(result).reshape(-1, 10, 1)
    # print(len(np_train_data_temp))
    
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], 2])'''

for i_0 in range(len(cls_0)):
    # cls_2_txt_path = f'/home/chenzy/FastInst-main/output/video_output/{cls_2[i_1]}_R_paddle.txt'
    # f_2 = open(cls_2_txt_path,"r")
    # data_2 = f_2.read()
    # data_2_paddle_list = data_2.split(',')
    # f_2.close()
    npy_pose = np.load(f'/home/chenzy/FastInst-main/{cls_0[i_0]}')
    print(len(npy_pose))

    result = []
    times = len(npy_pose) -(len(npy_pose)%10)
    for j in range(times):
        temp = npy_pose[j].ravel().tolist()
        # temp = [] # pose only
        # temp.append(float(data_0_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10, 99)
    # np_train_data_temp = np.array(result).reshape(-1, 10, 1)
    print(len(np_train_data_temp))
    
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], 0])

for i_1 in range(len(cls_1)):
    # cls_2_txt_path = f'/home/chenzy/FastInst-main/output/video_output/{cls_2[i_1]}_R_paddle.txt'
    # f_2 = open(cls_2_txt_path,"r")
    # data_2 = f_2.read()
    # data_2_paddle_list = data_2.split(',')
    # f_2.close()
    npy_pose = np.load(f'/home/chenzy/FastInst-main/{cls_1[i_1]}')
    # print(len(npy_pose))

    result = []
    times = len(npy_pose) -(len(npy_pose)%10)
    for j in range(times):
        temp = npy_pose[j].ravel().tolist()
        # temp = [] # area only
        # temp.append(float(data_0_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10, 99)
    # np_train_data_temp = np.array(result).reshape(-1, 10, 1)
    # print(len(np_train_data_temp))
    
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], 1])

for i_2 in range(len(cls_2)):
    # cls_2_txt_path = f'/home/chenzy/FastInst-main/output/video_output/{cls_2[i_1]}_R_paddle.txt'
    # f_2 = open(cls_2_txt_path,"r")
    # data_2 = f_2.read()
    # data_2_paddle_list = data_2.split(',')
    # f_2.close()
    npy_pose = np.load(f'/home/chenzy/FastInst-main/{cls_2[i_2]}',allow_pickle=True)
    # print(len(npy_pose))

    result = []
    times = len(npy_pose) -(len(npy_pose)%10)
    for j in range(times):
        temp = npy_pose[j].ravel().tolist()
        # temp = [] # area only
        # temp.append(float(data_0_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10, 99)
    # np_train_data_temp = np.array(result).reshape(-1, 10, 1)
    # print(len(np_train_data_temp))
    
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], 2])

for i_3 in range(len(cls_3)):
    # cls_2_txt_path = f'/home/chenzy/FastInst-main/output/video_output/{cls_2[i_1]}_R_paddle.txt'
    # f_2 = open(cls_2_txt_path,"r")
    # data_2 = f_2.read()
    # data_2_paddle_list = data_2.split(',')
    # f_2.close()
    npy_pose = np.load(f'/home/chenzy/FastInst-main/{cls_3[i_3]}')
    # print(len(npy_pose))

    result = []
    times = len(npy_pose) -(len(npy_pose)%10)
    for j in range(times):
        temp = npy_pose[j].ravel().tolist()
        # temp = [] # area only
        # temp.append(float(data_0_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10, 99)
    # np_train_data_temp = np.array(result).reshape(-1, 10, 1)
    # print(len(np_train_data_temp))
    
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], 3])

for i_4 in range(len(cls_4)):
    # cls_2_txt_path = f'/home/chenzy/FastInst-main/output/video_output/{cls_2[i_1]}_R_paddle.txt'
    # f_2 = open(cls_2_txt_path,"r")
    # data_2 = f_2.read()
    # data_2_paddle_list = data_2.split(',')
    # f_2.close()
    npy_pose = np.load(f'/home/chenzy/FastInst-main/{cls_4[i_4]}')
    # print(len(npy_pose))

    result = []
    times = len(npy_pose) -(len(npy_pose)%10)
    for j in range(times):
        temp = npy_pose[j].ravel().tolist()
        # temp = [] # area only
        # temp.append(float(data_0_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10, 99)
    # np_train_data_temp = np.array(result).reshape(-1, 10, 1)
    # print(len(np_train_data_temp))
    
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], 4])

for i_5 in range(len(cls_5)):
    # cls_2_txt_path = f'/home/chenzy/FastInst-main/output/video_output/{cls_2[i_1]}_R_paddle.txt'
    # f_2 = open(cls_2_txt_path,"r")
    # data_2 = f_2.read()
    # data_2_paddle_list = data_2.split(',')
    # f_2.close()
    npy_pose = np.load(f'/home/chenzy/FastInst-main/{cls_5[i_5]}')
    # print(len(npy_pose))

    result = []
    times = len(npy_pose) -(len(npy_pose)%10)
    for j in range(times):
        temp = npy_pose[j].ravel().tolist()
        # temp = [] # area only
        # temp.append(float(data_0_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10, 99)
    # np_train_data_temp = np.array(result).reshape(-1, 10, 1)
    # print(len(np_train_data_temp))
    
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], 5])

for i_6 in range(len(cls_6)):
    # cls_2_txt_path = f'/home/chenzy/FastInst-main/output/video_output/{cls_2[i_1]}_R_paddle.txt'
    # f_2 = open(cls_2_txt_path,"r")
    # data_2 = f_2.read()
    # data_2_paddle_list = data_2.split(',')
    # f_2.close()
    npy_pose = np.load(f'/home/chenzy/FastInst-main/{cls_6[i_6]}')
    # print(len(npy_pose))

    result = []
    times = len(npy_pose) -(len(npy_pose)%10)
    for j in range(times):
        temp = npy_pose[j].ravel().tolist()
        # temp = [] # area only
        # temp.append(float(data_0_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10, 99)
    # np_train_data_temp = np.array(result).reshape(-1, 10, 1)
    # print(len(np_train_data_temp))
    
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], 6])

for i_7 in range(len(cls_7)):
    # cls_2_txt_path = f'/home/chenzy/FastInst-main/output/video_output/{cls_2[i_1]}_R_paddle.txt'
    # f_2 = open(cls_2_txt_path,"r")
    # data_2 = f_2.read()
    # data_2_paddle_list = data_2.split(',')
    # f_2.close()
    npy_pose = np.load(f'/home/chenzy/FastInst-main/{cls_7[i_7]}')
    # print(len(npy_pose))

    result = []
    times = len(npy_pose) -(len(npy_pose)%10)
    for j in range(times):
        temp = npy_pose[j].ravel().tolist()
        # temp = [] # area only
        # temp.append(float(data_0_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10, 99)
    # np_train_data_temp = np.array(result).reshape(-1, 10, 1)
    # print(len(np_train_data_temp))
    
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], 7])

'''
for i_3 in range(len(cls_3)):
    cls_3_txt_path = f'/home/chenzy/FastInst-main/output/video_output/{cls_3[i_3]}_R_paddle.txt'
    f_3 = open(cls_3_txt_path,"r")
    data_3 = f_3.read()
    data_3_paddle_list = data_3.split(',')
    f_3.close()
    npy_pose = np.load(f'/home/chenzy/FastInst-main/mediapipe/dataset/{cls_3[i_3]}_Trim_mediaP.npy')

    result = []
    if len(npy_pose)%10!=0:
        times = len(npy_pose) -(len(npy_pose)%10)
    else:
        times= len(npy_pose)

    for j in range(times):
        temp = npy_pose[j].ravel().tolist()
        # print(j)
        temp.append(float(data_3_paddle_list[j]))
        result.append(temp)

    np_train_data_temp = np.array(result).reshape(-1, 10, 100)
    # print(len(np_train_data_temp))
    
    for k in range(len(np_train_data_temp)):
        train_data_list.append([np_train_data_temp[k], 3])
'''
# train_dataset_np.shape()
# print(len(train_data_list))
# exit()
dataset = train_data_list
# print(dataset)
train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print('------')
print(train_dataset[0][0].shape)
# print(train_dataset[0])

train_size2 = int(0.85 * len(train_dataset))
test_size = len(train_dataset) - train_size2
train_dataset, test_dataset = random_split(train_dataset, [train_size2, test_size])

print(len(train_dataset), len(val_dataset), len(test_dataset))

# define training hyperparameters
INIT_LR = 1e-4
BATCH_SIZE = 2
EPOCHS = 150

# initialize the train, validation, and test data loaders
# trainDataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
trainDataLoader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

trainSteps = len(train_dataset) // BATCH_SIZE
valSteps = len(val_dataset) // BATCH_SIZE

# initialize the Conv3DNet model
print("[INFO] initializing the Conv1DNet model...")
num_classes = 8
labels = [0, 1, 2, 3, 4, 5, 6, 7]
# model = BiLSTM(10,64,100,num_classes)
model = Conv1DNet(num_classes)
# print(model)


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

torch.save(model.state_dict(), "./classify_model/model_0611_poseonly_8cls.pth")

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



