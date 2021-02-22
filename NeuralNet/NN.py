'''
code = plain Neural Network
framework = pytorch
author= susant achary
email= sache.meet@yahoo.com
'''

#import
import torch
from torch._C import MobileOptimizerType
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as flt
from torch.utils.data import dataloader, dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes): #(28x28)
        super(NN, self).__init__()  #super calls initilaization methods of parent class
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = flt.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#load data
train_dataset = datasets.MNIST(root = 'dataset/', train=True, transform = transforms.ToTensor(), download=True)
train_loader = dataloader.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root = 'dataset/', train=False, transform = transforms.ToTensor(), download=True)
test_loader = dataloader.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialise network
model = NN(input_size=input_size,num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


#train network
for epoch in range(num_epochs):
    for batch_idx, (data , targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        #to correct shape
        data = data.reshape(data.shape[0], -1)

        #forward
        score = model(data)
        loss = criterion(score,targets)

        #backword
        optimizer.zero_grad()
        loss.backward()

        #gradient adam
        optimizer.step()

        #print(data.shape)

#check accuracy on training & test to see how good out

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("accuracy on train data")
    else:
        print("checking accuracy on test data")

    num_correct = 0
    num_sample = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            score = model(x)
            _, predictions = score.max(1)
            num_correct += (predictions == y).sum()
            num_sample += predictions.size(0)
        print(f'Got{num_correct}/{num_sample} with accuracy {float(num_correct)/float(num_sample)*100:.2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)