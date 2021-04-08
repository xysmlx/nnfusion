import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
from torchvision import datasets, transforms  
import torchvision  
from torch.autograd import Variable  
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, StepLR
import argparse

import time
import numpy as np
import copy

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(256, 256, bias=False)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(256, 256, bias=False)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(256, 256, bias=False)
        self.fc4 = nn.Linear(256, 10, bias=False)


    def forward(self, x):
        # flatten image input
        x = x.view(-1, 256)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

model = MLP().cuda()
dummy_input = torch.rand(256,256).cuda()
# print(model(data))
torch.onnx.export(model, dummy_input, 'mlp.onnx', verbose=True)
