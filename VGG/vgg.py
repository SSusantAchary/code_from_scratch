'''
code = VGG net (11,13,16,19)
framework = pytorch
author= susant achary
email= sache.meet@yahoo.com
'''


import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
import torch.optim as optim
import torch.nn.functional as ftl
from torch.utils.data import dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

VGG_version = {
    'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,'M'] ,
    'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,'M'],
    'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,'M'],
    'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,'M']
}

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_channels=1000,version=""):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_version[version])

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_channels)
        )
    def forward(self,x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            nn.BatchNorm2d(x),
                            nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]

        return nn.Sequential(*layers)

def test_vgg():
    model = VGG_net(in_channels=3, num_channels=1000,version="VGG13")
    x = torch.randn(1, 3, 224, 224)
    y = model(x).to('cuda')
    print(model)
    print(y.shape)


if __name__ == "__main__":
    test_vgg()

