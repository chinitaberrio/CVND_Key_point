## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)

        # maxpool layer: pool with kernel_size=2, stride=2
        # output size (32, 110, 110)
        self.pool1 = nn.MaxPool2d(2, 2)

        # batch norm
        self.bn1 = nn.BatchNorm2d(32)

        # 10 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        self.conv2 = nn.Conv2d(32, 64, 5)

        # maxpool layer: pool with kernel_size=2, stride=2
        # output size (10, 53, 53)
        self.pool2 = nn.MaxPool2d(2, 2)

        # batch norm
        self.bn2 = nn.BatchNorm2d(64)

        # 10 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        self.conv3 = nn.Conv2d(64, 128, 5)

        # maxpool layer: pool with kernel_size=2, stride=2
        # output size (128, 24, 24)
        self.pool3 = nn.MaxPool2d(2, 2)

        # batch norm
        self.bn3 = nn.BatchNorm2d(128)

        # 5 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (24-3)/1 +1 = 22
        self.conv4 = nn.Conv2d(128, 256, 3)

        # maxpool layer: pool with kernel_size=2, stride=2
        # output size (256, 11, 11)
        self.pool4 = nn.MaxPool2d(2, 2)

        # batch norm
        self.bn4= nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 11 * 11, 2000)

        self.fc2 = nn.Linear(2000, 800)

        self.fc3 = nn.Linear(800, 136)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        x = self.pool3(self.bn3(F.relu(self.conv3(x))))
        x = self.pool4(self.bn4(F.relu(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x