import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, enc_in, enc_out, n_dim, leaky_relu_alpha=0.3):
        super(Encoder, self).__init__()
        self.conv2d = nn.Conv2d(enc_in, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_1 = nn.Conv2d(n_dim+1, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_2 = nn.Conv2d(n_dim+1, n_dim, kernel_size=(3, 3), padding='same')
        self.average_pooling2d = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv2d_3 = nn.Conv2d(n_dim+1, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_4 = nn.Conv2d(n_dim+1, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_5 = nn.Conv2d(n_dim+1, n_dim, kernel_size=(3, 3), padding='same')
        self.average_pooling2d_1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv2d_6 = nn.Conv2d(n_dim+1, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_7 = nn.Conv2d(n_dim+1, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_8 = nn.Conv2d(n_dim+1, n_dim, kernel_size=(3, 3), padding='same')
        self.average_pooling2d_2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.mu = nn.Conv2d(n_dim+1, enc_out, kernel_size=(3, 3), padding='same')

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_alpha)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x_):
        x = x_[:,0].unsqueeze(1)
        m = x_[:,1].unsqueeze(1)

        x = torch.cat([x, m], 1)
        x = self.leaky_relu(self.conv2d(x))
        x = torch.cat([x, m], 1)
        x = self.leaky_relu(self.conv2d_1(x))
        x = torch.cat([x, m], 1)
        x = self.leaky_relu(self.conv2d_2(x))
        skip1 = x

        x = self.average_pooling2d(x)
        m = torch.nn.functional.interpolate(m, scale_factor = (0.5, 0.5))

        x = torch.cat([x, m], 1)
        x = self.leaky_relu(self.conv2d_3(x))
        x = torch.cat([x, m], 1)
        x = self.leaky_relu(self.conv2d_4(x))
        x = torch.cat([x, m], 1)
        x = self.leaky_relu(self.conv2d_5(x))
        skip2 = x

        x = self.average_pooling2d_1(x)
        m = torch.nn.functional.interpolate(m, scale_factor = (0.5, 0.5))

        x = torch.cat([x, m], 1)
        x = self.leaky_relu(self.conv2d_6(x))
        x = torch.cat([x, m], 1)
        x = self.leaky_relu(self.conv2d_7(x))
        x = torch.cat([x, m], 1)
        x = self.leaky_relu(self.conv2d_8(x))
        skip3 = x

        x = self.average_pooling2d_2(x)
        m = torch.nn.functional.interpolate(m, scale_factor = (0.5, 0.5))

        x = torch.cat([x, m], 1)
        x = self.leaky_relu(self.mu(x))
        return x, skip1, skip2, skip3