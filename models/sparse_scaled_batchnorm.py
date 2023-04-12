import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SparseScaledBNConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', scale=9):
      super().__init__()
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.kernel_size = kernel_size
      self.scale = scale
      self.stride = stride
      if self.stride != 1:
        raise Exception(f'Sparse Convolution only supports the stride of 1. Stride = {self.stride}')

      self.feature_conv = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, bias=False, padding=padding)
      self.norm_conv = torch.nn.Conv2d(1, self.out_channels, self.kernel_size, bias=False, padding=padding)
      self.norm_conv.requires_grad_(False)
      self.bias = torch.zeros(out_channels, 1, 1)
      if torch.cuda.is_available():
          self.bias = self.bias.cuda()
      self.bias = torch.nn.Parameter(self.bias) 
      self.batch_norm = torch.nn.BatchNorm2d(out_channels)

      self.binary_mask = None
      self.norm = None  

      self._init_weights()

    def forward(self, tensor, binary_mask=None):
      if binary_mask is None: 
          b, c, h, w = tensor.shape
          binary_mask = torch.ones(b, 1, h, w)

      features = tensor * binary_mask
      features = self.feature_conv(features)

      norm = self.norm_conv(binary_mask)
      norm = torch.where(torch.eq(norm, 0), torch.zeros_like(norm), torch.reciprocal(norm))

      # Assign latest values to layer attributes so that we can check them after training
      self.binary_mask = binary_mask
      self.norm = norm

      feature = features * norm * self.scale + self.bias
      feature = self.batch_norm(feature)

      return feature, binary_mask

    def _init_weights(self):
      torch.nn.init.ones_(self.norm_conv.weight)
      torch.nn.init.kaiming_normal_(self.feature_conv.weight, mode='fan_out', nonlinearity='relu')