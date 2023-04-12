import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class MapDataset(torch.utils.data.IterableDataset):
    def __init__(self, pickles, scaler=None, building_value=None, unsampled_value=None, sampled_value=None):
        super().__init__()
        self.pickles = pickles
        self.scaler = scaler
        self.building_value = building_value
        self.unsampled_value = unsampled_value
        self.sampled_value = sampled_value

    def __iter__(self):
        yield from file_path_generator(self.pickles, self.scaler, self.building_value, self.unsampled_value, self.sampled_value)

def file_path_generator(pickles, scaler, building_value=None, unsampled_value=None, sampled_value=None):
    for file_path in pickles:
        t_x_points, t_y_points, t_y_masks, t_channel_pows = load_numpy_array(
            file_path, scaler, building_value=building_value, unsampled_value=unsampled_value, sampled_value=sampled_value)
        for i, (t_x_point, t_y_point, t_y_mask, t_channel_pow) in enumerate(zip(t_x_points, t_y_points, t_y_masks, t_channel_pows)):
            yield t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, i


def load_numpy_array(file_path, scaler, building_value=None, unsampled_value=None, sampled_value=None):
    t_x_points, t_channel_pows, t_y_masks = np.load(file_path, allow_pickle=True)
    t_y_points = t_channel_pows * t_y_masks

    if scaler:
        t_x_mask = t_x_points[:,1,:,:] == 1
        t_x_points[:,0,:,:] = scaler.transform(t_x_points[:,0,:,:]) * t_x_mask
        t_channel_pows = scaler.transform(t_channel_pows)
        t_y_points = scaler.transform(t_y_points)

    if building_value:
        t_x_points[:,0][t_x_points[:,1] == -1] = building_value
    
    if unsampled_value:
        t_x_points[:,0][t_x_points[:,1] == 0] = unsampled_value

    if sampled_value:
        t_x_points[:,0][t_x_points[:,1] == 1] += sampled_value
    
    return t_x_points, t_y_points, t_y_masks, t_channel_pows
  

class Scaler():
    def __init__(self, scaler='minmax', bounds=(0, 1), min_trunc=None, max_trunc=None):
        self.scaler = scaler
        self.bounds = bounds
        self.min_trunc = min_trunc
        self.max_trunc = max_trunc
        if scaler == 'minmax':
            self.sc = MinMaxScaler(feature_range=self.bounds)
        else:
            self.sc = StandardScaler()
    
    def fit(self, data):
        data = data.flatten().reshape(-1,1)
        self.sc.partial_fit(data)
        if self.min_trunc:
            if self.sc.data_min_ < self.min_trunc:
                self.sc.data_min_ = self.min_trunc
        if self.max_trunc:
            if self.sc.data_max_ > self.max_trunc:
                self.sc.data_max_ = self.max_trunc

    def transform(self, data):
        data_shape = data.shape
        data = data.flatten().reshape(-1,1)
        if self.min_trunc:
            data[data < self.min_trunc] = self.min_trunc
        if self.max_trunc:
            data[data > self.max_trunc] = self.max_trunc
        data = self.sc.transform(data)
        data = data.reshape(data_shape)        
        return data
    
    def reverse_transform(self, data):
        data_shape = data.shape
        data = data.flatten().reshape(-1,1)
        data = self.sc.inverse_transform(data)
        data = data.reshape(data_shape)
        return data

def train_scaler(scaler, pickles):
    gen = file_path_generator(pickles, scaler=None)
    for t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, i in gen:
        scaler.fit(t_channel_pow)