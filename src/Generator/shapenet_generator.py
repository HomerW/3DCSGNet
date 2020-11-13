import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
import sys
from .stack import SimulateStack

def get_shapenet_data(train_data_path, val_data_path, train_num, val_num, batch_size=32):
    train_data = h5py.File(train_data_path, 'r')
    val_data = h5py.File(val_data_path, 'r')
    train_voxels = torch.from_numpy(train_data['voxels'][:]).flip(dims=[3]).float()
    val_voxels = torch.from_numpy(val_data['voxels'][:]).flip(dims=[3]).float()

    train_dataset = DataLoader(train_voxels[:train_num], batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(val_voxels[:val_num], batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset
