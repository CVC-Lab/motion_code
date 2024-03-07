import joblib
import re
import os
import numpy as np
import pandas as pd
from datetime import datetime
from .spatio_temporal_CNN import TemporalFusionFCN
from .ppmi_dataset import PPMIDataset, custom_collate_fn
from .data_processing import process_ppmi_data_for_motion_codes
from .motion_code import MotionCode
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import sys

batch_size = 1
# filter out rows for which labels do not exist in your dataset
root_dir = '/mnt/data/ashwin/DTI/PPMI/'
train_dataset = PPMIDataset(root_dir=root_dir, split_type='train')
test_dataset = PPMIDataset(root_dir=root_dir, split_type='test')
print('len train:', len(train_dataset))
print('len test:', len(test_dataset))

# load model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = TemporalFusionFCN(input_channels=256).to(device)
model.load_state_dict(torch.load('artifacts/best_st_cnn.pth'))
model.eval()

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

train_X = []
train_labels = []
test_X = []
test_labels = []

for scans, labels in train_dataloader:
    scans, labels = scans.to(device), labels.to(device)
    ts = model.fcn(scans)
    train_X.append(ts.detach().cpu().numpy())
    train_labels.append(labels.detach().cpu())
    
train_X = np.stack(train_X, axis=0).squeeze()
train_labels = np.stack(train_labels, axis=0).squeeze()

for scans, labels in test_dataloader:
    scans, labels = scans.to(device), labels.to(device)
    ts = model.fcn(scans)
    test_X.append(ts.detach().cpu().numpy())
    test_labels.append(labels.detach().cpu())
    
test_X = np.stack(test_X, axis=0).squeeze()
test_labels = np.stack(test_labels, axis=0).squeeze()

ds = {
    'train_X': train_X,
    'test_X': test_X,
    'train_labels': train_labels,
    'test_labels': test_labels
}
joblib.dump(ds, 'artifacts/timeseries_ds.pkl')


