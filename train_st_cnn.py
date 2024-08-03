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


# filter out rows for which labels do not exist in your dataset
root_dir = '/mnt/data/ashwin/DTI/PPMI/'
train_dataset = PPMIDataset(root_dir=root_dir, split_type='train')
test_dataset = PPMIDataset(root_dir=root_dir, split_type='test')
print('len train:', len(train_dataset))
print('len test:', len(test_dataset))
# Parameters (You might want to adjust these)
epochs = 20
learning_rate = 0.001
batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Example model and dataloader setup
model = TemporalFusionFCN(input_channels=256).to(device)
criterion = nn.CrossEntropyLoss()  # Update depending on your task
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

def train_one_epoch(epoch_index, dataloader):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch_idx, (scans, labels) in enumerate(dataloader):
        scans, labels = scans.to(device), labels.to(device)
        # Forward pass
        outputs = model(scans)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_predictions / total_predictions
    print(f"Epoch [{epoch_index+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

def test_model(dataloader):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    best_loss = np.inf

    with torch.no_grad():
        for scans, labels in dataloader:
            scans, labels = scans.to(device), labels.to(device)
            outputs = model(scans)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss

best_loss = np.inf
for epoch in range(epochs):
    start_time = time.time()
    train_one_epoch(epoch, train_dataloader)
    avg_loss = test_model(test_dataloader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        print('saved')
        torch.save(model.state_dict(), 'artifacts/best_st_cnn.pth')
    print(f"Epoch {epoch+1} of {epochs} took {time.time() - start_time:.2f}s\n")

# You can further include more detailed logging, saving model checkpoints, and more based on your requirements.

