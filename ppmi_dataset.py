import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import absolute_discretization, Zscore_normalization
import ants
import pdb
import pandas as pd
import re
from datetime import datetime
import numpy as np
from einops import rearrange
from torchvision.transforms import functional as F


class PPMIDataset(Dataset):
    def __init__(self, root_dir, split_type='train', split=0.2):
        """
        Args:
            root_dir (string): Directory with all the patients' folders.
        """
        self.root_dir = root_dir
        self.patients = os.listdir(root_dir)
        self.scans = []
        
        
        for patient in self.patients:
            patient_dir = os.path.join(root_dir, patient, 'T1-anatomical')
            visits = os.listdir(patient_dir)
            visits.sort()  # Sort to ensure chronological order
            for visit in visits:
                visit_dir = os.path.join(patient_dir, visit)
                uuid_folders = os.listdir(visit_dir)
                for uuid_folder in uuid_folders:
                    scan_path = os.path.join(visit_dir, uuid_folder)
                    scan_files = [f for f in os.listdir(scan_path) if f.endswith('.nii')]
                    for scan_file in scan_files:
                        full_scan_path = os.path.join(scan_path, scan_file)
                        self.scans.append(full_scan_path)
                break # consider only case of 1 visit per patient
            
        df = pd.read_csv("./dataset/PPMI_Curated_Data_Cut_Public_20230612_rev.csv")
        self.scans, self.labels = self.get_labels(df)
        sp_idx = int(len(self.scans)*split)
        if split_type == 'train':
            self.scans = self.scans[sp_idx:]
            self.labels = self.labels[sp_idx:]
        else:
            self.scans = self.scans[:sp_idx]
            self.labels = self.labels[:sp_idx]
            
            
            
        
    
    def get_labels(self, df):
        data_extracted = []
        for i, path in enumerate(self.scans):
            patno_match = re.search(r'/PPMI/(\d+)/', path)
            date_match = re.search(r'/(\d{4})-(\d{2})-(\d{2})_', path)
            if patno_match and date_match:
                patno = int(patno_match.group(1))
                date_str = f"{date_match.group(1)}-{date_match.group(2)}"
                # Convert to datetime to reformat
                date_dt = datetime.strptime(date_str, '%Y-%m')
                # Format date to match 'Feb-23' style
                formatted_date = date_dt.strftime('%b%Y')
                data_extracted.append({'PATNO': patno, 
                                    'visit_date': formatted_date.upper(), 
                                    'row_id': i})
                
        # Create DataFrame from extracted data
        df_paths = pd.DataFrame(data_extracted)
        df_joined = pd.merge(df, df_paths, on=['PATNO', 'visit_date'], how='inner')
        select_cols = [
        'row_id', 
        'NHY']
        df_select = df_joined[select_cols]
        df_select = df_select[df_select.NHY != '.'] # this get rids of 13 examples
        scans_select = [self.scans[row_id] for row_id in df_select['row_id']]
        labels_select = df_select['NHY'].to_numpy(dtype=np.int64)
        return scans_select, labels_select 
        

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan_path = self.scans[idx]
        scan = ants.image_read(scan_path)
        # normalize mri (subtract mean and divide by std deviation)
        scan = Zscore_normalization(scan.numpy())
        # TODO: check if you what to remove skull later
    
        # Convert ANTsImage to a PyTorch tensor
        scan_tensor = torch.tensor(scan, dtype=torch.float)
        # H x W x C
        scan_tensor = rearrange(scan_tensor, "H W C -> C H W")
        # Ensure the scan has 256 channels, pad if necessary
        n_channels = scan_tensor.size(0)  # Assuming the channel dimension is the first
        if n_channels < 256:
            # Calculate how many channels to pad
            n_pad = 256 - n_channels
            # Pad the scan tensor. Assuming the format is CHW (channels, height, width).
            # Adjust (0, 0, 0, 0) if padding is needed in other dimensions (like height and width).
            pad = (0, 0, 0, 0, 0, n_pad)
            scan_tensor = torch.nn.functional.pad(scan_tensor, pad, "constant", 0) 
            
        return scan_tensor, self.labels[idx] # size -> [176, 240, 256]
    
    
    
def custom_collate_fn(batch):
    # Separate images and labels
    images, labels = zip(*batch)
    
    # Find the maximum width and height in the batch
    max_width = max([img.shape[2] for img in images])  # Width is the 3rd dimension
    max_height = max([img.shape[1] for img in images])  # Height is the 2nd dimension

    # Pad the images
    padded_images = []
    for img in images:
        left = (max_width - img.shape[2]) // 2
        right = max_width - img.shape[2] - left
        top = (max_height - img.shape[1]) // 2
        bottom = max_height - img.shape[1] - top
        padded_img = torch.nn.functional.pad(img, (left, right, top, bottom), "constant", 0)
        padded_images.append(padded_img)

    # Stack images and labels into single tensors
    batch_images = torch.stack(padded_images)
    batch_labels = torch.tensor(labels)
    return batch_images, batch_labels