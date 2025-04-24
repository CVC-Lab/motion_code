from torch.utils.data import Dataset, DataLoader, default_collate, random_split
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np


def exists(x):
    return x is not None

class MultiPhaseDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        n_phases = 2,
        n_images = 10
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')][:n_images]

        self.transform = T.Compose([
            T.Grayscale(),
            T.ToTensor(),
        ])
        self.n_phases = n_phases

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        gray_img_tensor = self.transform(img)
        min_val = gray_img_tensor.min()
        max_val = gray_img_tensor.max()
        gray_img_tensor = 2 * (gray_img_tensor - min_val) / (max_val - min_val) - 1
        meta_path = str(path)[:-4] + '.txt'
        with open(meta_path, "r") as f:
            lines = f.readlines()

        # Parse metadata (lazy two phase setup)
        phase_meta = []
        for i in range(self.n_phases):
            phase_meta.append(np.array(list(map(float, lines[i].split()))))

        #Parse number of data points
        num_points = int(lines[self.n_phases])

        #Parse the coordinates into a NumPy array of shape (num_points, 2)
        data_points = np.array([
            list(map(float, line.split()))
            for line in lines[(self.n_phases + 1):(self.n_phases + 1 + num_points)]
        ])
        #TODO: radius is not a fixed number when generalize
        meta_data = {
            "phases": phase_meta,
            "num_clsutered_grains": num_points,
            "radius": 25.0,
            "clustered_center": data_points
        }
        return gray_img_tensor, meta_data

class RowDataset(Dataset):
    def __init__(self, image_dataset, rows_per_image=1):
        self.image_dataset = image_dataset
        self.rows_per_image = rows_per_image  # Number of rows to sample from each image

    def __len__(self):
        return len(self.image_dataset) * self.rows_per_image

    def __getitem__(self, idx):
        image_idx = idx // self.rows_per_image
        img, img_meta = self.image_dataset[image_idx] # shape: [C, H, W]
        size = img.shape[1]
        height = img_meta["phases"][0][0]
        r = img_meta["radius"]
        #row_idx = torch.randint(0, size - int(height + height // 2), (1,)).item()
        row_idx = (idx % self.rows_per_image) * (size // self.rows_per_image)
        row_meta = {
            "label": np.where(row_idx % height <= r or row_idx % height >= (height - r), 1, 0) # 0: clean, 1: noisy
        }
        row = img[:, row_idx, :]  # shape: [C, W]
        return row, row_meta

def row_collate_fn(batch):
    rows, metas = zip(*batch)
    rows = torch.stack(rows, dim=0)  # Stack row tensors
    rows = rows.permute(0, 2, 1)
    # Metadata stays as list of dicts (or customize further)
    metas = [torch.tensor(meta["label"], dtype=torch.int) for meta in metas]
    metas = torch.stack(metas, dim=0)
    return rows, metas

if __name__=="__main__":
    #data_path = '/mnt/data/yiwang/code/RandomMaterial/clustered_grain/massive_03_different_noise_levels_with_cluster'
    data_path = '/mnt/data/yiwang/code/RandomMaterial/clustered_grain/massive_05_testify_bilevel_sampling_2phases_with_clustered_grains'
    material_dataset = MultiPhaseDataset(data_path, [512,512], n_phases=2)
    img , label = material_dataset[0]
    print(label)
    plt.imshow(img.detach().numpy().squeeze())