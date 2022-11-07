import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self) -> None:
        # data loading
        super().__init__()

    def __getitem__(self, index) -> T_co:
        # dataset[0]
        return super().__getitem__(index)
    
    def __len__(self):
        # len(dataset)

