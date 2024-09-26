from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch

def data_spliter(dataset, ratio):
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size,test_size])
    return train_data, test_data
