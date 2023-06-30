from operator import is_
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import OrderedDict
import torch.nn.functional as F


class BaseDataset(Dataset):

    VALIDARE_SPLIT = 0.3

    def __init__(self, data, gts):
        super(BaseDataset, self).__init__()
        self.data = data
        self.gts = gts

    def __len__(self):
        if self.data.shape[0] != self.gts.shape[0]:
            raise ValueError("The length of the data and the labels doesn't match!")
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.gts[idx]

    @staticmethod
    def label_encoding(array):
        ordered_set = list(OrderedDict.fromkeys(array).keys())
        encoding_map = {elem: idx for idx, elem in enumerate(ordered_set)}
        label = np.zeros(array.shape, dtype=np.int32)
        for k, v in encoding_map.items():
            label[array == k] = v
        new_array = F.one_hot(torch.tensor(label).type(torch.long)).numpy()
        return encoding_map, new_array

    def split(self, is_training, is_shuffle=False):
        if is_training is None:
            return
        split_idx = int(len(self) * (1 - self.VALIDARE_SPLIT))
        if is_shuffle:
            idxs = np.arange(len(self))
            np.random.shuffle(idxs)
            if is_training:
                self.data = self.data[idxs[:split_idx]]
                self.gts = self.gts[idxs[:split_idx]]
            else:
                self.data = self.data[idxs[split_idx:]]
                self.gts = self.gts[idxs[split_idx:]]
        else:
            if is_training:
                self.data = self.data[:split_idx]
                self.gts = self.gts[:split_idx]
            else:
                self.data = self.data[split_idx:]
                self.gts = self.gts[split_idx:]
