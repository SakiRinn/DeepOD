from typing import OrderedDict
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F


class KDD(Dataset):

    VALIDARE_SPLIT = 0.2

    def __init__(self, path, is_training=None):
        """Support KDD Cup'99 and NSL-KDD dataset.

        Args:
            path (str): Path to the raw data.
            is_training (bool, optional): If True or False, split the dataset into
            train set and test set. Defaults to None, no split.
        """
        self.features = [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
            "label",
        ]
        table = pd.read_table(path, header=None, sep=",", on_bad_lines="warn")
        self.data = np.concatenate([table.values[:, 0][..., np.newaxis], table.values[:, 4:41]], axis=1)
        if table.shape[1] == 43:
            self.features.append("difficulty_level")
            self.difficulty = table.values[:, 42]

        # Protocol Type
        self.protocol, protocols = self.label_encoding(table.values[:, 1])
        self.data = np.concatenate([self.data, protocols], axis=1)
        # Service
        self.service, services = self.label_encoding(table.values[:, 2])
        self.data = np.concatenate([self.data, services], axis=1)
        # Flag
        self.flag, flags = self.label_encoding(table.values[:, 3])
        self.data = np.concatenate([self.data, flags], axis=1)
        # GT
        gts = np.vectorize(lambda s: s.strip(". "))(table.values[:, 41])
        ordered_set = list(OrderedDict.fromkeys(gts).keys())
        self.type = {elem: idx for idx, elem in enumerate(ordered_set)}
        self.gts = np.zeros(gts.shape, dtype=np.int32)
        for k, v in self.type.items():
            self.gts[gts == k] = v

        self.data = self.data.astype(np.float32)

        if is_training is not None:
            np.random.shuffle(idxs)
            # if is_training:
            #     self.data = self.data[:split_idx]
            #     self.gts = self.gts[:split_idx]
            # else:
            #     self.data = self.data[split_idx:]
            #     self.gts = self.gts[split_idx:]
            split_idx = int(len(self) * (1 - self.VALIDARE_SPLIT))
            idxs = np.arange(len(self))
            if is_training:
                self.data = self.data[idxs[:split_idx]]
                self.gts = self.gts[idxs[:split_idx]]
            else:
                self.data = self.data[idxs[split_idx:]]
                self.gts = self.gts[idxs[split_idx:]]


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


def get_KDDCup99(args, data_dir):
    """Returning train and test dataloaders."""
    train = KDD(data_dir, True)
    dataloader_train = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    test = KDD(data_dir, False)
    dataloader_test = DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    return dataloader_train, dataloader_test


if __name__ == "__main__":
    kdd = KDD("./data/kddcup.data.txt", is_training=True)
    print(kdd.data.shape)
