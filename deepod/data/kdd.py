from typing import OrderedDict
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from deepod.data.base import BaseDataset
import deepod.utils as utils


class KDD(BaseDataset):

    def __init__(self, path, is_training=None, is_shuffle=False, normalize=None):
        """Support KDD Cup'99 and NSL-KDD dataset.

        Args:
            path (str): Path to the raw data.
            is_training (bool, optional): If True or False, split the dataset into
            train set and test set. Defaults to None, no split.
            is_shuffle (bool): Whether to scramble the order when splitting.
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
        data = np.concatenate(
            [table.values[:, 0][..., np.newaxis], table.values[:, 4:41]], axis=1
        )
        if table.shape[1] == 43:
            self.features.append("difficulty_level")
            self.difficulty = table.values[:, 42]

        # Protocol Type
        self.protocol, protocols = utils.label_encoding(table.values[:, 1])
        data = np.concatenate([data, protocols], axis=1)
        # Service
        self.service, services = utils.label_encoding(table.values[:, 2])
        data = np.concatenate([data, services], axis=1)
        # Flag
        self.flag, flags = utils.label_encoding(table.values[:, 3])
        data = np.concatenate([data, flags], axis=1)
        # GT
        original_gts = np.vectorize(lambda s: s.strip(". "))(table.values[:, 41])
        ordered_set = list(OrderedDict.fromkeys(original_gts).keys())
        self.type = {elem: idx for idx, elem in enumerate(ordered_set)}
        gts = np.zeros(original_gts.shape, dtype=np.int32)
        for k, v in self.type.items():
            gts[original_gts == k] = v

        super(KDD, self).__init__(data.astype(np.float32), gts)
        self.split(is_training, is_shuffle)
        if normalize is not None:
            self.normalize(normalize)
