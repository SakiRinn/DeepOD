# -*- coding: utf-8 -*-
"""Utility functions for manipulating data
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause

from typing import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F


def _generate_data(n_inliers, n_outliers, n_features, coef, offset,
                   random_state, n_nan=0, n_inf=0):
    """Internal function to generate data samples.

    Parameters
    ----------
    n_inliers : int
        The number of inliers.

    n_outliers : int
        The number of outliers.

    n_features : int
        The number of features (dimensions).

    coef : float in range [0,1)+0.001
        The coefficient of data generation.

    offset : int
        Adjust the value range of Gaussian and Uniform.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_nan : int
        The number of values that are missing (np.NaN). Defaults to zero.

    n_inf : int
        The number of values that are infinite. (np.infty). Defaults to zero.

    Returns
    -------
    X : numpy array of shape (n_train, n_features)
        Data.

    y : numpy array of shape (n_train,)
        Ground truth.
    """

    inliers = coef * random_state.randn(n_inliers, n_features) + offset
    outliers = random_state.uniform(low=-1 * offset, high=offset,
                                    size=(n_outliers, n_features))
    X = np.r_[inliers, outliers]

    y = np.r_[np.zeros((n_inliers,)), np.ones((n_outliers,))]

    if n_nan > 0:
        X = np.r_[X, np.full((n_nan, n_features), np.NaN)]
        y = np.r_[y, np.full((n_nan), np.NaN)]

    if n_inf > 0:
        X = np.r_[X, np.full((n_inf, n_features), np.infty)]
        y = np.r_[y, np.full((n_inf), np.infty)]

    return X, y


def generate_data(n_train=1000, n_test=500, n_features=2, contamination=0.1,
                  train_only=False, offset=10,
                  random_state=None, n_nan=0, n_inf=0):
    """Utility function to generate synthesized data.
    Normal data is generated by a multivariate Gaussian distribution and
    outliers are generated by a uniform distribution.
    "X_train, X_test, y_train, y_test" are returned.

    Parameters
    ----------
    n_train : int, (default=1000)
        The number of training points to generate.

    n_test : int, (default=500)
        The number of test points to generate.

    n_features : int, optional (default=2)
        The number of features (dimensions).

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    train_only : bool, optional (default=False)
        If true, generate train data only.

    offset : int, optional (default=10)
        Adjust the value range of Gaussian and Uniform.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_nan : int
        The number of values that are missing (np.NaN). Defaults to zero.

    n_inf : int
        The number of values that are infinite. (np.infty). Defaults to zero.

    Returns
    -------
    X_train : numpy array of shape (n_train, n_features)
        Training data.

    X_test : numpy array of shape (n_test, n_features)
        Test data.

    y_train : numpy array of shape (n_train,)
        Training ground truth.

    y_test : numpy array of shape (n_test,)
        Test ground truth.

    """

    # initialize a random state and seeds for the instance
    random_state = check_random_state(random_state)
    offset_ = random_state.randint(low=offset)
    coef_ = random_state.random_sample() + 0.001  # in case of underflow

    n_outliers_train = int(n_train * contamination)
    n_inliers_train = int(n_train - n_outliers_train)

    X_train, y_train = _generate_data(n_inliers_train, n_outliers_train,
                                      n_features, coef_, offset_, random_state,
                                      n_nan, n_inf)

    if train_only:
        return X_train, y_train

    n_outliers_test = int(n_test * contamination)
    n_inliers_test = int(n_test - n_outliers_test)

    X_test, y_test = _generate_data(n_inliers_test, n_outliers_test,
                                    n_features, coef_, offset_, random_state,
                                    n_nan, n_inf)

    return X_train, X_test, y_train, y_test


def add_irrelevant_features(x, ratio, seed=None):
    n_samples, n_f = x.shape
    size = int(ratio * n_f)

    irr_new = np.zeros([n_samples, size])
    np.random.seed(seed)
    for i in tqdm(range(size)):
        irr_new[:, i] = np.random.rand(n_samples)

    # irr_new = np.zeros([n_samples, size])
    # np.random.seed(seed)
    # for i in range(size):
    #     array = x[:, np.random.choice(x.shape[1], 1)]
    #     new_array = array[np.random.permutation(n_samples)].flatten()
    #     irr_new[:, i] = new_array

    x_new = np.hstack([x, irr_new])

    return x_new


def adjust_contamination(x, y, contamination_r, swap_ratio=0.05, random_state=42):
    """
    add/remove anomalies in training data to replicate anomaly contaminated data sets.
    randomly swap 5% features of two anomalies to avoid duplicate contaminated anomalies.
    """
    rng = np.random.RandomState(random_state)

    anom_idx = np.where(y == 1)[0]
    norm_idx = np.where(y == 0)[0]
    n_cur_anom = len(anom_idx)
    n_adj_anom = int(len(norm_idx) * contamination_r / (1. - contamination_r))

    # x_train = np.delete(x_train, unknown_anom_idx, axis=0)
    # y_train = np.delete(y_train, unknown_anom_idx, axis=0)
    # noises = inject_noise(true_anoms, n_adj_noise, 42)
    # x_train = np.append(x_train, noises, axis=0)
    # y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

    # inject noise
    if n_cur_anom < n_adj_anom:
        n_inj_noise = n_adj_anom - n_cur_anom
        print(f'Control Contamination Rate: injecting [{n_inj_noise}] Noisy samples')

        seed_anomalies = x[anom_idx]

        n_sample, dim = seed_anomalies.shape
        n_swap_feat = int(swap_ratio * dim)
        inj_noise = np.empty((n_inj_noise, dim))
        for i in np.arange(n_inj_noise):
            idx = rng.choice(n_sample, 2, replace=False)
            o1 = seed_anomalies[idx[0]]
            o2 = seed_anomalies[idx[1]]
            swap_feats = rng.choice(dim, n_swap_feat, replace=False)
            inj_noise[i] = o1.copy()
            inj_noise[i, swap_feats] = o2[swap_feats]

        x = np.append(x, inj_noise, axis=0)
        y = np.append(y, np.ones(n_inj_noise))

    # remove noise
    elif n_cur_anom > n_adj_anom:
        n_remove = n_cur_anom - n_adj_anom
        print(f'Control Contamination Rate: Removing [{n_remove}] Noise')

        remove_id = anom_idx[rng.choice(n_cur_anom, n_remove, replace=False)]
        print(x.shape)

        x = np.delete(x, remove_id, 0)
        y = np.delete(y, remove_id, 0)
        print(x.shape)

    return x, y


def label_encoding(array):
        ordered_set = list(OrderedDict.fromkeys(array).keys())
        encoding_map = {elem: idx for idx, elem in enumerate(ordered_set)}
        label = np.zeros(array.shape, dtype=np.int32)
        for k, v in encoding_map.items():
            label[array == k] = v
        new_array = F.one_hot(torch.tensor(label).type(torch.long)).numpy()
        return encoding_map, new_array
