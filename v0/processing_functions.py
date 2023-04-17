"""
define functions to be used in model
"""
from sklearn.preprocessing import StandardScaler
import numpy as np


# scale original data to avoid overflow, eg. exp(100) will be too large to be handled
def softmin(x):
    scaler = StandardScaler()
    x = [-i for i in x]
    x = np.reshape(x, [-1, 1])
    x = scaler.fit_transform(x)

    return np.exp(x) / np.sum(np.exp(x))


def softmax(x):
    scaler = StandardScaler()
    x = np.reshape(x, [-1, 1])
    x = scaler.fit_transform(x)

    return np.exp(x) / np.sum(np.exp(x))
