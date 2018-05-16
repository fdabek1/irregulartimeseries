import numpy as np


def repeat():
    pass
    # Repeat data
    # x_train = np.repeat(x_train, 15, axis=0)
    # y_train = np.repeat(y_train, 15, axis=0)


# Normalize the output to be between 0 and 1
def normalize_y(y_train, y_test):
    y_max = max(np.max(y_train), np.max(y_test))
    y_min = max(np.min(y_train), np.min(y_test))
    y_train = (y_train - y_min) / (y_max - y_min)
    y_test = (y_test - y_min) / (y_max - y_min)

    return y_train, y_test
