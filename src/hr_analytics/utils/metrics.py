import numpy as np


def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return 100 * np.mean(
        np.abs(y_true - y_pred)
        / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    )