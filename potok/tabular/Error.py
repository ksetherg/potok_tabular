import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import r2_score

from potok.core import ApplyToDataDict


class Error:
    def __init__(self, name):
        self.name = name

    @ApplyToDataDict(mode='all')
    def get_error(self, y, p, x=None, **kwargs):
        target = y.target
        if not isinstance(y, (pd.DataFrame, np.ndarray)):
            y = y.data[target]
            p = p.data

        y2 = y.to_numpy()
        p2 = p.to_numpy()

        if self.name == 'MAE':
            diff = np.abs(y2 - p2)
            return diff.mean(axis=0)[0], diff.std(axis=0)[0]

        # elif self.name == 'MSE':
        #     diff = np.square(y2 - p2)
        #     return diff.mean(axis=0)[0], diff.std(axis=0)[0]
        #
        # elif self.name == 'R2':
        #     return r2_score(y2, p2), 0
        #
        # elif self.name == 'BinaryLogloss':
        #     epsilon = 1e-15
        #     p2 = np.clip(p2.to_numpy()[:, 1:2], epsilon, 1 - epsilon)
        #     y2 = y2.to_numpy()
        #     diff = -(y2 * np.log(p2) + (1 - y2) * np.log(1 - p2))
        #     return diff.mean(axis=0), diff.std(axis=0)
        #
        # elif self.name == 'MultiLogLoss':
        #     labels = kwargs['labels']
        #
        #     epsilon = 1e-15
        #     p2 = np.clip(p2.to_numpy(), epsilon, 1 - epsilon)
        #
        #     lb = LabelBinarizer()
        #     lb.fit(labels)
        #     y2 = lb.transform(y2)
        #
        #     diff = -(y2 * np.log(p2)).sum(axis=1)
        #
        #     return diff.mean(axis=0), diff.std(axis=0)
        #
        # else:
        #     raise Exception('Unknown error name')
