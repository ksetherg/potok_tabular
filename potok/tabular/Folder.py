from sklearn.model_selection import KFold, train_test_split, TimeSeriesSplit
import pandas as pd
import numpy as np

from potok.core import Operator, DataDict


class Folder(Operator):
    def __init__(self,
                 n_folds: int = 5,
                 split_ratio: float = 0.2,
                 index_name: str = None,
                 seed: int = 4242,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_folds = n_folds
        self.split_ratio = split_ratio
        self.index_name = index_name
        self.seed = seed
        self.folds = None

    def _fit_(self, x: DataDict, y: DataDict) -> None:
        assert x['train'] is not None, 'Train data is required.'
        if self.index_name is not None:
            self.folds = self.generate_folds_by_index(x, y)
        else:
            self.folds = self.generate_folds(x, y)
        return None

    def generate_folds(self, x: DataDict, y: DataDict) -> dict:
        index = x['train'].index
        
        if self.n_folds > 1:
            folder = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            folds = DataDict(**{f'Fold_{i+1}': DataDict(train=index[train_idx], valid=index[valid_idx])
                                for i, (train_idx, valid_idx) in enumerate(folder.split(index))})
        else:
            train_idx, valid_idx = train_test_split(index, test_size=self.split_ratio, random_state=self.seed)
            folds = {'Fold_1': DataDict(train=index[train_idx], valid=index[valid_idx])}
        return folds

    def generate_folds_by_index(self, x: DataDict, y: DataDict) -> DataDict:
        index = x['train'].index
        values = x['train'].index.get_level_values(self.index_name).unique().to_numpy()

        if self.n_folds > 1:
            folder = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

            folds = DataDict(**{f'Fold_{i+1}':
                                DataDict(
                                    train=index[index.get_level_values(self.index_name).isin(values[train_idx])],
                                    valid=index[index.get_level_values(self.index_name).isin(values[valid_idx])])
                                for i, (train_idx, valid_idx) in enumerate(folder.split(values))})
        else:
            train_idx, valid_idx = train_test_split(values, test_size=self.split_ratio, random_state=self.seed)
            folds = DataDict(**{'Fold_1':
                         DataDict(train=index[index.get_level_values(self.index_name).isin(values[train_idx])],
                                  valid=index[index.get_level_values(self.index_name).isin(values[valid_idx])])})
        return folds

    def x_forward(self, x: DataDict) -> DataDict:
        x2 = self.get_folds(x)
        return x2

    def y_forward(self, y: DataDict, x: DataDict = None, x_frwd: DataDict = None) -> DataDict:
        y2 = self.get_folds(y)
        return y2

    def y_backward(self, y_frwd: DataDict) -> DataDict:
        if self.n_folds > 1:
            y_frwd = DataDict.combine(y_frwd.values())
        return y_frwd

    def get_folds(self, xy: DataDict) -> DataDict:
        units = xy.keys()
        if 'train' in units:
            units = [unit + f'_{i}' if unit == 'valid' else unit for i, unit in enumerate(units)]
            units.remove('train')
            valid_xy = DataDict(train=xy['train'], valid=xy['train'])
            folds = {k: valid_xy.get_by_index(v) for k, v in self.folds.items()}
            [fold.__setstate__({unit: xy[unit] for unit in units}) for k, fold in folds.items()]
        else:
            folds = {f'Fold_{i+1}': xy for i in range(self.n_folds)}
        return DataDict(**folds)


class FolderByTime(Folder):
    def __init__(self,
                 n_folds: int = 5,
                 split_ratio: float = 0.2,
                 index_name: str = None,
                 seed: int = 4242,
                 **kwargs):
        super().__init__(n_folds, split_ratio, index_name, seed, **kwargs)

    def generate_folds_by_index(self, x: DataDict, y: DataDict) -> DataDict:
        index = x['train'].index
        values = np.sort(x['train'].index.get_level_values(self.index_name).unique().to_numpy())

        folder = TimeSeriesSplit(n_splits=self.n_folds)

        folds = DataDict(**{f'Fold_{i + 1}':
                            DataDict(train=index[index.get_level_values(self.index_name).isin(values[train_idx])],
                                     valid=index[index.get_level_values(self.index_name).isin(values[valid_idx])])
                                         for i, (train_idx, valid_idx) in enumerate(folder.split(values))})

        return folds

