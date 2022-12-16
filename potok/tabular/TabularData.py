import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

from potok.core import Data


@dataclass(init=False)
class TabularData(Data):
    data: pd.DataFrame

    def __init__(self, 
                 data: pd.DataFrame,
                 target: list = None, 
                 ):
        assert isinstance(target, list) and isinstance(data, pd.DataFrame), 'Invalid input type.'
        self.data = data
        self.target = target

    def __getitem__(self, cols: List[str]) -> 'TabularData':
        assert isinstance(cols, list), 'Invalid key type.'
        sub_df = self.data[cols]
        new = self.copy(data=sub_df)
        return new

    def __len__(self) -> int:
        return len(self.data)

    @property
    def X(self) -> 'TabularData':
        columns = [col for col in self.data.columns if col not in self.target]
        X = self.copy(data=self.data[columns])
        return X

    @property
    def Y(self) -> 'TabularData':
        columns = [col for col in self.data.columns if col in self.target]
        y = self.copy(data=self.data[columns])
        return y

    @property
    def index(self):
        return self.data.index

    def get_by_index(self, index) -> 'TabularData':
        chunk = self.data.loc[self.data.index.isin(index)]
        new = self.copy(data=chunk)
        return new

    def reindex(self, index) -> 'TabularData':
        df = self.data.reindex(index)
        new = self.copy(data=df)
        return new

    @staticmethod
    def combine(datas: List['TabularData']) -> 'TabularData':
        dfs = [data.data for data in datas]
        df_cmbn = pd.concat(dfs, axis=1, keys=range(len(dfs)))
        df_cmbn = df_cmbn.groupby(level=[1], axis=1).mean()
        new = datas[0].copy(data=df_cmbn)
        return new
