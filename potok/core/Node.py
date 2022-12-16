import copy
import dill
from pathlib import Path
from typing import List, Tuple, Dict

from potok.core.Data import Data, DataDict


class Serializable:
    def __init__(self, **kwargs):
        if bool(kwargs) and ('name' in kwargs):
            self.name = kwargs['name']
        else:
            self.name = self.__class__.__name__

    def _restate_(self):
        # self.__dict__[key] = None
        pass

    def _save_(self, prefix: Path = None):
        pass

    def _load_(self, prefix: Path = None):
        pass

    def save(self, prefix: Path = None):
        prefix.mkdir(parents=True, exist_ok=True)
        self._save_(prefix)
        self._restate_()
        file_name = prefix / (self.name + '.dill')
        with open(file_name, "wb") as dill_file:
            dill.dump(self, dill_file)

    def load(self, prefix: Path = None):
        file_name = prefix / (self.name + '.dill')
        with open(file_name, "rb") as dill_file:
            instance = dill.load(dill_file)
            self.__setstate__(instance.__dict__)
        self._load_(prefix)

    # @classmethod
    # def load(cls, prefix: Path = None):
    #     TODO: по факту этот подход лучше, так как в теории не зависит от иницилизации пайплайна
    #     file_name = prefix / (cls.__name__ + '.dill')
    #     with open(file_name, "rb") as dill_file:
    #         instance = dill.load(dill_file)
    #     instance._load_(prefix)
    #     return instance
    
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state: Dict):
        self.__dict__.update(state)
        return


class Node(Serializable):
    def fit(self, x: Data, y: Data) -> Tuple[Data, Data]:
        return x, y

    def predict_forward(self, x: Data) -> Data:
        return x
    
    def predict_backward(self, y_frwd: Data) -> Data:
        return y_frwd

    @property
    def copy(self) -> Node:
        return copy.copy(self)

    def __str__(self) -> str:
        return self.name


class Operator(Node):
    def x_forward(self, x: Data) -> Data:
        return x

    def y_forward(self, y: Data, x: Data = None, x_frwd: Data = None) -> Data:
        return y

    def y_backward(self, y_frwd: Data) -> Data:
        return y_frwd

    def _fit_(self, x: Data, y: Data):
        return None

    def fit(self, x: Data, y: Data) -> Tuple[Data, Data]:
        self._fit_(x, y)
        x_frwd = self.x_forward(x)
        y_frwd = self.y_forward(y, x, x_frwd)
        return x_frwd, y_frwd

    def predict_forward(self, x: Data) -> Data:
        x_frwd = self.x_forward(x)
        return x_frwd

    def predict_backward(self, y_frwd: Data) -> Data:
        y = self.y_backward(y_frwd)
        return y


class Function(Node):
    # Not parallelizable
    def __init__(self, leaf, **kwargs):
        super().__init__(**kwargs)
        self.leaf = leaf
    # TODO: переопределить save and load можно сохранять leaf отдельно, примеры func это Early stopping, HyperOpt, Trainer


class Regressor(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _transform_forward_(self):
        pass

    def _transform_backward_(self):
        pass

    def _fit_(self, x: Data, y: Data):
        return None

    def _predict_(self, x: Data) -> Data:
        return x

    def fit(self, x: Data, y: Data) -> Tuple[Data, Data]:
        self._fit_(x, y)
        y_frwd = self._predict_(x)
        return x, y_frwd

    def predict_forward(self, x: Data) -> Data:
        y_frwd = self._predict_(x)
        return y_frwd
