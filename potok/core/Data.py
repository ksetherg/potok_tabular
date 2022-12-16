from dataclasses import dataclass
from typing import List, Iterator, Tuple, Dict
import copy



@dataclass
class Data:
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: Dict):
        self.__dict__.update(state)
        return 
    
    # def __reduce__(self):
    #     return self.__class__, copy.copy(tuple(self.__dict__.values()))
        
    @property
    def X(self) -> "Data":
        raise Exception('Not implemented.')

    @property
    def Y(self) -> "Data":
        raise Exception('Not implemented.')

    @property
    def index(self):
        raise Exception('Not implemented.')

    def get_by_index(self, index) -> "Data":
        raise Exception('Not implemented')

    def reindex(self, index) -> "Data":
        raise Exception('Not implemented')

    @staticmethod
    def combine(datas: List["Data"]) -> "Data":
        raise Exception('Not implemented')
    
    def copy(self, **kwargs) -> "Data":
        new_data = copy.copy(self)
        new_data.__dict__.update(kwargs)
        return new_data


class DataDict(Data):
    def __init__(self, **kwargs):
        if kwargs:
            types = []
            for k, v in kwargs.items():
                setattr(self, k, v)
                types.append(type(v))
            assert types.count(types[0]) == len(types), 'All unit types must be the same.'
                    
    def __repr__(self) -> str:
        prefix = self.__class__.__name__ + '(' 
        body = ', '.join([f'{k}={v!r}' for k, v in self.items()])
        suffix = ')'
        return prefix + body + suffix
    
    def __len__(self) -> int:
        return len(self.__dict__)
    
    # def __iter__(self) -> Iterator:
    #     return iter(self.__dict__.items())
    
    def __getitem__(self, key: str) -> Data:
        return getattr(self, key, None)
    
    def __setitem__(self, key: str, value: Data) -> None:
        assert key not in list(self.__dict__.keys()), f'Key Error: {key}.'
        self.__setstate__({key: value})
        return

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: Dict) -> None:
        self.__dict__.update(state)
        return

    def keys(self) -> List:
        units = list(self.__dict__.keys())
        return units

    def values(self) -> List:
        return list(self.__dict__.copy().values())

    def items(self) -> Iterator:
        return iter(self.__dict__.copy().items())

    @property
    def X(self) -> "DataDict":
        X = {k: v.X for k, v in self.items()}
        return DataDict(**X)

    @property
    def Y(self) -> "DataDict":
        Y = {k: v.Y for k, v in self.items()}
        return DataDict(**Y)

    @property
    def index(self) -> "DataDict":
        indx = {k: v.index for k, v in self.items()}
        return DataDict(**indx)

    def get_by_index(self, index: "DataDict") -> "DataDict":
        assert self.keys() == index.keys(), 'Units must match.'
        res = {k1: v1.get_by_index(v2) for (k1, v1), (k2, v2) in zip(self.items(), index.items())}
        return DataDict(**res)

    def reindex(self, index: "DataDict") -> "DataDict":
        assert self.keys() == index.keys(), 'Units must match.'
        res = {k1: v1.reindex(v2) for (k1, v1), (k2, v2) in zip(self.items(), index.items())}
        return DataDict(**res)

    @staticmethod
    def combine(data: List["DataDict"]) -> "DataDict":
        # можно сделать просто как метод класса, потому что все равно комбайним поля класса
        if all([hasattr(v, 'keys') for v in data]):
            units_list = [v.keys() for v in data]
            units = sorted(set.intersection(*map(set, units_list)), key=units_list[0].index)
            assert len(units) >= 1, 'Units intersection is empty.'
            new_data = [[arg[unit] for arg in data] for unit in units]
        else:
            units = ['combined']
            new_data = data

        data_cls = new_data[0][0]
        res = {unit: data_cls.combine(new_data[i]) for i, unit in enumerate(units)}
        return DataDict(**res)
