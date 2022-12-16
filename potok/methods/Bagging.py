from typing import List, Iterator, Tuple
import copy

from potok.core import Operator, DataDict


class Bagging(Operator):
    def __init__(self, n_iter: int, **kwargs):
        super().__init__(**kwargs)
        self.n_iter = n_iter
        self.index = None

    def x_forward(self, x: DataDict) -> DataDict:
        self.index = x.index
        x2 = self._repeat_(x)
        return x2

    def y_forward(self, y: DataDict, x: DataDict = None, x_frwd: DataDict = None) -> DataDict:
        y2 = self._repeat_(y)
        return y2

    def y_backward(self, y_frwd: DataDict) -> DataDict:
        res = DataDict.combine(y_frwd.values())
        return res

    def _repeat_(self, data: DataDict) -> DataDict:
        out = DataDict()
        for i in range(self.n_iter):
            out[f'Bag_{i + 1}'] = data
        return out


#
# class BaggingEffective(Function):
#     def __init__(self, leaf: Node, n_iter: int, **kwargs):
#         super().__init__(leaf, **kwargs)
#         self.n_iter = n_iter
#         self.leafs = None
#
#     def fit(self, x: DataDict, y: DataDict) -> Tuple[DataDict, DataDict]:
#         # x_outs = {}
#         y_outs = {}
#         leafs = []
#         for i in range(self.n_iter):
#             print(f'Bag = {i + 1} / {self.n_iter}')
#             leaf_new = copy.deepcopy(self.leaf)
#             x2, y2 = leaf_new.fit(x, y)
#             leafs.append(leaf_new)
#             # x_outs[f'Bag_{i + 1}'] = x2
#             y_outs[f'Bag_{i + 1}'] = y2
#             # x_frwd = DataDict.combine(x_outs.values())
#             y_frwd = DataDict.combine(y_outs.values())
#         self.leafs = leafs
#         return x, y_frwd
#
#     def predict_forward(self, x: DataDict) -> DataDict:
#         y_outs = {}
#         for i, leaf in enumerate(self.leafs):
#             y = leaf.predict_forward(x)
#             y_outs[f'Bag_{i + 1}'] = y
#             y_frwd = DataDict.combine(y_outs.values())
#         return y_frwd
#
#     def predict_backward(self, y_frwd: DataDict) -> Data:
#         return y_frwd
