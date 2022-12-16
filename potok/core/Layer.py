from pathlib import Path
from typing import List, Tuple

from potok.core.Node import Node
from potok.core.Data import Data, DataDict


class Layer(Node):
    def __init__(self, *nodes, **kwargs):
        super().__init__(**kwargs)
        layer = []
        for node in nodes:
            if isinstance(node, Node):
                layer.append(node)
            else:
                raise Exception(f'Unknown node type={node.__class__.__name__}')
                
        self.layer = layer

    def __len__(self) -> int:
        return len(self.layer)

    def __getitem__(self, idx: int) -> Node:
        res = self.layer[idx]
        return res

    def save(self, prefix: Path = None):
        for i, node in enumerate(self.layer):
            suffix_nd = node.name + '_' + str(i)
            prefix_nd = prefix / suffix_nd
            node.save(prefix_nd)
    
    def load(self, prefix: Path = None):
        for i, node in enumerate(self.layer):
            suffix_nd = node.name + '_' + str(i)
            prefix_nd = prefix / suffix_nd
            node.load(prefix_nd)

    def fit(self, x: DataDict, y: DataDict):
        assert len(self.layer) == len(x) == len(y), 'Layer and data shapes must be same.'
        res = {k1: node.fit(xx, yy) for node, (k1, xx), (k2, yy) in zip(self.layer, x.items(), y.items())}
        x2 = self._flatten_forward_(DataDict(**{k: v[0] for k, v in res.items()}))
        y2 = self._flatten_forward_(DataDict(**{k: v[1] for k, v in res.items()}))
        return x2, y2
    
    def predict_forward(self, x: DataDict):
        assert len(self.layer) == len(x), 'Layer and data shapes must be same.'
        res = {k: node.predict_forward(xx) for node, (k, xx) in zip(self.layer, x.items())}
        x2 = self._flatten_forward_(DataDict(**res))
        return x2
    
    def predict_backward(self, y: DataDict):
        y2 = self._flatten_backward_(y)
        assert len(self.layer) == len(y2), 'Layer and data shapes must be same.'
        res = {k: node.predict_backward(yy) for node, (k, yy) in zip(self.layer, y2.items())}
        result = DataDict(**res)
        return result

    def _flatten_forward_(self, data: DataDict) -> DataDict:
        keys1 = data.keys()
        if isinstance(data[keys1[0]], DataDict):
            keys2 = data[keys1[0]].keys()
            if isinstance(data[keys1[0]][keys2[0]], DataDict):
                data = DataDict(**{k1 + '_' + k2: v2 for k1, v1 in data.items() for k2, v2 in v1.items()})
        return data

    def _flatten_backward_(self, data: DataDict) -> DataDict:
        if len(data) != len(self.layer):
            grouper = int(round(len(data) / len(self.layer)))
            assert grouper > 1, 'Something super wrong.'
            n_iter = int(round(len(data) / grouper))
            shaped = DataDict()
            for i in range(n_iter):
                subkeys = data.keys()[i * grouper: (i + 1) * grouper]
                subdict = {k: v for k, v in data.items() if k in subkeys}
                shaped[f'data_{i + 1}'] = DataDict(**subdict)
            return shaped
        return data

    # @staticmethod
    # def _flatten_forward_(data: DataDict) -> DataDict:
    #     units1 = data.keys
    #     if isinstance(data[units1[0]], DataDict):
    #         units2 = data[units1[0]].units
    #         if isinstance(data[units1[0]][units2[0]], DataDict):
    #             data = data[units1[0]]
    #     return data
    #
    # def _flatten_backward_(self, data: DataDict) -> DataDict:
    #     if len(data) != len(self.layer):
    #         data = DataDict(data_1=data)
    #     return data


# class Layer(Node):
#     def __init__(self, *nodes, **kwargs):
#         super().__init__(**kwargs)
#         layer = []
#         for node in nodes:
#             if isinstance(node, Node):
#                 layer.append(node.copy)
#             else:
#                 raise Exception(f'Unknown node type={node.__class__.__name__}')
                
#         self.layer = layer
#         self.shapes = None

#     def fit(self, x, y):
#         assert len(self.layer) == len(x) == len(y), 'Layer and data shapes must be same.'

#         actors = [ray.remote(node.__class__).remote(**node.__dict__) for node in self.layer]
#         res = [node.fit.remote(xx, yy) for node, xx, yy in zip(actors, x, y)]

#         states = ray.get([node.__getstate__.remote() for node in actors])
#         [node.__setstate__(state) for node, state in zip(self.layer, states)]
        
#         res = ray.get(res)

#         x2 = DataLayer(*self._flatten_forward_(list(map(lambda x: x[0], res))))
#         y2 = DataLayer(*self._flatten_forward_(list(map(lambda x: x[1], res))))
#         return x2, y2
    
#     def predict_forward(self, x):
#         assert len(self.layer) == len(x), 'Layer and data shapes must be same.'        
#         res = [ray.remote(node.predict_forward.__func__).remote(node, xx) for node, xx in zip(self.layer, x)]
#         result = ray.get(res)
#         result1d = DataLayer(*self._flatten_forward_(result))
#         return result1d
    
#     def predict_backward(self, y):
#         y2 = self._flatten_backward_(y)
#         assert len(self.layer) == len(y2), 'Layer and data shapes must be same.'
#         res = [ray.remote(node.predict_backward.__func__).remote(node, yy) for node, yy in zip(self.layer, y2)]
#         result = DataLayer(*ray.get(res))
#         return result
    
#     def _flatten_forward_(self, irr_list):
#         flat = []
#         shapes = []
#         for i in irr_list:
#             if isinstance(i, (DataLayer, list)):
#                 flat.extend(i)
#                 shapes.append(len(i))
#             elif isinstance(i, (DataUnit, Data)):
#                 flat.append(i)
#                 shapes.append(0)
#             else:
#                 raise Exception(f'Unknown type of data={i.__class__.__name__}')
#         self.shapes = shapes
#         return flat
    
#     def _flatten_backward_(self, list1d):
#         if isinstance(list1d, DataLayer):
#             list1d = list1d.to_list()
#         start = 0
#         end = 0
#         irr_list = []
#         for d in self.shapes:
#             if d == 0:
#                 sl = list1d[start]
#                 start += 1
#                 irr_list.append(sl)
#             else:
#                 end = start + d
#                 sl = DataLayer(*list1d[start:end])
#                 start = end
#                 irr_list.append(sl)
#         return irr_list