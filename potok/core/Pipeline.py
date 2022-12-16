# import copy
from potok.core.Data import Data, DataDict
from potok.core.Node import Node, Operator
from potok.core.Layer import Layer

import gc
from pathlib import Path
from time import gmtime, strftime
from typing import List, Tuple, Dict


class Pipeline(Node):
    """Pipeline works with DataLayer and Layer"""
    def __init__(self, nodes: List[Node], **kwargs: Dict):
        super().__init__(**kwargs)
        _nodes_ = []
        for node in nodes:
            if isinstance(node, (Node, Operator)):
                _nodes_.append(node)
            else:
                raise Exception(f'Unknown node type = {type(node)}')
            # if isinstance(node, Pipeline):
                # _nodes_.extend(node.nodes)
                
        self.nodes = _nodes_
        self.layers = None
        # Todo: решить проблему с шейпами, хорошо бы их генерировать автоматом
        self.shapes = kwargs['shapes']
        assert len(self.shapes) == len(self.nodes), 'Data and nodes shapes do not match.'

        # self.current_fit = 0
        # self.current_predict = 0

    def _compile_(self):
        layers = []
        for node, num in zip(self.nodes, self.shapes):
            layer = Layer(*[node.copy for _ in range(num)])
            layers.append(layer)
        self.layers = layers
        return None
        
    def save(self, prefix: Path = None):
        if self.layers is None:
            raise Exception('Fit your model before.')
    
        suffix = strftime("%y_%m_%d_%H_%M_%S", gmtime())
        pipeline_name = self.name + suffix
        for i, layer in enumerate(self.layers):
            suffix_lyr = layer.name + '_' + str(i)
            prefix_lyr = prefix / pipeline_name / suffix_lyr
            layer.save(prefix_lyr)
        return None

    def load(self, prefix: Path = None):
        self._compile_()
        for i, layer in enumerate(self.layers):
            suffix_lyr = layer.name + '_' + str(i)
            prefix_lyr = prefix / suffix_lyr
            layer.load(prefix_lyr)
        return None
    
    def fit(self, x: DataDict, y: DataDict) -> Tuple[DataDict, DataDict]:
        self._compile_()
        for layer in self.layers:
            assert len(x) == len(y) == len(layer), 'Invalid shapes.'
            x, y = layer.fit(x, y)
        return x, y
    
    def predict_forward(self, x: DataDict) -> DataDict:
        if self.layers is None:
            raise Exception('Fit your model before.')
        for layer in self.layers:     
            x = layer.predict_forward(x)
        return x
    
    def predict_backward(self, y: DataDict) -> DataDict:
        if self.layers is None:
            raise Exception('Fit your model before.')
        for layer in self.layers[::-1]:
            y = layer.predict_backward(y)
        return y

    def predict(self, x: DataDict) -> DataDict:
        y2 = self.predict_forward(x)
        y1 = self.predict_backward(y2)
        return y1

    def fit_predict(self, x: DataDict, y: DataDict) -> DataDict:
        x2, y2 = self.fit(x, y)
        y1 = self.predict_backward(y2)
        # y1 = self.predict(x)
        return y1
    
    def __str__(self) -> str:
        pipe = f'({self.name}: '
        for node in self.nodes:
            pipe += str(node)
            if node != self.nodes[-1]:
                pipe += ' -> '
        pipe += ')'
        return pipe
        
    # def fit_step(self, x, y):
    #     self.current_fit += 1
    #     assert self.current_fit <= len(self.nodes)
    #     x2, y2 = self._fit_until_(x, y)
    #     return x2, y2
    
    # def _fit_until_(self, x, y):
    #     i = self.current_fit
    #     assert i >= 0
    #     layers = []
    #     for node in self.nodes:
    #         assert len(x) == len(y)
    #         layer = self.next_layer(node, len(x))
    #         x, y = layer.fit(x, y)
    #         layers.append(layer)
    #     self.layers = layers
    #     return x, y
