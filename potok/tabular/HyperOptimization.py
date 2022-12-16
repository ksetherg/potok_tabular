# from pprint import pprint
# from typing import Union
# from collections import Mapping, Set, Sequence
#
# from ax import ParameterType, RangeParameter, ChoiceParameter, SearchSpace, SimpleExperiment, modelbridge
#
# from potok.core import Function
#
#
# class HrPrmOptRange:
#     def __init__(self, value: float, mn: float, mx: float):
#         self.value = value
#         self.min = mn
#         self.max = mx
#
#
# class HrPrmOptChoise:
#     def __init__(self, value: Union[str, int, bool], choises: list):
#         self.value = value
#         self.choises = choises
#
#
# class HyperParamOptimization(Function):
#     def __init__(self, leaf, error, n_iter, minimize=True, **kwargs):
#         super().__init__(leaf, **kwargs)
#         self.n_iter = n_iter
#         self.error = error
#         self.minimize = minimize
#
#         self._best_values_ = None
#
#     def fit(self, x, y):
#         opt_range_dict = DeepSearch(HrPrmOptRange).get_places(self.leaf)
#         opt_choise_dict = DeepSearch(HrPrmOptChoise).get_places(self.leaf)
#
#         types = {
#             str: ParameterType.STRING,
#             int: ParameterType.INT,
#             bool: ParameterType.BOOL,
#             float: ParameterType.FLOAT
#         }
#         opt_rng_list = [RangeParameter(name=k, parameter_type=types[type(v.value)], lower=v.min, upper=v.max)
#                         for k, v in opt_range_dict.items()]
#
#         opt_chs_list = [ChoiceParameter(name=k, parameter_type=types[type(v.value)], values=v.choises)
#                         for k, v in opt_choise_dict.items()]
#
#         search_space = SearchSpace(parameters=opt_rng_list + opt_chs_list)
#         opt_dict = {**opt_range_dict, **opt_choise_dict}
#
#         error_values = []
#         opt_params_values = []
#
#         def evaluate(parameterization):
#             for k, v in parameterization.items():
#                 """переопределиние параметров"""
#                 opt_dict[k].value = v
#
#             x2, y2 = self.leaf.fit(x, y)
#
#             error_dict = self._get_error_(y, y2, x)
#             error_values.append(error_dict['valid'])
#             opt_params_values.append(parameterization)
#
#             return {'train': error_dict['valid']}
#
#         exp = SimpleExperiment(
#             name='Hyper Parametrs Optimization',
#             search_space=search_space,
#             evaluation_function=evaluate,
#             minimize=self.minimize,
#             objective_name='train',
#         )
#
#         sobol = modelbridge.get_sobol(search_space=search_space)
#
#         for _ in range(self.n_iter):
#             exp.new_trial(generator_run=sobol.gen(1))
#         for i in range(self.n_iter):
#             # print(f'Running GP+EI optimization trial {i+1}/{self.n_iter}')
#             gpei = modelbridge.get_GPEI(experiment=exp, data=exp.eval())
#             batch = exp.new_trial(generator_run=gpei.gen(1))
#
#         # dat = exp.eval()
#         final_errors = sorted(zip(error_values, opt_params_values), key=lambda k: k[0])
#         if self.minimize:
#             self._best_values_ = final_errors[0]
#         else:
#             self._best_values_ = final_errors[-1]
#
#         print('Iteration Errors:')
#         pprint(exp.eval().df)
#         print('Best result:')
#         pprint(self._best_values_)
#
#         for k, v in self._best_values_[1].items():
#             opt_dict[k].value = v
#         p = self.leaf.fit(x, y)
#         return p
#
#     def _get_error_(self, y, p, x):
#         errors = self.error.get_error(y, p, x)
#         return errors
#
#
# class DeepSearch:
#     def __init__(self, target_class):
#         self.target_class = target_class
#
#     def get_places(self, obj):
#         objects = self.walk_through(obj)
#         return dict(objects)
#
#     def walk_through(self, obj, path=None):
#         if path is None:
#             path = ''
#         if isinstance(obj, self.target_class):
#             yield path, obj
#
#         if isinstance(obj, Mapping) or hasattr(obj, '__dict__'):
#             iterator = lambda x: getattr(x, 'items')()
#         elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, (str, bytes)):
#             iterator = enumerate  # lambda x: getattr(x, '__iter__')()
#         else:
#             iterator = None
#
#         if iterator:
#             iterable = obj.__dict__ if hasattr(obj, '__dict__') else obj
#             for k, v in iterator(iterable):
#                 for req in self.walk_through(v, path + str(k) + '_'):
#                     yield req
