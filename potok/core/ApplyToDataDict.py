import wrapt
from potok.core.Data import DataDict


class ApplyToDataDict:
    def __init__(self, mode='all'):
        self.mode = mode
      
    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        return self.apply(wrapped, instance, *args, **kwargs)
      

    def apply(self, wrapped, instance, *args, **kwargs):
        units_list = [arg.keys() for arg in args]
        units = sorted(set.intersection(*map(set, units_list)), key=units_list[0].index)

        if ('train' in units) and (self.mode != 'all'):
            units.remove('train')

        args2 = [[arg[unit] for arg in args] for unit in units]
        kwargs2 = {unit: {k: v[unit] for k, v in kwargs.items()} for unit in units}
        res = self.apply_with_map(wrapped, instance, *args2, **kwargs2)
        result = DataDict(**dict(zip(units, res)))
        return result

    @staticmethod
    def apply_with_map(wrapped, instance, *args, **kwargs):
        return [wrapped(*arg, **kwarg) for arg, kwarg in zip(args, kwargs.values())]
