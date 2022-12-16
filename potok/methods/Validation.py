from pathlib import Path
from potok.core import Operator, DataDict


class Validation(Operator):
    def __init__(self, folder, **kwargs):
        super().__init__(**kwargs)
        self.folder = folder
        self.index = None

    # def _restate_(self):
    #     self.__dict__['folder'] = None
    #     return None
    #
    # def _save_(self, prefix: Path = None) -> None:
    #     if self.folder is not None:
    #         self.folder.save(prefix)
    #
    # def _load_(self, prefix: Path = None) -> None:
    #     try:
    #         print('CHE', self.folder, prefix)
            # Проблема, что когда я сохраняю валидацию, оно сохраняется с нановым фолдером, далее применяю некорректно
    #       # Возможное решение, возврашать инстанс загрузки просто, загружать классом, как класс метод
    #         self.folder.load(prefix)
    #     except:
    #         raise Exception('Folder does not exsist.')

    def x_forward(self, x: DataDict) -> DataDict:
        self.index = x.index
        x2 = self.folder.x_forward(x)
        return x2

    def y_forward(self, y: DataDict, x: DataDict = None, x_frwd: DataDict = None) -> DataDict:
        y2 = self.folder.y_forward(y)
        return y2

    def y_backward(self, y_frwd: DataDict) -> DataDict:
        y_bck = self.folder.y_backward(y_frwd)
        y = DataDict(train=y_bck['valid'])
        units = y_bck.keys()
        units.remove('valid')
        y.__setstate__({unit: y_bck[unit] for unit in units})
        y = y.reindex(self.index)
        return y
    
    def _fit_(self, x: DataDict, y: DataDict) -> None:
        self.folder._fit_(x, y)
        return None
