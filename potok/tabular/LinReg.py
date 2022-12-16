# import statsmodels.api as sm
import pandas as pd
# from typing import List, Iterator, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.mixture import GaussianMixture

from potok.core import Regressor, ApplyToDataDict, DataDict
from potok.tabular.TabularData import TabularData


class LinReg(Regressor):
    def __init__(self, 
                 target=None,
                 features=None,
                 weight=None,
                 **kwargs,):
 
        super().__init__(**kwargs)
        self.target = target
        self.features = features
        self.weight = weight

        self.model = None
        self.index = None
    
    def _fit_(self, x: DataDict, y: DataDict) -> None:
        if self.target is None:
            self.target = x['train'].target

        if self.features is None:
            self.features = x['train'].data.columns
    
        x_train = x['train'].data[self.features]
        y_train = y['train'].data.dropna()[self.target]
        x_train = x_train.reindex(y_train.index)

        if self.weight is not None:
            w_train = y['train'].data.dropna()[self.weight]
            # self.model = sm.WLS(y_train, X_train, weights=w_train).fit()
        else:
            w_train = None
            # self.model = sm.OLS(y_train, X_train).fit()

        print('Training Linear Model')
        print(f'X_train = {x_train.shape} y_train = {y_train.shape}')

        self.model = LinearRegression().fit(x_train, y_train,)
        return None

    @ApplyToDataDict()
    def _predict_(self, x: DataDict) -> DataDict:
        assert self.model is not None, 'Fit model before or load from file.'
        x_new = x.data[self.features]
        # prediction = self.model.predict(exog=X)
        prediction = self.model.predict(x_new)
        prediction = pd.DataFrame(prediction, index=x.index, columns=self.target)
        # TODO: сделать инвариантно к типу, например x.__class__.__init__(data=prediction, target=self.target)
        y = TabularData(data=prediction, target=self.target)
        return y

