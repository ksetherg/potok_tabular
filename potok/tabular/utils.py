import scipy.stats as sst
import numpy as np
import pandas as pd

from potok.tabular import TabularData


class SyntheticData:
    def __init__(self,
                 problem='regression',
                 pdf_train=sst.norm(loc=-1, scale=2),
                 pdf_test=sst.norm(loc=1, scale=3),
                 seed=None):

        assert problem in ['regression', 'classification']
        self.problem = problem

        self.pdf_train = pdf_train
        self.pdf_test = pdf_test
        self.seed = seed

    def _create_sample_(self, pdf, size):
        x = pdf.rvs(size=size, random_state=self.seed)
        if self.problem == 'regression':
            y = 2*x + 1
        if self.problem == 'classification':
            y = (x > np.mean(x)).astype(int)

        df = pd.DataFrame({'X': x, 'Target': y}, index=list(range(size)))
        return df

    def create_train(self, size=10):
        df_train = self._create_sample_(self.pdf_train, size)
        return TabularData(df_train, target=['Target'])

    def create_test(self, size=10):
        data_test = self._create_sample_(self.pdf_test, size)
        return TabularData(data_test, target=['Target'])
