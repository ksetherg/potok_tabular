import numpy as np
import pandas as pd
from typing import Tuple, Union, List
from category_encoders.target_encoder import TargetEncoder
from category_encoders.cat_boost import CatBoostEncoder

from potok.core import Operator, DataDict, ApplyToDataDict


class TransformY(Operator):
    def __init__(self, transform: Union[tuple, str], target: str, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform
        self.target = target

        std_funcs = {
            'exp': (np.exp, np.log),
            'log': (np.log, np.exp),
            'square': (np.square, np.sqrt),
            'sqrt': (np.sqrt, np.square),
        }

        if transform in std_funcs:
            forward, backward = std_funcs[transform]
        else:
            forward, backward = transform

        self.forward = forward
        self.backward = backward

    @ApplyToDataDict()
    def y_forward(self, y: DataDict, x: DataDict = None, x_frwd: DataDict = None) -> DataDict:
        df = y.data
        df[self.target] = self.forward(df[self.target])
        y_frwd = y.copy(data=df)
        return y_frwd

    @ApplyToDataDict()
    def y_backward(self, y_frwd: DataDict) -> DataDict:
        df = y_frwd.data
        df[self.target] = self.backward(df[self.target])
        y = y_frwd.copy(data=df)
        return y


class CreateFeatureSpace(Operator):
    def __init__(self,
                 dynamic_features,
                 lag_params=None,
                 window_params=None,
                 min_max_params=None,
                 std_params=None,
                 diffs_params=None,
                 rate_params=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.dynamic_features = dynamic_features
        self.lag_params = lag_params
        self.window_params = window_params
        self.min_max_params = min_max_params
        self.std_params = std_params
        self.diffs_params = diffs_params
        self.rate_params = rate_params

    @ApplyToDataDict()
    def x_forward(self, x: DataDict) -> DataDict:
        df = x.data
        features = self._create_feature_space_(df)
        df_frwd = pd.concat([df, features], axis=1)
        x_frwd = x.copy(data=df_frwd)
        return x_frwd

    @staticmethod
    def _create_lag_tao_i_(data, keys, i):
        df_lag_tao_i = data[keys].groupby(level=0).shift(i)
        df_lag_tao_i = df_lag_tao_i.rename(columns={key: key + '_Lag_Tao+{}'.format(i) for key in keys})
        return df_lag_tao_i

    @staticmethod
    def _create_rolling_mean_(data, keys, window):
        df_rolling_mean = data[keys].groupby(level=0).apply(lambda x: x.rolling(window=window).mean())
        df_rolling_mean = df_rolling_mean.rename(
            columns={key: key + '_Rolling_Mean_Window={}'.format(window) for key in keys})
        return df_rolling_mean

    @staticmethod
    def _create_rolling_std_(data, keys, window):
        df_rolling_std = data[keys].groupby(level=0).apply(lambda x: x.rolling(window=window).std())
        df_rolling_std = df_rolling_std.rename(
            columns={key: key + '_Rolling_STD_Window={}'.format(window) for key in keys})
        return df_rolling_std

    @staticmethod
    def _create_rolling_max_(data, keys, window):
        df_rolling_mean = data[keys].groupby(level=0).apply(lambda x: x.rolling(window=window).max())
        df_rolling_mean = df_rolling_mean.rename(
            columns={key: key + '_Rolling_Max_Window={}'.format(window) for key in keys})
        return df_rolling_mean

    @staticmethod
    def _create_rolling_min_(data, keys, window):
        df_rolling_mean = data[keys].groupby(level=0).apply(lambda x: x.rolling(window=window).min())
        df_rolling_mean = df_rolling_mean.rename(
            columns={key: key + '_Rolling_Min_Window={}'.format(window) for key in keys})
        return df_rolling_mean

    @staticmethod
    def _create_diff_(data, keys, periods):
        df_rolling_diff = data[keys].groupby(level=0).diff(periods=periods)
        df_rolling_diff = df_rolling_diff.rename(columns={key: key + '_Diff_Lag={}'.format(periods) for key in keys})
        return df_rolling_diff

    @staticmethod
    def _create_rate_(data, keys, lag):
        df_rolling_rate = data[keys] / data[keys].groupby(level=0).shift(lag)
        df_rolling_rate = df_rolling_rate.rename(columns={key: key + '_Rate_Lag={}'.format(lag) for key in keys})
        return df_rolling_rate

    def _create_feature_space_(self, data):
        dfs = []

        if self.lag_params is not None:
            print('Calculate lags')
            for i in self.lag_params:
                df_lag_tao_i = self._create_lag_tao_i_(data, self.dynamic_features, int(i))
                dfs.append(df_lag_tao_i)

        if self.window_params is not None:
            print('Calculate rolling mean')
            for i in self.window_params:
                df_shift_roll_mean_i = self._create_rolling_mean_(data, self.dynamic_features, int(i))
                dfs.append(df_shift_roll_mean_i)

        if self.min_max_params is not None:
            print('Calculate rolling max')
            for i in self.min_max_params:
                df_roll_max_i = self._create_rolling_max_(data, self.dynamic_features, int(i))
                dfs.append(df_roll_max_i)

            print('Calculate rolling min')
            for i in self.min_max_params:
                df_roll_min_i = self._create_rolling_min_(data, self.dynamic_features, int(i))
                dfs.append(df_roll_min_i)

        if self.std_params is not None:
            print('Calculate shifted rolling mean')
            for i in self.std_params:
                df_shift_roll_mean_i = self._create_rolling_std_(data, self.dynamic_features, int(i))
                dfs.append(df_shift_roll_mean_i)

        if self.diffs_params is not None:
            print('Calculate diffs')
            for i in self.diffs_params:
                df_diff_i = self._create_diff_(data, self.dynamic_features, int(i))
                dfs.append(df_diff_i)

        if self.rate_params is not None:
            print('Calculate rates')
            for i in self.rate_params:
                df_rate_i = self._create_rate_(data, self.dynamic_features, i)
                dfs.append(df_rate_i)

        print('Merging')
        df_merged = pd.concat(dfs, axis=1)
        return df_merged


class EncodeX(Operator):
    def __init__(
        self,
        features_to_encode: List[str],
        categorizer_name: str,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.features_to_encode = features_to_encode
        self.categorizer_name = categorizer_name
        self.categorizer = None

    @ApplyToDataDict()
    def x_forward(self, x: DataDict) -> DataDict:
        df = x.data
        df_frwd = self.categorizer.transform(df)
        x_frwd = x.copy(data=df_frwd)
        return x_frwd

    def _fit_(self, x: DataDict, y: DataDict) -> None:
        df_x = x['train'].data
        df_y = y['train'].data[y['train'].target]
        features = [f for f in self.features_to_encode if f in df_x.columns]

        if self.categorizer_name == 'target':
            self.categorizer = TargetEncoder(cols=features)
        elif self.categorizer_name == 'ctbst':
            self.categorizer = CatBoostEncoder(cols=features)
        else:
            raise Exception('Error: unknown categorizer name')

        self.categorizer.fit(df_x, df_y)
        return None

        # def db_state(self, state):
        #     state = super().db_state(state)
        #     state['categorizer'] = None
        #     return state
        #
        # def _db_save_data(self, path):
        #     if self.categorizer is not None:
        #         joblib.dump(self.categorizer, path + 'categorizer.trfrm')
        #
        # def _db_load_data(self, path):
        #     try:
        #         self.categorizer = joblib.load(path + 'categorizer.trfrm')
        #     except:
        #         pass