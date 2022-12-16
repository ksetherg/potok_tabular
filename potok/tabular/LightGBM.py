import lightgbm as lgb
import pandas as pd
from pathlib import Path
import joblib
# from typing import List, Iterator, Tuple

from potok.core import Regressor, ApplyToDataDict, DataDict
from potok.tabular.TabularData import TabularData
# from potok.tabular.HyperOptimization import HrPrmOptRange, HrPrmOptChoise


class LightGBM(Regressor):
    def __init__(self,
                 target=None,
                 features=None,
                 cat_features=None,
                 mode='Regressor',
                 objective='mse',
                 eval_metric='mse',
                 num_class=None,
                 weight=None,
                 **kwargs,
                 ):

        super().__init__(**kwargs)
        self.target = target
        self.features = features
        self.cat_features = cat_features
        self.mode = mode
        self.weight = weight

        self.model_params = dict(
            n_estimators=2000,
            learning_rate=0.03,
            num_class=num_class,
            objective=objective,
            # max_depth=HrPrmOptChoise(8, list(range(2, 12))),
            # num_leaves=HrPrmOptChoise(31, list(range(8, 56))),
            # log10_min_child_weight=HrPrmOptRange(0, -3.0, 3.0),
            # min_split_gain=HrPrmOptRange(0.5, 0.0, 1.0),
            # subsample=HrPrmOptRange(0.8, 0.5, 1.0),
            # colsample_bytree=HrPrmOptRange(0.8, 0.5, 1.0),
            # reg_alpha=HrPrmOptRange(1.0, 0.0, 3.0),
            # reg_lambda=HrPrmOptRange(0.0, 0.0, 3.0),
            # class_weight='balanced',
            importance_type='gain',
            n_jobs=-1,
        )

        self.training_params = dict(
            eval_metric=eval_metric,
            early_stopping_rounds=50,
            verbose=100,
        )

        self.model = None
        self.cat_features_idx = None
        self.feature_importance_df = None

    def _restate_(self):
        self.__dict__['model'] = None
        self.__dict__['feature_importance_df'] = None
        return None

    def _save_(self, prefix: Path = None) -> None:
        path = prefix / 'lightgbm.pkl'
        if self.model is not None:
            joblib.dump(self.model, path)

    def _load_(self, prefix: Path = None) -> None:
        path = prefix / 'lightgbm.pkl'
        try:
            self.model = joblib.load(path)
        except OSError:
            raise Exception('Cant find Model weights.')

    def _set_model_(self):
        if self.mode == 'Regressor':
            self.model = lgb.LGBMRegressor()
        elif self.mode == 'Classifier':
            self.model = lgb.LGBMClassifier()
        else:
            raise Exception('Unknown mode %s' % self.mode)
        # params = {k: (x.value if isinstance(x, (HrPrmOptRange, HrPrmOptChoise)) else x) for k, x in self.model_params.items()}
        params = self.model_params
        self.model.set_params(**params)

    def _fit_(self, x: DataDict, y: DataDict) -> None:
        self._set_model_()

        if self.target is None:
            self.target = x['train'].target

        if self.features is None:
            self.features = x['train'].data.columns

        x_train, x_valid = x['train'].data[self.features], x['valid'].data[self.features]
        y_train, y_valid = y['train'].data.dropna()[self.target], y['valid'].data[self.target]
        x_train = x_train.reindex(y_train.index)

        if self.weight is not None:
            w_train, w_valid = y['train'].data.dropna()[self.weight], y['valid'].data[self.weight]
        else:
            w_train, w_valid = None, None

        if self.cat_features is not None:
            self._set_cat_features_(list(self.features))
        else:
            self.cat_features_idx = 'auto'

        # if len(self.target) > 1 and self.mode == "Classifier":
        #     y_train = self._ohe_decode_(y_train)
        #     y_valid = self._ohe_decode_(y_valid)

        print('Training LightGBM')
        print(f'X_train = {x_train.shape} y_train = {y_train.shape}')
        print(f'X_valid = {x_valid.shape} y_valid = {y_valid.shape}')

        self.model = self.model.fit(X=x_train, y=y_train, sample_weight=w_train,
                                    eval_set=[(x_valid, y_valid)], eval_sample_weight=[w_valid],
                                    categorical_feature=self.cat_features_idx,
                                    **self.training_params)

        self._make_feature_importance_df_()
        return None

    @ApplyToDataDict(mode='efficient')
    def _predict_(self, x: DataDict) -> DataDict:
        assert self.model is not None, 'Fit model before or load from file.'
        x_new = x.data[self.features]
        if self.mode == 'Classifier':
            prediction = self.model.predict_proba(x_new)
            prediction = pd.DataFrame(prediction, index=x.index)
        elif self.mode == 'Regressor':
            prediction = self.model.predict(x_new)
            prediction = pd.DataFrame(prediction, index=x.index, columns=[self.target])
        else:
            raise Exception('Unknown mode.')
        # TODO: сделать инвариантно к типу, например x.__class__.__init__(data=prediction, target=self.target)
        y = TabularData(data=prediction, target=[self.target])
        return y

    def _set_cat_features_(self, features):
        cat_features_idx = []
        for cat_feature in self.cat_features:
            idx = features.index(cat_feature)
            cat_features_idx.append(idx)
        self.cat_features_idx = cat_features_idx

    def _ohe_decode_(self, data):
        df = data[self.target]
        df = df.idxmax(axis=1).to_frame('Target')
        df['Target'] = df['Target'].astype(int)
        return df

    def _make_feature_importance_df_(self):
        feature_importance = self.model.feature_importances_
        feature_names = self.model.feature_name_

        importance = {}
        for pair in sorted(zip(feature_importance, feature_names)):
            importance[pair[1]] = pair[0]

        self.feature_importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['weight'])
        self.feature_importance_df.index.name = 'features'

