from typing import List, Iterator, Tuple

from potok.core import DataDict, Pipeline
from potok.tabular import Folder, LightGBM, TransformY, LinReg, SyntheticData
from potok.methods import Validation, Bagging


def generate_regression_data(problem) -> Tuple[DataDict, DataDict]:
    gene = SyntheticData(problem=problem)
    train = gene.create_train()
    test = gene.create_test()
    data = DataDict(train=train, test=test)
    return data.X, data.Y


def test_lightgbm_regression() -> DataDict:
    x, y = generate_regression_data(problem='regression')
    x = DataDict(data_1=x)
    y = DataDict(data_1=y)
    transform = TransformY(transform='square', target='Target')
    folder = Folder(n_folds=1, seed=2424)
    validation = Validation(folder)
    algo = LightGBM(target=['Target'], features=['X'])
    model = Pipeline([transform, validation, algo], shapes=[1, 1, 1])
    prediction = model.fit_predict(x, y)
    return prediction


def test_lightgbm_classification() -> DataDict:
    x, y = generate_regression_data(problem='classification')
    x = DataDict(data_1=x)
    y = DataDict(data_1=y)
    folder = Folder(n_folds=3, seed=2424)
    validation = Validation(folder)
    algo = LightGBM(target=['Target'], mode='Classifier', features=['X'])
    model = Pipeline([validation, algo], shapes=[1, 3])
    prediction = model.fit_predict(x, y)
    return prediction


def test_linear_regression() -> DataDict:
    x, y = generate_regression_data(problem='regression')
    x = DataDict(data_1=x)
    y = DataDict(data_1=y)
    algo = LinReg(target=['Target'], features=['X'])
    model = Pipeline(algo, shapes=[1, ])
    prediction = model.fit_predict(x, y)
    return prediction


def test_bagging() -> DataDict:
    x, y = generate_regression_data(problem='regression')
    x = DataDict(data_1=x,)
    y = DataDict(data_1=y,)
    folder = Folder(n_folds=3, seed=None)
    validation = Validation(folder)
    algo = LightGBM(target=['Target'], features=['X'])
    model = Pipeline(Bagging(5), validation, algo, shapes=[1, 5, 15])
    prediction = model.fit_predict(x, y)
    return prediction


if __name__ == "__main__":
    pred1 = test_lightgbm_regression()
    pred2 = test_lightgbm_classification()
    pred3 = test_linear_regression()
    pred4 = test_bagging()
