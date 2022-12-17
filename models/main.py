import pandas as pd
import numpy as np


from potok.core import DataDict, Pipeline
from potok.tabular import LightGBM,  StratifiedFolder
from potok.methods import Validation
from potok.tabular.TabularData import TabularData
from potok.tabular import EncodeX


def preprocess_df(df):
    columns = set(df.columns)
    required_columns = set([
        'eventTime', 'email', 'ip', 'cardToken', 'paymentSystem',
         'providerId', 'bankCountry', 'partyId', 'shopId', 'amount', 'currency',
         'result', 'bin_hash', 'ms_pan_hash'
    ])
    assert required_columns.issubset(columns), f'Dataframe must contain all required columns. {required_columns}'

    df = df.drop(columns=['fingerprint', 'cnt', 'sum'])
    df['bankCountry'] = df['bankCountry'].fillna('Unknown')
    df['email'] = df['email'].fillna('Unknown')
    df['ip'] = df['ip'].fillna('Unknown')

    df = df.drop_duplicates(subset=['eventTime', 'cardToken'])
    df = df.rename(columns={'result': 'target'})
    df = df.sort_values(by=['eventTime', 'cardToken']).reset_index(drop=True)
    df['dt_diff'] = (df['eventTime'] - df.groupby('cardToken')['eventTime'].shift()).dt.seconds
    df = df.set_index(['eventTime', 'cardToken'])
    return df


def create_features(df):
    # переводим валюты в рубли
    cross_rate = {
        'RUB': 1,
        'USD': 64,
        'EUR': 69,
    }
    df['amount_rub'] = df.apply(lambda x: x['amount'] * cross_rate[x['currency']], axis=1)

    # фичи по времени
    df['dt_diff_sum'] = df.groupby('cardToken').expanding()[['dt_diff']].sum().droplevel(0)['dt_diff'].fillna(-1)
    df['dt_diff_mean'] = df.groupby('cardToken').expanding()[['dt_diff']].mean().droplevel(0)['dt_diff'].fillna(-1)
    df['dt_diff_std'] = df.groupby('cardToken').expanding()[['dt_diff']].std().droplevel(0)['dt_diff'].fillna(0)

    # фичи по количеству транзакций
    df_cnt = df.copy()
    df_cnt['counter'] = 1
    df['tx_cardToken_count'] = df_cnt.groupby('cardToken').expanding()[['counter']].count().droplevel(0)['counter']
    df['tx_shopId_count'] = df_cnt.groupby(['cardToken', 'shopId']).expanding()[['counter']].count().droplevel([0, 1])['counter']

    # фичи по сумме транзакций
    df['amount_sum'] = df.groupby('cardToken').expanding()[['amount_rub']].sum().droplevel(0)['amount_rub']
    df['amount_mean'] = df.groupby('cardToken').expanding()[['amount_rub']].mean().droplevel(0)['amount_rub']
    df['amount_std'] = df.groupby('cardToken').expanding(1)[['amount_rub']].std().droplevel(0)['amount_rub'].fillna(0)
    df['uniq_amount'] = df.groupby('cardToken').expanding()['amount'].apply(lambda x: len(np.unique(x))).droplevel(0)

    # фичи по количеству уникальных категориальных фичей
    df['uniq_paymentSystem'] = df.groupby('cardToken')['paymentSystem'].transform(
        lambda x: [len(set(x[:i + 1])) for i in range(len(x))])
    df['uniq_bankCountry'] = df.groupby('cardToken')['bankCountry'].transform(
        lambda x: [len(set(x[:i + 1])) for i in range(len(x))])
    df['uniq_partyId'] = df.groupby('cardToken')['partyId'].transform(
        lambda x: [len(set(x[:i + 1])) for i in range(len(x))])
    df['uniq_shopId'] = df.groupby('cardToken')['shopId'].transform(
        lambda x: [len(set(x[:i + 1])) for i in range(len(x))])
    df['uniq_currency'] = df.groupby('cardToken')['currency'].transform(
        lambda x: [len(set(x[:i + 1])) for i in range(len(x))])
    df['uniq_bin_hash'] = df.groupby('cardToken')['bin_hash'].transform(
        lambda x: [len(set(x[:i + 1])) for i in range(len(x))])
    df['uniq_ms_pan_hash'] = df.groupby('cardToken')['ms_pan_hash'].transform(
        lambda x: [len(set(x[:i + 1])) for i in range(len(x))])
    df['uniq_providerId'] = df.groupby('cardToken')['providerId'].transform(
        lambda x: [len(set(x[:i + 1])) for i in range(len(x))])
    df['uniq_email'] = df.groupby('cardToken')['email'].transform(
        lambda x: [len(set(x[:i + 1])) for i in range(len(x))])
    df['uniq_ip'] = df.groupby('cardToken')['ip'].transform(lambda x: [len(set(x[:i + 1])) for i in range(len(x))])

    return df



def prepare_train_test_df(train_df, test_df):
    train_df_idx = preprocess_df(train_df)
    train_df_index = train_df_idx.index

    test_df_idx = preprocess_df(test_df)
    test_df_index = test_df_idx.index

    df_all = pd.concat([train_df, test_df], axis=0)
    prep_df_all = preprocess_df(df_all)
    ft_df_all = create_features(prep_df_all)

    train_df_orig = ft_df_all.loc[train_df_index]
    test_df_orig = ft_df_all.loc[test_df_index]
    return train_df_orig, test_df_orig


def train_model(df_train, df_test):
    num_ft = [
        'dt_diff', 'amount_rub', 'dt_diff_sum', 'dt_diff_mean', 'dt_diff_std',
        'tx_cardToken_count', 'tx_shopId_count', 'amount_sum', 'amount_mean', 'amount_std',
        'uniq_paymentSystem', 'uniq_bankCountry', 'uniq_partyId', 'uniq_shopId',
        'uniq_currency', 'uniq_bin_hash', 'uniq_ms_pan_hash', 'uniq_providerId',
        'uniq_email', 'uniq_ip',  'uniq_amount',
    ]
    cat_ft = [
        'paymentSystem', 'bankCountry', 'currency',
    ]
    all_ft = num_ft + cat_ft

    train_data = TabularData(df_train, target=['target'])
    test_data = TabularData(df_test, target=['target'])

    data = DataDict(train=train_data, test=test_data)
    x = DataDict(data_1=data.X)
    y = DataDict(data_1=data.Y)

    model_params = dict(
        n_estimators=2000,
        objective='binary',
        is_unbalance=True,
        n_jobs=-1,
    )

    folder = StratifiedFolder(n_folds=5, seed=2424)
    validation = Validation(folder)
    encoder = EncodeX(cat_ft, categorizer_name='ctbst')
    algo = LightGBM(target=['target'], mode='Classifier', model_params=model_params, features=all_ft)
    model = Pipeline([validation, encoder, algo], shapes=[1, 5, 5])
    prediction = model.fit_predict(x, y)

    return prediction


def post_process(prediction):
    df_pred = prediction.data_1.test.data
    df_labels = (df_pred[1] > 0.9)
    return df_labels


def main():
    df_train = pd.read_csv('models/X_train.csv', parse_dates=['eventTime'])
    df_test = pd.read_csv('models/X_test.csv', parse_dates=['eventTime'])
    df_train_ft, df_test_ft = prepare_train_test_df(df_train, df_test)
    print(f"df train shape {df_train_ft.shape}, df test shape {df_test_ft.shape}")
    prediction = train_model(df_train_ft, df_test_ft)
    df_labels = post_process(prediction)
    df_labels.to_csv('models/submission.csv')


if __name__ == '__main__':
    main()