import argparse
import gc
import yaml
import joblib
from typing import List
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn import decomposition
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from contextlib import contextmanager
import warnings
from datetime import datetime as dt
from pandas.core.common import SettingWithCopyWarning
import json
import collections as cl

import category_encoders as ce

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Argment setting
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str)
args = parser.parse_args()

class TargetTransformer:
    def __init__(self, target, transform_types: List[str] = None):
        """
        Args:
            target:目的変数
            transform_types:目的変数の変換方法のリスト
                'log':log1p変換
                'standard_scaler':標準正規化
        """
        self.transform_types = transform_types
        self.target = target


    def transform_target(self):
        """
        目的変数の変換。25で除算した後、transform_typesに設定した処理を行う。
        """
        target = self.target / 25
        if "log" in self.transform_types:
            target = np.log1p(target)

        return target

    def transform_output(self, output):
        """
        予測値の変換。transform_typesに設定した処理の反対の処理を行い、25倍する。

        Args:
            output:

        """
        if "log" in self.transform_types:
            output = np.expm1(output)
        output = output * 25
        return output


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def save_importances(feature_importance_df, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False).index)

    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(
        cols)]

    # for checking all importance
    _feature_importance_df = feature_importance_df.groupby('Feature').sum()
    _feature_importance_df.to_csv(os.path.join(
        output_path, 'feature_importance_lgbm.csv'))

    plt.figure(figsize=(28, 56))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(
        by="importance", ascending=False))
    plt.title('Features importance (averaged/folds)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'FI.png'))


def param_csv(params, path):
    fw = open(os.path.join(path, 'catboost_parameters.json'), 'w')
    json.dump(params, fw, indent=4)


def svd_tiidf(tf_idf_vec, trn_idx, val_idx):
    trn_tfidf_vec = tf_idf_vec[trn_idx]
    val_tfidf_vec = tf_idf_vec[val_idx]
    test_tfidf_vec = tf_idf_vec[91333:]

    svd = decomposition.TruncatedSVD(n_components=2, random_state=3655)

    trn_tfidf_svd = pd.DataFrame(svd.fit_transform(
        trn_tfidf_vec), columns=['tfidf_svd1', 'tfidf_svd2'])
    val_tfidf_svd = pd.DataFrame(svd.transform(val_tfidf_vec), columns=[
                                 'tfidf_svd1', 'tfidf_svd2'])
    test_tfidf_svd = pd.DataFrame(svd.transform(test_tfidf_vec), columns=[
                                  'tfidf_svd1', 'tfidf_svd2'])

    return trn_tfidf_svd, val_tfidf_svd, test_tfidf_svd


def discretize_predictions(predictions: np.ndarray) -> np.ndarray:
    discrete_amounts = [25 * i for i in range(1, 800, 1)]
    discreteized = list(map(
        lambda x: discrete_amounts[np.argmin(np.abs(discrete_amounts - x))], predictions))
    return discreteized
# def stratified_k_fold_train():


def scaling_target(loan_amount_ser: pd.Series, save_path):
    loan_amount_2d = np.expand_dims(loan_amount_ser, 1)#np.array([list(loan_amount_ser.values)]).reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(loan_amount_2d)
    joblib.dump(scaler, os.path.join(save_path, 'scaler.joblib'))

    loan_amount_2d_scaled = scaler.transform(loan_amount_2d)
    return pd.Series(loan_amount_2d_scaled[:, 0], name='LOAN_AMOUNT')


def StratifiedKFold_catboost_train(train_df, test_df, tf_idf_vec, config):
    # config parameters
    save_path = config["save_path"]
    fold_num = config['train_setting']['fold_num']
    fold_key = config['train_setting']['key']
    remove_features = list(config['remove_features'])
    label_enc_features = list(config['label_enc_features'])

    # path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_model_path = os.path.join(save_path, 'model')
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    save_enc_model_path = os.path.join(save_path, 'enc')
    if not os.path.exists(save_enc_model_path):
        os.mkdir(save_enc_model_path)

    # target
    real_target = train_df['LOAN_AMOUNT']
    fold_df = train_df[fold_key]
    target_transformer = TargetTransformer(target=real_target, transform_types=[])
    target = target_transformer.transform_target()
    train_df = train_df.drop(['LOAN_AMOUNT', 'is_loan_amount_outlier'], axis=1)

    # define catboost parameter
    cat_param = config["cat_params"]

    print("Starting CatBoost. Train shape: {}, test shape: {}".format(
        train_df.shape, test_df.shape))

    # define folds
    folds = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=cat_param['random_seed'])

    # Create arrays and dataframes to store results
    oof = np.zeros(len(train_df))
    fold_predictions: np.ndarray = np.zeros((len(test_df), fold_num))
    feature_importance_df = pd.DataFrame()

    # label encording
    ce_label_enc = ce.OrdinalEncoder(
        cols=label_enc_features, handle_unknown='impute')

    train_df = ce_label_enc.fit_transform(train_df)
    test_df = ce_label_enc.transform(test_df)
    test_df[label_enc_features] = test_df[label_enc_features].fillna(
        -1).astype('int')

    joblib.dump(ce_label_enc, os.path.join(
        save_enc_model_path, 'ce_label_enc.joblib'))

    # n-folds
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, fold_df)):
        print("fold n: {}".format(fold_))

        # split-trn-val
        trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        fold_test_df = test_df.copy()

        # tfidf-SVD
        trn_tfidf_svd, val_tfidf_svd, test_tfidf_svd = svd_tiidf(
            tf_idf_vec, trn_idx, val_idx)
        trn_df = pd.concat([trn_df, trn_tfidf_svd], axis=1)
        val_df = pd.concat([val_df, val_tfidf_svd], axis=1)
        fold_test_df = pd.concat([fold_test_df, test_tfidf_svd], axis=1)

        features = [c for c in trn_df.columns if c not in remove_features]
        trn_data = cb.Pool(
            trn_df[features], label=target.iloc[trn_idx], cat_features=label_enc_features)
        val_data = cb.Pool(
            val_df[features], label=target.iloc[val_idx], cat_features=label_enc_features)

        # train model
        model = cb.CatBoostRegressor(
            **cat_param,
            cat_features=label_enc_features,
            num_boost_round=100000000,
        train_dir=os.path.join(save_path, 'catboost_info'),
        )

        model.fit(trn_data, verbose_eval=200,
                  eval_set=val_data)

        # save model
        joblib.dump(model, os.path.join(
            save_model_path, 'cb_'+str(fold_)+'.joblib'))

    # save param
    param_csv(cat_param, path=save_path)


def main():
    # load setting file
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    with timer("Load Datasets"):
        # load Dataset
        train_df = pd.read_csv(config['data']['train'])
        test_df = pd.read_csv(config['data']['test'])
        tf_idf_vec = joblib.load(config['data']['tfidf'])

    with timer("Run LightGBM with StratifiedKFold"):
        StratifiedKFold_catboost_train(
            train_df=train_df,
            test_df=test_df,
            tf_idf_vec=tf_idf_vec,
            config=config
        )


if __name__ == "__main__":
    main()
