import argparse
import gc
from pdb import Pdb
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
        # if "standard_scaler" in self.transform_types:
        #     self.standard_scaler = StandardScaler()
        #     self.standard_scaler.fit(np.expand_dims(target, 1))
        # if "minmax_scaler" in self.transform_types:
        #     self.minmax_scaler = MinMaxScaler()
        #     self.minmax_scaler.fit(np.expand_dims(target, 1))

    def transform_target(self):
        """
        目的変数の変換。25で除算した後、transform_typesに設定した処理を行う。
        """
        target = self.target / 25
        if "log" in self.transform_types:
            target = np.log1p(target)
        # if "minmax_scaler" in self.transform_types:
        #     target = pd.Series(self.minmax_scaler.transform(np.expand_dims(target, 1))[:, 0])
        # if "standard_scaler" in self.transform_types:
        #     target = pd.Series(self.standard_scaler.transform(np.expand_dims(target, 1))[:, 0])
        return target

    def transform_output(self, output):
        """
        予測値の変換。transform_typesに設定した処理の反対の処理を行い、25倍する。

        Args:
            output:

        """
        # if "standard_scaler" in self.transform_types:
        #     output = pd.Series(self.standard_scaler.inverse_transform(np.expand_dims(output, 1))[:, 0])
        # if "minmax_scaler" in self.transform_types:
        #     output = pd.Series(self.minmax_scaler.inverse_transform(np.expand_dims(output, 1))[:, 0])
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


def StratifiedKFold_lightgbm(train_df, test_df, tf_idf_vec, config):
    # config parameters
    save_path = config["save_path"]
    fold_num = config['train_setting']['fold_num']
    # es_rounds = config['train_setting']['es_rounds']
    fold_key = config['train_setting']['key']
    remove_features = list(config['remove_features'])
    label_enc_features = list(config['label_enc_features'])
    # target_enc_features = list(config['target_enc_features'])

    # target
    real_target = train_df['LOAN_AMOUNT']
    fold_df = train_df[fold_key]
    target = real_target * 1  # もともと25で割ってた
    train_df = train_df.drop(['LOAN_AMOUNT', 'is_loan_amount_outlier'], axis=1)

    # path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_model_path = os.path.join(save_path, 'model')
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    save_enc_model_path = os.path.join(save_path, 'enc')
    if not os.path.exists(save_enc_model_path):
        os.mkdir(save_enc_model_path)

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

    joblib.dump(ce_label_enc, os.path.join(save_enc_model_path, 'ce_label_enc.joblib'))

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
        # trn_data = cb.Pool(
        #     trn_df[features], label=target.iloc[trn_idx], cat_features=label_enc_features)
        # val_data = cb.Pool(
        #     val_df[features], label=target.iloc[val_idx], cat_features=label_enc_features)

        # train model
        model = joblib.load(os.path.join(
            save_model_path, 'cb_'+str(fold_)+'.joblib'))
        # model = cb.CatBoostRegressor()

        # model.load_model(save_model_path, 'lgbm_'+str(fold_)+'.joblib')

        # model.fit(trn_data, verbose_eval=200, eval_set=val_data)

        # save model
        # joblib.dump(model, os.path.join(
        # save_model_path, 'lgbm_'+str(fold_)+'.joblib'))

        # predict
        oof[val_idx] = model.predict(val_df[features])
        fold_predictions[:, fold_] = model.predict(fold_test_df[features])

        # save feature importances
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = features
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0)

    predictions = np.median(fold_predictions, axis=1)

    # save importances
    save_importances(feature_importance_df=feature_importance_df,
                     output_path=os.path.join(save_path, 'importance'))

    # scaler = joblib.load(os.path.join(save_path, 'scaler.joblib'))
    # show local CV
    target_transformer = TargetTransformer(target=real_target, transform_types=[])
    oof = target_transformer.transform_output(oof)
    # scaler.inverse_transform(np.expand_dims(oof, 1))
    local_CV = mean_absolute_error(real_target, oof)
    print("CV score: {:<8.5f}".format(local_CV))

    file_template = '{score:.6f}_{model_key}_cv{fold}_{timestamp}'

    # save test submit
    file_stem = file_template.format(
        score=local_CV,
        model_key='catboost',
        fold=fold_num,
        timestamp=dt.now().strftime('%Y-%m-%d-%H-%M'))

    sub_filename = 'subm_{}.csv'.format(file_stem)
    sub_filename_not_postprocess = 'not_postprocess_subm_{}.csv'.format(file_stem)

    sub_df = pd.DataFrame({'LOAN_ID': test_df['LOAN_ID'].values})
    sub_df_not_postprocess = pd.DataFrame({'LOAN_ID': test_df['LOAN_ID'].values})

    sub_df['LOAN_AMOUNT'] = discretize_predictions(target_transformer.transform_output(predictions))
    sub_df_not_postprocess['LOAN_AMOUNT'] = target_transformer.transform_output(predictions)

    sub_df.to_csv(os.path.join(save_path, sub_filename), index=False)
    sub_df_not_postprocess.to_csv(os.path.join(save_path, sub_filename_not_postprocess), index=False)

    # save oof
    oof_filename = 'oof_{}.csv'.format(file_stem)
    oof_df = pd.DataFrame({"LOAN_ID": train_df["LOAN_ID"].values})
    oof_df['oof_pred'] = oof 
    # scaler.inverse_transform(
        # np.array([list(oof)]).reshape(-1, 1))
    oof_df.to_csv(os.path.join(save_path, oof_filename), index=None)


def main():
    # load setting file
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    with timer("Load Datasets"):
        # load Dataset
        train_df = pd.read_csv(config['data']['train'])
        test_df = pd.read_csv(config['data']['test'])
        tf_idf_vec = joblib.load(config['data']['tfidf'])

    with timer("Run CatBoost Prediction for validation and test"):
        StratifiedKFold_lightgbm(
            train_df=train_df,
            test_df=test_df,
            tf_idf_vec=tf_idf_vec,
            config=config
        )


if __name__ == "__main__":
    main()
