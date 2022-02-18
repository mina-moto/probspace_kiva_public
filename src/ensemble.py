"""
複数のモデルの予測のensemble。
ensemble後、25-10000で離散化する。
線形回帰はrmseで計算されている。

Todo:
    線形回帰をmaeで計算

Examples
    python src/ensemble.py

Output:
    output/ensemble/{now datetime}/
"""
import datetime
import logging
import os
import shutil
import sys
from typing import List, Tuple

from sklearn.metrics import mean_absolute_error as mae
import scipy.optimize as opt
import numpy as np
from pathlib import Path
import traceback
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.metrics import mean_absolute_error

load_dotenv()  # noqa
sys.path.append(f"{os.getenv('PROJECT_ROOT')}src/")  # noqa

config = {
    "train_path": "input/train.csv",
    "test_path": "input/test.csv",
    # アンサンブル方法、出力されるファイルに含まれる、mean、median、ridge、lasso、elasticnet、linear、nelder_mead
    "ensemble_type": "mean",

    # Ensembleするモデル
    # train_path、test_pathに予測結果のファイルを指定。
    # train_pred_name、test_pred_nameに予測値が格納されているカラムを指定。
    "prediction_dict": {
        # lgb fold 10、特徴量009
        # "lgb_011_seed_777": {
        #     "train_path": "output/table/seed_change_exec_lgbm_train/011/seed_777/oof_192.832124_lgb_gbdt_cv10_2022-02-13-15-28.csv",
        #     "train_pred_name": "oof_pred",
        #     "test_path": "output/table/seed_change_exec_lgbm_train/011/seed_777/subm_192.832124_lgb_gbdt_cv10_2022-02-13-15-28.csv",
        #     "test_pred_name": "LOAN_AMOUNT",
        # },
        "lgb_011_seed_3407": {
            "train_path": "output/table/seed_change_exec_lgbm_train/011/seed_3407/oof_192.628555_lgb_gbdt_cv10_2022-02-13-14-33.csv",
            "train_pred_name": "oof_pred",
            "test_path": "output/table/seed_change_exec_lgbm_train/011/seed_3407/subm_192.628555_lgb_gbdt_cv10_2022-02-13-14-33.csv",
            "test_pred_name": "LOAN_AMOUNT",
        },
        # cat 10fold
        "cat_110": {
            "train_path": "output/table/catboost/110/oof_191.585607_catboost_cv10_2022-02-13-09-15.csv",
            "train_pred_name": "oof_pred",
            "test_path": "output/table/catboost/110/not_postprocess_subm_191.585607_catboost_cv10_2022-02-13-09-15.csv",
            "test_pred_name": "LOAN_AMOUNT",
        },
        # catパラメータ変更、特徴量009
        "cat_325": {
            "train_path": "output/table/catboost/325/oof_190.238310_catboost_cv10_2022-02-13-17-59.csv",
            "train_pred_name": "oof_pred",
            "test_path": "output/table/catboost/325/subm_190.238310_catboost_cv10_2022-02-13-17-59.csv",
            "test_pred_name": "LOAN_AMOUNT",
        },
    }
}


def discretize_predictions(predictions: np.ndarray) -> List[int]:
    """
    25から10000で離散化
    """
    discrete_amounts = [25 * i for i in range(1, 400, 1)]
    discreteized = list(map(lambda x: discrete_amounts[np.argmin(np.abs(discrete_amounts - x))], predictions))
    return discreteized


def merge_pred(df, pred_df, pred_name, new_pred_name):
    """
    dfにpred_dfのpred_nameを'LOAN_ID'軸で、new_pred_nameとしてmerge
    """
    return pd.merge(df, pred_df[["LOAN_ID", pred_name]], how='left', on='LOAN_ID').rename(columns={pred_name: new_pred_name})


def nelder_mead(x_train_, y_train_, x_test_):
    n_dim = x_train_.shape[1]

    def cost_func(x):
        weighted_pred = np.dot(x_train_, x)
        ans = mae(y_train_, weighted_pred)
        return ans

    # 制約条件の設定
    def cons_eq(x):
        return sum(x) - 1.0

    cons = (
        {'type': 'eq', 'fun': cons_eq}
    )

    ini_weights = [1 / n_dim] * n_dim
    # 変数のとりうる範囲の設定
    bounds = opt.Bounds([-0.5] * n_dim, [0.5] * n_dim)
    # Solution
    opt_result = opt.minimize(cost_func, ini_weights, method='Nelder-Mead', constraints=cons, bounds=bounds)

    print('weights are: ', opt_result.x)
    y_train_opted = np.dot(x_train_, opt_result.x)
    y_test_ = np.dot(x_test_, opt_result.x)
    return y_train_opted, y_test_


def ensemble(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    各列に各モデルの予測が格納されているdfに対してアンサンブルした結果を返す。
    """
    if "mean" == config["ensemble_type"]:
        return np.mean(x_train, axis=1), np.mean(x_test, axis=1)
    elif "median" == config["ensemble_type"]:
        return np.median(x_train, axis=1), np.median(x_test, axis=1)
    elif "nelder_mead" == config["ensemble_type"]:
        return nelder_mead(x_train, y_train, x_test)
    elif "ridge" == config["ensemble_type"]:
        model = Ridge(alpha=1e-2, normalize=True, fit_intercept=False)
    elif "lasso" == config["ensemble_type"]:
        model = Lasso(alpha=1e-2, normalize=True, fit_intercept=False)
    elif "elasticnet" == config["ensemble_type"]:
        model = ElasticNet(alpha=1e-2, normalize=True, fit_intercept=False)
    elif "linear" == config["ensemble_type"]:
        model = LinearRegression(normalize=True, fit_intercept=False)
    else:
        return

    # 線形モデルによる重み付け
    model.fit(x_train, y_train)
    train_ensemble_prediction, test_ensemble_prediction = model.predict(x_train), model.predict(x_test)
    return train_ensemble_prediction, test_ensemble_prediction


def run(output_dir):
    train_df = pd.read_csv(config["train_path"])
    test_df = pd.read_csv(config["test_path"])
    train_y = train_df["LOAN_AMOUNT"]

    # 各列に各モデルの予測を格納
    all_train_predictions = train_df[["LOAN_ID"]]
    all_test_predictions = test_df[["LOAN_ID"]]
    for model_type, model_prediction in config["prediction_dict"].items():
        train_prediction = pd.read_csv(model_prediction["train_path"])
        test_prediction = pd.read_csv(model_prediction["test_path"])
        all_train_predictions = merge_pred(all_train_predictions, train_prediction, model_prediction["train_pred_name"], model_type)
        all_test_predictions = merge_pred(all_test_predictions, test_prediction, model_prediction["test_pred_name"], model_type)

    train_ensemble_prediction, test_ensemble_prediction = ensemble(
        all_train_predictions.drop(columns=["LOAN_ID"]).values,
        all_test_predictions.drop(columns=["LOAN_ID"]).values,
        train_y,
    )

    train_ensemble_prediction = discretize_predictions(train_ensemble_prediction)
    test_ensemble_prediction = discretize_predictions(test_ensemble_prediction)

    mae = mean_absolute_error(train_y, train_ensemble_prediction)
    print(f"Ensemble type:{config['ensemble_type']}")
    print(f"CV:{mae}")

    submit_df = pd.DataFrame()
    submit_df["LOAN_ID"] = test_df["LOAN_ID"]
    submit_df["LOAN_AMOUNT"] = test_ensemble_prediction

    file_template = '{cv:.6f}_{ensemble_type}_{timestamp}'
    file_stem = file_template.format(
        cv=mae,
        ensemble_type=config["ensemble_type"],
        timestamp=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    )
    sub_filename = 'sub_{}.csv'.format(file_stem)
    submit_df.to_csv(output_dir / sub_filename, index=False)


def main():
    output_root = Path("output/")
    script_name = Path(__file__).stem
    os.makedirs(output_root / script_name, exist_ok=True)
    dt_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir_path = output_root / script_name / dt_now
    os.makedirs(output_dir_path, exist_ok=True)

    try:
        run(output_dir_path)
    except Exception as e:
        logging.basicConfig(filename=f'{output_dir_path}/error.log', level=logging.DEBUG)
        logging.error(e)
        traceback.print_exc()
    shutil.copy(__file__, output_dir_path)
    print(f"Save {output_dir_path}")


if __name__ == "__main__":
    main()
