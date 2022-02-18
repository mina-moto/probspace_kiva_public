"""
cvの各foldのモデルで各foldのvalidation_dataに対する予測を行い評価する。
testに対する推論も行う。
"""
import argparse

import yaml
from sklearn.metrics import mean_absolute_error

from dotenv import load_dotenv
import sys
import glob
import os
import mlflow
import numpy as np
import pandas as pd

load_dotenv()  # noqa
sys.path.append(f"{os.getenv('PROJECT_ROOT')}src/")  # noqa

from image_predict.module import mlflow_module
from image_predict.module.cv_model_manager import CvModelManager


class Evaluator:
    def __init__(
        self,
        validation_dataset_dir_path: str,
        test_path: str,
        model_dir_path: str,
        model_class_name: str,
        valid_prediction_save_path: str,
        test_prediction_save_path: str,
        output_transform: dict = None,
        *args,
        **kwargs,
    ):
        self.test_path = test_path
        self.validation_dataset_dir_path = validation_dataset_dir_path
        self.model_dir_path = model_dir_path
        self.model_class_name = model_class_name
        self.valid_prediction_save_path = valid_prediction_save_path
        self.test_prediction_save_path = test_prediction_save_path
        self.output_transform = output_transform

    def run(self):
        model_paths = glob.glob(f"{self.model_dir_path}*")
        cv_model_manager = CvModelManager(self.model_class_name, self.model_dir_path)
        valid_prediction_df = pd.DataFrame()

        def transform_output(output, is_exp=False):
            output = np.expm1(output) if is_exp else output
            output = output * 25
            return output

        for fold in range(len(model_paths)):
            valid_fold_df = pd.read_csv(f"{self.validation_dataset_dir_path}valid_fold_{fold}.csv")
            output = cv_model_manager.predict_model(valid_fold_df["IMAGE_PATH"], model_paths[0], load_batch=32)

            # モデルの出力値を変換
            output = transform_output(output, **self.output_transform)

            valid_fold_prediction_df = valid_fold_df.copy()
            valid_fold_prediction_df["PREDICTION_FROM_IMAGE"] = output
            mae = mean_absolute_error(
                valid_fold_prediction_df["LOAN_AMOUNT"],
                valid_fold_prediction_df["PREDICTION_FROM_IMAGE"])
            mlflow.log_metric(f"fold_{fold}_mae", mae)

            valid_prediction_df = pd.concat([valid_prediction_df, valid_fold_prediction_df], axis=0)
        mae = mean_absolute_error(valid_prediction_df["LOAN_AMOUNT"], valid_prediction_df["PREDICTION_FROM_IMAGE"])
        mlflow.log_metric("mae", mae)
        os.makedirs(os.path.dirname(self.valid_prediction_save_path), exist_ok=True)
        valid_prediction_df.to_csv(self.valid_prediction_save_path, index=False)

        # test predict
        test_df = pd.read_csv(self.test_path)
        test_prediction_df = test_df.copy()
        output = np.median(cv_model_manager.predict_models(test_df["IMAGE_PATH"], load_batch=32), axis=0)
        output = transform_output(output, **self.output_transform)
        test_prediction_df["PREDICTION_FROM_IMAGE"] = output
        test_prediction_df.to_csv(self.test_prediction_save_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='config/image_predict/evaluator001.yaml',
        help='config path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    mlflow_module.start_experiment(tracking_uri=os.getenv("TRACKING_URI"), **config["experiment_setting"])
    mlflow.log_artifact(args.config)
    evaluator = Evaluator(**config)
    evaluator.run()
    mlflow.end_run()


if __name__ == '__main__':
    main()
