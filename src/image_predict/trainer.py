"""

"""
import argparse
import os
import sys

import mlflow
import pandas as pd
import pytorch_lightning as pl
import yaml
from dotenv import load_dotenv

load_dotenv()  # noqa
sys.path.append(f"{os.getenv('PROJECT_ROOT')}src/")  # noqa

from image_predict.data_module.kiva_data_module import KivaDataModule
from image_predict.module import mlflow_module
from module.utils import set_seed
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.model_selection import KFold

from image_predict.models.swin_t_transfer_model import SwinTTransferModel
from image_predict.models.swin_t_finetune_model import SwinTFinetuneModel


class Trainer:
    def __init__(
            self,
            train_path: str,
            validation_dataset_save_dir: str,
            model_dir_save_path: str,
            model_class_name: str = "SwinTTransferModel",
            seed: int = 0,
            validation_num: int = 4,
            model_params: dict = None,
            pl_trainer_params: dict = None,
            early_stopping_params: dict = None,
            train_loader_params: dict = None,
            val_loader_params: dict = None,
            *args,
            **kwargs,
    ):
        """

        Args:
            model_class_name:
            train_path:
            validation_dataset_save_dir:
            model_dir_save_path:
            seed:
            validation_num:
            pl_trainer_params:
            early_stopping_params:
            train_loader_params:
            val_loader_params:
        """
        self.model_class_name = model_class_name
        self.data_df = pd.read_csv(train_path)
        self.validation_dataset_save_dir = validation_dataset_save_dir
        self.model_dir_save_path = model_dir_save_path
        self.seed = seed
        self.validation_num = validation_num
        self.model_params = model_params
        self.pl_trainer_params = pl_trainer_params
        self.early_stopping_params = early_stopping_params
        self.train_loader_params = train_loader_params
        self.val_loader_params = val_loader_params

    def __train(self, train, valid, fold_name):
        model = eval(self.model_class_name)(model_params=self.model_params, fold_name=fold_name)
        datamodule = KivaDataModule(
            train,
            valid,
            train_loader_params=self.train_loader_params,
            val_loader_params=self.val_loader_params,
        )
        early_stopping = callbacks.EarlyStopping(
            monitor=f"val_{fold_name}_loss",
            **self.early_stopping_params
        )
        lr_monitor = callbacks.LearningRateMonitor()
        os.makedirs(self.model_dir_save_path, exist_ok=True)
        loss_checkpoint = callbacks.ModelCheckpoint(
            dirpath=self.model_dir_save_path,
            filename=fold_name,
            monitor=f"val_{fold_name}_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
        )
        mlf_logger = MLFlowLogger()
        mlf_logger._run_id = mlflow.active_run().info.run_id
        trainer = pl.Trainer(
            logger=mlf_logger,
            callbacks=[lr_monitor, loss_checkpoint, early_stopping],
            **self.pl_trainer_params,
        )
        trainer.fit(model, datamodule=datamodule)
        mlflow.log_metric(f"epoch_{fold_name}", trainer.current_epoch)

    def run(self):
        set_seed(self.seed)
        kf = KFold(n_splits=self.validation_num, shuffle=True, random_state=self.seed)
        for fold, (train_index, valid_index) in enumerate(kf.split(self.data_df["IMAGE_PATH"])):
            train = self.data_df.loc[train_index]
            valid = self.data_df.loc[valid_index]
            os.makedirs(self.validation_dataset_save_dir, exist_ok=True)
            train.to_csv(f"{self.validation_dataset_save_dir}train_fold_{fold}.csv", index=False)
            valid.to_csv(f"{self.validation_dataset_save_dir}valid_fold_{fold}.csv", index=False)
            self.__train(train=train, valid=valid, fold_name=f"fold_{fold}")
        params = {
            "validation_dataset_save_dir": self.validation_dataset_save_dir,
            "model_dir_save_path": self.model_dir_save_path,
            "seed": self.seed,
            "validation_num": self.validation_num,
            "model_params": self.model_params,
            "pl_trainer_params": self.pl_trainer_params,
            "early_stopping_params": self.early_stopping_params,
            "train_loader_params": self.train_loader_params,
            "val_loader_params": self.val_loader_params,
        }
        mlflow.log_params(params)
        mlflow.log_artifact(self.validation_dataset_save_dir)
        mlflow.log_artifact(self.model_dir_save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='config/image_predict/trainer/trainer001.yaml',
        help='config path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    mlflow_module.start_experiment(tracking_uri=os.getenv("TRACKING_URI"), **config["experiment_setting"])
    mlflow.log_artifact(args.config)
    trainer = Trainer(**config)
    trainer.run()
    mlflow.end_run()


if __name__ == '__main__':
    main()
