import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from image_predict.data_module.kiva_dataset import KivaDataset


class KivaDataModule(LightningDataModule):
    def __init__(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            train_loader_params: dict,
            val_loader_params: dict,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._train_loader_params = train_loader_params
        self._val_loader_params = val_loader_params

    def train_dataloader(self):
        dataset = KivaDataset(self._train_df["IMAGE_PATH"].values, self._train_df["LOAN_AMOUNT"].values, is_train=True)
        return DataLoader(dataset, **self._train_loader_params)

    def val_dataloader(self):
        dataset = KivaDataset(self._val_df["IMAGE_PATH"].values, self._val_df["LOAN_AMOUNT"].values, is_train=False)
        return DataLoader(dataset, **self._val_loader_params)
