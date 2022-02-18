import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from image_predict.data_module.kiva_dataset import KivaDataset
from image_predict.models.swin_t_finetune_model import SwinTFinetuneModel
from image_predict.models.swin_t_transfer_model import SwinTTransferModel
from image_predict.module.utils import retrieve_device


class CvModelManager:
    def __init__(self, model_class_name, model_dir_path):
        """

        Args:
            model_class_name:
            model_dir_path:モデルが入ったディレクトリのパス
        """
        self.model_class_name = model_class_name
        self.model_dir_path = model_dir_path
        self.model_paths = sorted(glob.glob(f"{self.model_dir_path}/*"))

        # model_pathごとのモデル
        self.models = {}
        device = retrieve_device()
        for model_path in self.model_paths:
            model = eval(model_class_name).load_from_checkpoint(
                model_path,
                # required positional argument: 'fold_name'に対処
                # 恐らくモデルでsave_hyperparametersするべき
                fold_name=""
            )
            model.to(device)
            model.eval()
            self.models[model_path] = model

    def predict(self, image_path_list: list, load_batch: int = 1):
        """
        image_path_listをmodelsのモデルで予測してsoftmaxした結果を返す。

        Args:
            image_path_list:予測する画像のリスト
            load_batch:data_dfを何件ずつ読み込むか

        """
        return np.mean(self.predict_models(image_path_list, load_batch), axis=0)

    def predict_models(self, image_path_list: list, load_batch=1):
        """
        image_path_listをmodelsのモデルごとに予測してsoftmaxした結果を返す。

        Args:
            image_path_list:予測する画像のリスト
            load_batch:data_dfを何件ずつ読み込むか

        """
        return np.array([self.predict_model(image_path_list, model, load_batch) for model in self.models])

    def predict_model(self, image_path_list: list, model_path: str, load_batch: int = 1):
        """
        image_path_listをmodel_pathのモデルで予測してsoftmaxした結果を返す。

        Args:
            image_path_list:予測する画像のリスト
            model_path:予測を行うモデルのパス
            load_batch:data_dfを何件ずつ読み込むか
        """
        batch_preds = []
        model = self.models[model_path]

        generator = DataLoader(
            dataset=KivaDataset(image_path_list=image_path_list, is_train=False),
            batch_size=load_batch
        )
        device = retrieve_device()
        with torch.no_grad():
            for batch in iter(generator):
                batch_preds.append(model(batch.to(device)))
            preds = np.array(torch.cat(batch_preds).cpu())
        return preds
