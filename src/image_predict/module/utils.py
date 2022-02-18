import os

import pytorch_lightning as pl
import torch


def set_seed(seed=0):
    pl.seed_everything(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def retrieve_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
