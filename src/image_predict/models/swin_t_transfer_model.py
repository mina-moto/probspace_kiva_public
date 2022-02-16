import pytorch_lightning as pl
import torch
from timm import create_model
from torch import nn


class SwinTTransferModel(pl.LightningModule):
    """
    Swin_Tの転移学習
    """

    def __init__(self, model_params: dict, fold_name: str = ""):
        super().__init__()
        self.fold_name = fold_name
        self.backbone = create_model(model_name="swin_tiny_patch4_window7_224",
                                     pretrained=True, num_classes=0, in_chans=3,)
        self.model = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.backbone.num_features, 1))

        self.optimizer_params = model_params["optimizer_params"]
        self.scheduler_params = model_params["scheduler_params"]
        self._criterion = nn.L1Loss()
        self.save_hyperparameters(model_params)

    def forward(self, x):
        self.backbone.eval()
        with torch.no_grad():
            f = self.backbone(x)
        out = self.model(f)
        return out

    def training_step(self, batch, batch_idx):
        loss, pred, target = self.__share_step(batch, 'train')
        return {'loss': loss, 'pred': pred, "target": target}

    def validation_step(self, batch, batch_idx):
        loss, pred, target = self.__share_step(batch, 'val')
        return {'loss': loss, 'pred': pred, "target": target, }

    def __share_step(self, batch, mode):
        images, targets = batch
        logits = self.forward(images).squeeze(1)
        loss = self._criterion(logits, targets)
        return loss, logits, targets

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'val')

    def __share_epoch_end(self, outputs, mode):
        preds = [out['pred'] for out in outputs]
        targets = [out['target'] for out in outputs]
        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()
        loss = self._criterion(preds, targets)
        self.log(f'{mode}_{self.fold_name}_loss', value=loss, on_step=False, on_epoch=True, )

    def configure_optimizers(self):

        optimizer = eval("torch.optim.AdamW")(params=self.parameters(), **self.optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]
