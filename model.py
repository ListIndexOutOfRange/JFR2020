""" Base Model Class: A Lighning Module
    This class implements all the logic code.
    This model class will be the one to be fit by a Trainer
 """

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models.init import init_optimizer, init_scheduler
from models.net import DeepLabV3_3D
from models.losses import DC_and_topk_loss,SoftDiceLoss,IoULoss,FocalTversky_loss

from pytorch_lightning.metrics.functional import iou
from pytorch_lightning.metrics.classification import accuracy


class LightningModel(pl.LightningModule):
    """
        LightningModule handling everything training related.
        Pytorch Lightning will be referred as Lightning in all the following.
        Many attributes and method used aren't explicitely defined here
        but comes from the LightningModule class.
        This behavior should be specify in a docstring.
        Please refer to the Lightning documentation for further details.

        Note that Lighning handles tensorboard logging, early stopping, and auto checkpoints
        for this class.
    """

    def __init__(self, config):
        """  All the params are saved alltogether with the weights.
             Hence one can load an already trained model and acces all its hyperparameters.
             This call to save_hyperparameters() is handled by Lightning.
             It makes something like self.hparams.dataloader.train_batch_size callable.

        Args:
            config (dataclass): config is a dataclass containing 4 dataclasses:
                                    1. network
                                    2. optimizer
                                    3. scheduler
                                    4. criterion
                                See config.py for further details.
        """
        
        super().__init__()
        self.config      = config
        #self.criterion = FocalTversky_loss
        #self.criterion = IoULoss()
        #self.criterion   = SoftDiceLoss(batch_dice=True, smooth=1e-5, do_bg=False)
        self.criterion   = DC_and_topk_loss({'batch_dice':True, 'smooth':1e-5, 'do_bg':False}, {'k':10})
        self.save_hyperparameters()
        self.net = DeepLabV3_3D(num_classes=1, input_channels=1, resnet='resnet18_os16', last_activation='sigmoid')

    def forward(self, scan): # list because of different input sizes
        return self.net(scan)

    def configure_optimizers(self):
        optimizer = init_optimizer(self.net, self.config.optimizer)
        scheduler = init_scheduler(optimizer, self.config.scheduler)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        scans, true_masks = batch
        if torch.isnan(scans.max()):
            print("Nan detected at batch %s" % batch_idx)
            scans[torch.isnan(scans)] = 0
        predicted_masks = self(scans)
        loss = self.criterion(predicted_masks, true_masks)
        result = pl.TrainResult(loss, early_stop_on=loss, checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def validation_step(self, batch, batch_idx):
        scans, true_masks = batch
        if torch.isnan(scans.max()):
            print("Nan detected at batch %s" % batch_idx)
            scans[torch.isnan(scans)] = 0
        predicted_masks = self(scans)
        loss = self.criterion(predicted_masks, true_masks)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def test_step(self, batch, batch_idx):
        scans, true_masks = batch
        predicted_masks = self(scans)
        loss = self.criterion(predicted_masks, true_masks)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        return result