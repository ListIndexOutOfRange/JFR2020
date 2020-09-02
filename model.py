""" Base Model Class: A Lighning Module
    This class implements all the logic code.
    This model class will be the one to be fit by a Trainer
 """

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from test_net import LightningNetwork
from utils.init import init_optimizer, init_scheduler



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
        self.criterion   = torch.nn.BCEWithLogitsLoss()
        self.save_hyperparameters()
        # self.net = LightningNetwork()
        self.net = smp.Unet('resnet18', in_channels=1, classes=1, activation='sigmoid')

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = init_optimizer(self.net, self.config.optimizer)
        scheduler = init_scheduler(optimizer, self.config.scheduler)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.TrainResult(loss)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        return result