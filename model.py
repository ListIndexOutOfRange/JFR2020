""" Base Model Class: A Lighning Module
    This class implements all the logic code.
    This model class will be the one to be fit by a Trainer
 """

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
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

    # TODO handle list as input instead of torch Tensor: 
    #   1. forward          : DONE
    #   2. training_step    : |
    #   3. validation_step  : |-> need to work on the loss computation
    #   4. test_step        : |

    def forward(self, scan_list): # list because of different input sizes
        predicted_masks = []
        for scan in scan_list:
            predicted_mask.append(self.net(scan))
        return predicted_masks

    def configure_optimizers(self):
        optimizer = init_optimizer(self.net, self.config.optimizer)
        scheduler = init_scheduler(optimizer, self.config.scheduler)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        scans, true_masks = batch
        predicted_masks = self(scans)
        loss = self.criterion(predicted_masks, true_masks)
        result = pl.TrainResult(loss, early_stop_on=loss, checkpoint_on=loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self, batch, batch_idx):
        scans, true_masks = batch
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