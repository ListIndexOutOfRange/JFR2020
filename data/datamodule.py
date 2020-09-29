import os
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from data.dataset import JFRDataset


class JFRDataModule(pl.LightningDataModule):
    
    """ Generates three dataloaders (train, eval, test) to be used by a Lightning Model. """
    
    def __init__(self, config):
        super().__init__()
        self.config = config # instance of a Dataloader dataclass (see config.py)
        self.transform_scan = None
        self.transform_mask = None

    def setup(self, stage=None):
        """ Basically train/val split

        Args:
            stage (str, optional): fit or test. Defaults to None.
        """
        train_length = int(0.8*len(os.listdir(self.config.scan_rootdir)))
        val_length   = int(0.2*len(os.listdir(self.config.scan_rootdir)))+1
        if stage == 'fit' or stage is None:
            jfr_full = JFRDataset(scan_root = self.config.scan_rootdir, mask_root = self.config.mask_rootdir,
                                    train = True,
                                    transform = self.transform_scan, target_transform = self.transform_mask)
            print()
            print(80*'-')
            print(len(jfr_full))
            print(train_length)
            print(val_length)
            print(80*'-')
            self.jfr_train, self.jfr_val = random_split(jfr_full, [train_length, val_length])


        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.jfr_test = JFRDataset(scan_root = self.config.scan_rootdir,
                                         train = False,
                                         transform = self.transform_scan)

    def train_dataloader(self):
        return DataLoader(self.jfr_train, num_workers = self.config.num_workers,
                          batch_size = self.config.train_batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.jfr_val, num_workers = self.config.num_workers, 
                          batch_size = self.config.val_batch_size, shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.jfr_test, num_workers = self.config.num_workers,
                          batch_size = self.config.val_batch_size, shuffle = False)