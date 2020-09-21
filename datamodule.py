import os
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from dataset import JFRDataset


class JFRDataModule(pl.LightningDataModule):
    
    """ Generates three dataloaders (train, eval, test) to be used by a Lightning Model. """
    
    def __init__(self, config):
        super().__init__()
        self.config = config # instance of a Dataloader dataclass (see config.py)
        #TODO: add augmentation bellow
        self.transform_scan = None
        self.transform_mask = None

    def setup(self, stage=None):
        """ Basically train/val split

        Args:
            stage (str, optional): fit or test. Defaults to None.
        """
        train_length = int(0.8*len(os.listdir(self.config.scan_rootdir)))
        val_length   = int(0.2*len(os.listdir(self.config.scan_rootdir)))

        if stage == 'fit' or stage is None:
            jfr_full = JFRDataset(scan_root = self.config.scan_rootdir, mask_root = self.config.mask_rootdir,
                                    train = True,
                                    transform = self.transform_scan, target_transform = self.transform_mask)
            self.jfr_train, self.jfr_val = random_split(jfr_full, [train_length, val_length])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.jfr_test = JFRDataset(scan_root = self.config.scan_rootdir,
                                         train = False,
                                         transform = self.transform_scan)

    @staticmethod
    def collate(batch):
        """
        Override `default_collate` https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader

        Reference:
        def default_collate(batch) at https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
        https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
        https://github.com/pytorch/pytorch/issues/1512

        We need our own collate function that wraps things up (scan, mask) because of the varying z shape.

        :param batch: list of tuples (scan, mask)
        :return: 3 elements: list of tensors of scans, list of tensors of masks.
        """
        # list insted of torck stack because of varying length
        # scans = torch.stack([item[0] for item in batch])
        # masks = torch.stack([item[1] for item in batch])
        scans = [item[0] for item in batch]
        masks = [item[1] for item in batch] 
        return scans, masks

    def train_dataloader(self):
        return DataLoader(self.jfr_train, num_workers = self.config.num_workers, collate_fn=self.collate,
                          batch_size = self.config.train_batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.jfr_val, num_workers = self.config.num_workers, collate_fn=self.collate,
                          batch_size = self.config.val_batch_size, shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.jfr_test, num_workers = self.config.num_workers, collate_fn=self.collate,
                          batch_size = self.config.val_batch_size, shuffle = False)