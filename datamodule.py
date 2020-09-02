import os
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from dataset import ISBIDataset


class DicomDataModule(pl.LightningDataModule):
    
    """ Generates three dataloaders (train, eval, test) to be used by a Lightning Model. """
    
    def __init__(self, config):
        super().__init__()
        self.config = config # instance of a Dataloader dataclass (see config.py)
        self.transform_image = transforms.Compose([
            # augmentation to add here
            transforms.Grayscale(),
            transforms.ToTensor(),
            # or here 
        ])
        self.transform_label = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    # def prepare_data(self):
    #    return 

    def setup(self, stage=None):
        """ A priori rien de plus que train/val split

        Args:
            stage (str, optional): fit or test. Defaults to None.
        """

        # Assign train/val datasets for use in dataloaders

        train_length = int(0.8*len(os.listdir(self.config.image_rootdir)))
        val_length   = int(0.2*len(os.listdir(self.config.image_rootdir)))

        if stage == 'fit' or stage is None:
            isbi_full = ISBIDataset(img_root = self.config.image_rootdir, label_root = self.config.mask_rootdir,
                                    train = True,
                                    transform = self.transform_image, target_transform = self.transform_label)
            self.isbi_train, self.isbi_val = random_split(isbi_full, [train_length, val_length])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.isbi_test = ISBIDataset(img_root = self.config.image_rootdir,
                                         train = False,
                                         transform = self.transform_image)

    def train_dataloader(self):
        return DataLoader(self.isbi_train, num_workers = self.config.num_workers,
                          batch_size = self.config.train_batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.isbi_val, num_workers = self.config.num_workers,
                          batch_size = self.config.val_batch_size, shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.isbi_test, num_workers = self.config.num_workers,
                          batch_size = self.config.val_batch_size, shuffle = False)