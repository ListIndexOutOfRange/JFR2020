import os
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class JFRDataset(Dataset):
    """
    Sample Pytorch Dataset. We just overwrite __getitem__ and __len__ methods.
    """
    def __init__(self, scan_root=None, mask_root=None, train=True,
                 transform=None, target_transform=None):
        
        self.scan_root, self.mask_root = scan_root,  mask_root
        self.scan_list = sorted(os.listdir(scan_root))
        self.mask_list = sorted(os.listdir(mask_root))
        self.transform, self.target_transform  = transform, target_transform
        self.train = train
         
    def __getitem__(self, index):
        scan = np.load(os.path.join(self.scan_root, self.scan_list[index]))
        scan = scan.astype(np.float32)
        if self.transform is not None:
            scan = self.transform(scan)
        if self.train:
            mask = np.load(os.path.join(self.mask_root, self.mask_list[index]))
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return scan, mask
        else:
            return scan

    def __len__(self):
        return len(self.scan_list)
