import os
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class ISBIDataset(Dataset):
    """
    Sample Pytorch Dataset. We just overwrite __getitem__ and __len__ methods.
    """
    def __init__(self, img_root=None, label_root=None, train=True,
                 transform=None, target_transform=None):
        
        self.img_root, self.label_root = img_root,  label_root
        self.img_list   = sorted(os.listdir(img_root))
        self.label_list = sorted(os.listdir(label_root))
        self.transform, self.target_transform  = transform, target_transform
        self.train = train
         
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_root, self.img_list[index]))
        if self.transform is not None:
            image = self.transform(image)
        if self.train:
            label = Image.open(os.path.join(self.label_root, self.label_list[index]))
            if self.target_transform is not None:
                label = self.target_transform(label)
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_list)
