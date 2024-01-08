import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', transform=None):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.transform = transform

        self.images_dir = os.path.join(self.dataset_dir, self.mode, 'images')
        self.masks_dir = os.path.join(self.dataset_dir, self.mode, 'masks')

        self.image_filenames = os.listdir(self.images_dir)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.image_filenames[index])
        mask_path = os.path.join(self.masks_dir, self.image_filenames[index])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def __len__(self):
        return len(self.image_filenames)

def get_validation_augmentation():
    test_transform = [
        # transforms.RandomCrop((256, 256)),
        transforms.Resize((352, 352)),
        
        transforms.ToTensor(),
    ]
    return transforms.Compose(test_transform)

def to_tensor(x):
    return torch.from_numpy(x.transpose(2, 0, 1)).float()  # Add the 'return' statement
