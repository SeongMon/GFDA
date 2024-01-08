from .CustomDataset import CustomDataset
from CustomDataset import get_validation_augmentation

dir_CVC = 'dataset/CVC-ClinicDB'
dir_Kvasir = 'dataset/Kvasir'
dir_BKAI = 'dataset/BKAI-IGH-NeoPolyp'

train_dataset_CVC = CustomDataset(
    dir_CVC,transform=get_validation_augmentation()
)
test_dataset_CVC = CustomDataset(
    dir_CVC, mode='test',transform=get_validation_augmentation()
)
train_dataset_Kvasir = CustomDataset(
    dir_Kvasir,transform=get_validation_augmentation()
)
test_dataset_Kvasir = CustomDataset(
    dir_Kvasir, mode='test',transform=get_validation_augmentation()
)
train_dataset_BKAI = CustomDataset(
    dir_BKAI,transform=get_validation_augmentation()
)
test_dataset_BKAI = CustomDataset(
    dir_BKAI, mode='test',transform=get_validation_augmentation()
)