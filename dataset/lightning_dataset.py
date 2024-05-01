import os
import glob
import random
from .augmentation import get_augmentation 
from .torch_dataset import BrainDataset
from torch.utils.data import DataLoader
import lightning as L

class BrainDataModule(L.LightningDataModule):
    def __init__(
            self, 
            data_directory: str, 
            split_size: float = 0.8,
            batch_size: int = 32,
        ):
        super().__init__()
        self.data_directory = data_directory
        self.split_size = split_size
        self.batch_size = batch_size

        self.transform = get_augmentation()

    def setup(self, stage: str):

        mask = os.path.join(self.data_directory, "*", "*mask.tif")
        mask = glob.glob(mask)
        mask = random.sample(mask, len(mask)) 
        image = list(map(lambda x: x.replace('_mask', ''), mask))

        #split 
        split_size = int(self.split_size*len(image)) 
        train_image, train_mask = image[:split_size], mask[:split_size]
        val_image, val_mask = image[split_size:], mask[split_size:] 
        
        augmentation = get_augmentation()
        self.train_dataset = BrainDataset(train_image, train_mask, augmentation['train']) 
        self.val_dataset = BrainDataset(val_image, val_mask, augmentation['val'])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)