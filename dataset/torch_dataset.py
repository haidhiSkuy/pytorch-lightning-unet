import cv2
import torch
from torch.utils.data import Dataset 


class BrainDataset(Dataset): 
    def __init__(self, image_list, mask_list, transform): 
        self.image_list = image_list 
        self.mask_list = mask_list
        self.transform = transform 

    def __len__(self): 
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index]) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        mask = cv2.imread(self.mask_list[index], cv2.IMREAD_UNCHANGED) 

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"] / 255.0
            mask = transformed["mask"] / 255.0
        
        return image, mask.to(torch.float32)

